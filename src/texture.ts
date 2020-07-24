// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {expectContents} from './fixture';
import * as tex_util from './tex_util';

export class TextureOp {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  times: [];
  resultMatrixTexture: GPUTexture;
  resultMatrixTextureSize: number;
  shape: Uint32Array;
  computePipeline: any;
  bindGroup: any;
  format: GPUTextureFormat;
  kBytesPerTexel: number;
  constructor(
      device: GPUDevice, glslang: Glslang, format: GPUTextureFormat,
      kBytesPerTexel: number) {
    this.device = device;
    this.queue = device.defaultQueue;
    this.glslang = glslang;
    this.commandQueue = [];
    this.format = format;
    this.kBytesPerTexel = kBytesPerTexel;
  }

  createCopyForMapRead(src: any, size: any) {
    const dst = this.device.createBuffer(
        {size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    const c = this.device.createCommandEncoder();
    c.copyBufferToBuffer(src, 0, dst, 0, size);
    this.device.defaultQueue.submit([c.finish()]);
    return dst;
  }

  async checkContents(src: any, expected: any) {
    const exp = new Uint8Array(
        expected.buffer, expected.byteOffset, expected.byteLength);
    const dst = this.createCopyForMapRead(src, expected.buffer.byteLength);
    console.log(exp);
    console.log(dst);
    const actual = new Uint8Array((await dst.mapReadAsync()));
    const result = expectContents(actual, exp);
    console.log(result);
    dst.destroy();
  }

  now(): number {
    return performance.now();
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    return new Float32Array(arrayBuffer);
  }

  private copyFromHostBufferToDeviceTexture(
      src: GPUBuffer, width: number, height: number) {
    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            width, height, this.format);

    const texture = this.device.createTexture({
      size: {width: widthTex, height: heightTex, depth: 1},
      format: this.format,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.STORAGE
    });
    const encoder = this.device.createCommandEncoder();
    // TODO: fix the width height.
    // copyBufferToTexture(source, destination, copySize).
    const bytesPerRow = tex_util.getBytesPerRow(widthTex, this.kBytesPerTexel);
    // console.log('bytesPerRow=' + bytesPerRow);
    encoder.copyBufferToTexture(
        {buffer: src, bytesPerRow: bytesPerRow},
        {texture: texture, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: widthTex, height: heightTex, depth: 1});
    this.device.defaultQueue.submit([encoder.finish()]);
    return texture;
  }

  // From: Dawn:ComputeTextureCopyBufferSize
  // TODO: Make this works with different input size
  getBufferSize() {
    const blockHeight = 1;
    const blockWidth = 1;

    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            this.shape[0], this.shape[1], this.format);

    const bytesPerRow = tex_util.getBytesPerRow(widthTex, this.kBytesPerTexel);

    const sliceSize = bytesPerRow * (heightTex / blockHeight - 1) +
        (widthTex / blockWidth) * this.kBytesPerTexel;
    return sliceSize;
  }

  compile(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      computeShaderCode: any) {
    this.shape = shape;
    // console.log('B2T this.getBufferSize()=' + this.getBufferSize());
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] =
        this.device.createBufferMapped({
          size: this.getBufferSize(),  // (firstMatrix as
                                       // Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST
        });
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
    const gpuTextureFirstMatrix = this.copyFromHostBufferToDeviceTexture(
        gpuBufferFirstMatrix, this.shape[0], this.shape[1]);

    const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] =
        this.device.createBufferMapped({
          size: this.getBufferSize(),  //(secondMatrix as
                                       // Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST
        });
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();

    const gpuTextureSecondMatrix = this.copyFromHostBufferToDeviceTexture(
        gpuBufferSecondMatrix, this.shape[2], this.shape[3]);

    // Result Matrix.
    this.resultMatrixTextureSize =
        Float32Array.BYTES_PER_ELEMENT * (shape[4] * shape[5]);

    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            this.shape[4], this.shape[5], this.format);
    this.resultMatrixTexture = this.device.createTexture({
      size: {width: widthTex, height: heightTex, depth: 1},
      format: this.format,
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE
    });

    // This works.
    const [shapeBuffer, shapeMapping] = this.device.createBufferMapped({
      size: shape.byteLength,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Uint32Array(shapeMapping).set(shape);
    shapeBuffer.unmap();

    // This works too.
    /*
     const shapeBuffer = this.uploadToGPUBuffer(
         shape, shape.byteLength,
         GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC |
             GPUBufferUsage.COPY_DST);
     */

    return this.createLayout(
        gpuTextureFirstMatrix, gpuTextureSecondMatrix, shapeBuffer,
        computeShaderCode);
  }

  private createLayout(
      gpuTextureFirstMatrix: GPUTexture, gpuTextureSecondMatrix: GPUTexture,
      shapeBuffer: GPUBuffer, computeShaderCode: any) {
    // Bind group layout and bind group
    // TODO: currently this doesn't support read write storage.
    // https://gpuweb.github.io/gpuweb/#enumdef-gpubindingtype
    // Use old layout:
    // Bind group layout and bind group

    // Old layout.
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          type: 'uniform-buffer'
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          type: 'writeonly-storage-texture',
          storageTextureFormat: this.format
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          type: 'readonly-storage-texture',
          storageTextureFormat: this.format
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          type: 'readonly-storage-texture',
          storageTextureFormat: this.format
        }
      ]
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: shapeBuffer}},
        {binding: 1, resource: this.resultMatrixTexture.createView()},
        {binding: 2, resource: gpuTextureFirstMatrix.createView()},
        {binding: 3, resource: gpuTextureSecondMatrix.createView()},
      ]
    });
    // Old layout end.

    // Pipeline setup
    const result =
        this.glslang.compileGLSLZeroCopy(computeShaderCode, 'compute', false);
    if (result.data.length === 0) {
      throw new Error('Shader compilation failed');
    }
    this.computePipeline = this.device.createComputePipeline({
      // For new layout, remove this line.
      layout: this.device.createPipelineLayout(
          {bindGroupLayouts: [bindGroupLayout]}),
      computeStage: {
        module: this.device.createShaderModule({code: result.data}),
        entryPoint: 'main'
      }
    });

    /* New layout
    const bindGroup = this.device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {binding: 0, resource: {buffer: shapeBuffer}},
        {binding: 1, resource: this.resultMatrixTexture.createView()},
        {binding: 2, resource: gpuTextureFirstMatrix.createView()},
        {binding: 3, resource: gpuTextureSecondMatrix.createView()},
      ]
    });
        */

    return;
  }

  // TODO: Float32Array is bad. And buffer is bad.
  async compileAndRun(workGroupSize: [number, number, number]) {
    // TODO: figure out how to return non const two values.
    await this.dispatchAndSubmitWithFence(
        this.computePipeline, this.bindGroup, this.shape[0], this.shape[1],
        workGroupSize);

    return true;
  }

  compileAndRunSync(workGroupSize: [number, number, number]) {
    // TODO: figure out how to return non const two values.
    this.dispatchAndSubmit(
        this.computePipeline, this.bindGroup, this.shape[0], this.shape[1],
        workGroupSize);

    return true;
  }

  private dispatchAndSubmit(
      computePipeline: any, bindGroup: any, dispatchX: number,
      dispatchY: number, workGroupSize: [number, number, number]) {
    // Commands submission.
    const commandEncoder = this.device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    // console.log('dispatchX Y =' + dispatchX + ', ' + dispatchY);
    // console.log('dispatchX Y/WG =' + dispatchX / workGroupSize[0] + ', ' +
    // dispatchY / workGroupSize[1]);
    passEncoder.dispatch(
        dispatchX / workGroupSize[0], dispatchY / workGroupSize[1]);
    passEncoder.endPass();
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.device.defaultQueue.submit([gpuCommands]);
  }

  private async dispatchAndSubmitWithFence(
      computePipeline: any, bindGroup: any, dispatchX: number,
      dispatchY: number, workGroupSize: [number, number, number]) {
    const start = this.now();
    // Commands submission.
    this.dispatchAndSubmit(
        this.computePipeline, this.bindGroup, this.shape[0], this.shape[1],
        workGroupSize);
    const fence = this.queue.createFence();
    this.queue.signal(fence, 1);
    await fence.onCompletion(1);
    console.log((this.now() - start).toFixed(2));
  }


  async getBufferData() {
    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = this.device.createBuffer({
      size: this.getBufferSize(),  // Float32Array.BYTES_PER_ELEMENT *
                                   // (this.shape[0] * this.shape[1]),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // console.log('T2B this.getBufferSize()=' + this.getBufferSize());
    // Commands submission.
    const commandEncoder = this.device.createCommandEncoder();

    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            this.shape[0], this.shape[1], this.format);
    // console.log('widthTex = ' + widthTex + '; heightTex = ' + heightTex);
    const bytesPerRow = tex_util.getBytesPerRow(widthTex, this.kBytesPerTexel);
    // Encode commands for copying texture to buffer.
    commandEncoder.copyTextureToBuffer(
        {
          texture: this.resultMatrixTexture,
          mipLevel: 0,
          origin: {x: 0, y: 0, z: 0}
        },
        {buffer: gpuReadBuffer, bytesPerRow: bytesPerRow},
        {width: widthTex, height: heightTex, depth: 1});
    // Submit GPU commands.
    this.device.defaultQueue.submit([commandEncoder.finish()]);
    // t.expectContents(dst, data);
    /*
    const fence = this.queue.createFence();
    this.queue.signal(fence, 2);
    await fence.onCompletion(2);
    */
    // Read buffer.
    const arrayBuffer = await gpuReadBuffer.mapReadAsync();
    return arrayBuffer;
  }
}

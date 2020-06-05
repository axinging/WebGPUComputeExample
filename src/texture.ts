// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {expectContents} from './fixture';

export class TextureOp {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  times: [];
  resultMatrixTexture: GPUTexture;
  resultMatrixTextureSize: number;
  shape: Int32Array;
  constructor(device: GPUDevice, glslang: Glslang) {
    this.device = device;
    this.queue = device.defaultQueue;
    this.glslang = glslang;
    this.commandQueue = [];
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

  createCopyForMapRead2(size: any) {
    const data = new Uint32Array([0x01020304]);
    // The HOST buffer.
    const [src, map] = this.device.createBufferMapped({
      size: 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST |
          GPUBufferUsage.MAP_READ,
    });
    new Uint32Array(map).set(data);
    src.unmap();
    // The Device buffer.
    const dst = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const c = this.device.createCommandEncoder();
    c.copyBufferToBuffer(src, 0, dst, 0, size);
    this.device.defaultQueue.submit([c.finish()]);
    return dst;
  }

  private copyFromHostBufferToDeviceTexture(src: GPUBuffer) {
    const texture = this.device.createTexture({
      size: {width: 2, height: 2, depth: 1},
      format: 'rgba32float',
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
    });
    const encoder = this.device.createCommandEncoder();
    // TODO: fix the width height.
    encoder.copyBufferToTexture(
        {buffer: src, bytesPerRow: 256},
        {texture: texture, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: 2, height: 2, depth: 1});
    this.device.defaultQueue.submit([encoder.finish()]);
    return texture;
  }

  private compile(
      firstMatrix: Float32Array, secondMatrix: Float32Array, shape: Int32Array,
      computeShaderCode: any) {
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] =
        this.device.createBufferMapped({
          size: (firstMatrix as Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE
        });
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
    const gpuTextureFirstMatrix =
        this.copyFromHostBufferToDeviceTexture(gpuBufferFirstMatrix);

    const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] =
        this.device.createBufferMapped({
          size: (secondMatrix as Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE
        });
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();

    const gpuTextureSecondMatrix =
        this.copyFromHostBufferToDeviceTexture(gpuBufferSecondMatrix);
    // Result Matrix
    this.resultMatrixTextureSize =
        Float32Array.BYTES_PER_ELEMENT * (shape[4] * shape[5]);

    this.resultMatrixTexture = this.device.createTexture({
      size: {width: 1, height: 1, depth: 1},
      format: 'rgba32float',
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
    });

    // This works.
    const [shapeBuffer, shapeMapping] = this.device.createBufferMapped({
      size: shape.byteLength,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Int32Array(shapeMapping).set(shape);
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
          storageTextureFormat: 'rgba32float'
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          type: 'readonly-storage-texture',
          storageTextureFormat: 'rgba32float'
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          type: 'readonly-storage-texture',
          storageTextureFormat: 'rgba32float'
        }
      ]
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: shapeBuffer}},
        {binding: 1, resource: this.resultMatrixTexture.createView()},
        {binding: 2, resource: gpuTextureFirstMatrix.createView()},
        {binding: 3, resource: gpuTextureSecondMatrix.createView()},

      ]
    });

    // Pipeline setup
    const result =
        this.glslang.compileGLSLZeroCopy(computeShaderCode, 'compute', false);
    if (result.data.length === 0) {
      throw new Error('Shader compilation failed');
    }
    const computePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout(
          {bindGroupLayouts: [bindGroupLayout]}),

      computeStage: {
        module: this.device.createShaderModule({code: result.data}),
        entryPoint: 'main'
      }
    });
    return {
      computePipeline, bindGroup
    }
  }

  // TODO: Float32Array is bad. And buffer is bad.
  async compileAndRun(
      firstMatrix: Float32Array, secondMatrix: Float32Array, shape: Int32Array,
      computeShaderCode: any, mode: number) {
    // TODO: figure out how to return non const two values.
    this.shape = shape;
    const {computePipeline, bindGroup} =
        this.compile(firstMatrix, secondMatrix, shape, computeShaderCode);
    await this.dispatchAndSubmit(
        computePipeline, bindGroup, shape[0], shape[1]);

    return true;
  }

  private async dispatchAndSubmit(
      computePipeline: any, bindGroup: any, dispatchX: number,
      dispatchY: number) {
    const start = this.now();
    // Commands submission
    const commandEncoder = this.device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    console.log(dispatchX + '+' + dispatchY);
    passEncoder.dispatch(dispatchX, dispatchY);
    passEncoder.endPass();
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.device.defaultQueue.submit([gpuCommands]);
    const fence = this.queue.createFence();
    this.queue.signal(fence, 1);
    await fence.onCompletion(1);
    console.log('Fence time: ' + (this.now() - start));
  }

  async getBufferData() {
    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = this.device.createBuffer({
      size: this.resultMatrixTextureSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Commands submission
    const commandEncoder = this.device.createCommandEncoder();
    // Encode commands for copying texture to buffer.
    commandEncoder.copyTextureToBuffer(
        {
          texture: this.resultMatrixTexture,
          mipLevel: 0,
          origin: {x: 0, y: 0, z: 0}
        },
        {buffer: gpuReadBuffer, bytesPerRow: 256},
        {width: 2, height: 2, depth: 1});
    // Submit GPU commands.
    this.device.defaultQueue.submit([commandEncoder.finish()]);
    // t.expectContents(dst, data);
    const fence = this.queue.createFence();
    this.queue.signal(fence, 2);
    await fence.onCompletion(2);
    // Read buffer.
    const arrayBuffer = await gpuReadBuffer.mapReadAsync();
    return arrayBuffer;
  }
}

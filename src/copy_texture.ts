// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import * as tex_util from './tex_util';

export class CopyTextureOp {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  times: [];
  gpuTextureFirstMatrix: GPUTexture;
  shape: Uint32Array;
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

  now(): number {
    return performance.now();
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
    console.log(
        'w = ' + width + ', h = ' + height + '; tex w = ' + widthTex +
        ', h = ' + heightTex);
    const encoder = this.device.createCommandEncoder();
    // TODO: fix the width height.
    // copyBufferToTexture(source, destination, copySize).
    const bytesPerRow = tex_util.getBytesPerRow(widthTex, this.kBytesPerTexel);
    console.log('bytesPerRow=' + bytesPerRow);
    encoder.copyBufferToTexture(
        {buffer: src, bytesPerRow: bytesPerRow},
        {texture: texture, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: widthTex, height: heightTex, depth: 1});
    this.device.defaultQueue.submit([encoder.finish()]);
    return texture;
  }

  // From: Dawn:ComputeTextureCopyBufferSize
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

  private compile(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      computeShaderCode: any) {
    console.log('B2T this.getBufferSize()=' + this.getBufferSize());
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] =
        this.device.createBufferMapped({
          size: this.getBufferSize(),  // (firstMatrix as
                                       // Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST
        });
    console.log(
        'firstMatrix as Float32Array).byteLength=' +
        (firstMatrix as Float32Array).byteLength);
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
    this.gpuTextureFirstMatrix = this.copyFromHostBufferToDeviceTexture(
        gpuBufferFirstMatrix, this.shape[0], this.shape[1]);
    return;
  }

  async execute(
      firstMatrix: Float32Array|Uint32Array, shape: Uint32Array, mode = 0) {
    const result = await this.compileAndRun(firstMatrix, null, shape, '', mode);
    return result;
  }

  // TODO: Float32Array is bad. And buffer is bad.
  async compileAndRun(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      computeShaderCode: any, mode: number) {
    // TODO: figure out how to return non const two values.
    this.shape = shape;
    this.compile(firstMatrix, secondMatrix, shape, computeShaderCode);
    return true;
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    // TODO: why this needs to be float.
    return new Float32Array(arrayBuffer);
  }

  async getBufferData() {
    // Get a GPU buffer for reading in an unmapped state.

    const gpuReadBuffer = this.device.createBuffer({
      size: this.getBufferSize(),  // Float32Array.BYTES_PER_ELEMENT *
                                   // (this.shape[0] * this.shape[1]),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    console.log('T2B this.getBufferSize()=' + this.getBufferSize());
    // Commands submission.
    const commandEncoder = this.device.createCommandEncoder();

    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            this.shape[0], this.shape[1], this.format);
    console.log('widthTex = ' + widthTex + '; heightTex = ' + heightTex);
    const bytesPerRow = tex_util.getBytesPerRow(widthTex, this.kBytesPerTexel);
    // Encode commands for copying texture to buffer.
    commandEncoder.copyTextureToBuffer(
        {
          texture: this.gpuTextureFirstMatrix,
          mipLevel: 0,
          origin: {x: 0, y: 0, z: 0}
        },
        {buffer: gpuReadBuffer, bytesPerRow: bytesPerRow},
        {width: widthTex, height: heightTex, depth: 1});

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

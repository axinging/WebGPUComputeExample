// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import * as tex_util from './tex_util';

export class CopyTextureRGBA8Op {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  times: [];
  gpuTextureFirstMatrix: GPUTexture;
  shape: Int32Array;
  format: GPUTextureFormat;
  constructor(device: GPUDevice, glslang: Glslang) {
    this.device = device;
    this.queue = device.defaultQueue;
    this.glslang = glslang;
    this.commandQueue = [];
    this.format = 'rgba8uint';
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
        'w =' + width + ', h=' + height + '; tex w=' + widthTex +
        ', h= ' + heightTex);
    const encoder = this.device.createCommandEncoder();
    // TODO: fix the width height.
    // copyBufferToTexture(source, destination, copySize).
    encoder.copyBufferToTexture(
        {buffer: src, bytesPerRow: 256},
        {texture: texture, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: widthTex, height: heightTex, depth: 1});
    this.device.defaultQueue.submit([encoder.finish()]);
    return texture;
  }

  private compile(
      firstMatrix: Float32Array, secondMatrix: Float32Array, shape: Int32Array,
      computeShaderCode: any) {
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] =
        this.device.createBufferMapped({
          size: (firstMatrix as Float32Array).byteLength,
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

  createArray(w: number, h: number) {
    let matrix = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      matrix[i] = i;
    }
    return matrix;
  }

  async execute(mode = 0) {
    // First Matrix
    const firstMatrixSize = [4, 8];
    const firstMatrix = this.createArray(4, 8);
    const shape = new Int32Array([firstMatrixSize[0], firstMatrixSize[1]]);
    const result = await this.compileAndRun(firstMatrix, null, shape, '', mode);
    return result;
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    return new Float32Array(arrayBuffer);
  }

  // TODO: Float32Array is bad. And buffer is bad.
  async compileAndRun(
      firstMatrix: Float32Array, secondMatrix: Float32Array, shape: Int32Array,
      computeShaderCode: any, mode: number) {
    // TODO: figure out how to return non const two values.
    this.shape = shape;
    this.compile(firstMatrix, secondMatrix, shape, computeShaderCode);
    return true;
  }

  async getBufferData() {
    // Get a GPU buffer for reading in an unmapped state.

    const gpuReadBuffer = this.device.createBuffer({
      size: Float32Array.BYTES_PER_ELEMENT * (this.shape[0] * this.shape[1]),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Commands submission.
    const commandEncoder = this.device.createCommandEncoder();

    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            this.shape[0], this.shape[1], this.format);

    // Encode commands for copying texture to buffer.
    commandEncoder.copyTextureToBuffer(
        {
          texture: this.gpuTextureFirstMatrix,
          mipLevel: 0,
          origin: {x: 0, y: 0, z: 0}
        },
        {buffer: gpuReadBuffer, bytesPerRow: 256},
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

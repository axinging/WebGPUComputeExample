
// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {expectContents} from './fixture';

export class TextureOp {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  constructor(device: GPUDevice, glslang: Glslang) {
    this.device = device;
    this.queue = device.defaultQueue;
    this.glslang = glslang;
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
    // eventualAsyncExpectation(async (niceStack: {message: any;}) => {
    const actual = new Uint8Array((await dst.mapReadAsync()));
    const result = expectContents(actual, exp);
    console.log(result);
    dst.destroy();
  }

  async compileAndRun() {
    const data = new Uint32Array([0x01020304]);

    // Staging buffer.
    const [src, map] = this.device.createBufferMapped({
      size: 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    new Uint32Array(map).set(data);
    src.unmap();

    // Device buffer.
    const dst = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const midDesc: GPUTextureDescriptor = {
      size: {width: 1, height: 1, depth: 1},
      format: 'rgba8uint',
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    };
    const mid1 = this.device.createTexture(midDesc);
    const mid2 = this.device.createTexture(midDesc);

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToTexture(
        {buffer: src, bytesPerRow: 256},
        {texture: mid1, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: 1, height: 1, depth: 1});
    encoder.copyTextureToTexture(
        {texture: mid1, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {texture: mid2, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: 1, height: 1, depth: 1});
    encoder.copyTextureToBuffer(
        {texture: mid2, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {buffer: dst, bytesPerRow: 256}, {width: 1, height: 1, depth: 1});
    this.device.defaultQueue.submit([encoder.finish()]);
    await this.checkContents(dst, data);
    return true;
  }
}

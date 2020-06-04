
// import glslangInit from '@webgpu/glslang/dist/web-devel-onefile/glslang';

import {expectContents} from './fixture';


function createCopyForMapRead(device: any, src: any, size: any) {
  const dst = device.createBuffer(
      {size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const c = device.createCommandEncoder();
  c.copyBufferToBuffer(src, 0, dst, 0, size);
  device.defaultQueue.submit([c.finish()]);
  return dst;
}  // TODO: add an expectContents for textures, which logs data: uris on failure


async function checkContents(device: any, src: any, expected: any) {
  const exp =
      new Uint8Array(expected.buffer, expected.byteOffset, expected.byteLength);
  const dst = createCopyForMapRead(device, src, expected.buffer.byteLength);
  console.log(exp);
  console.log(dst);
  // eventualAsyncExpectation(async (niceStack: {message: any;}) => {
  const actual = new Uint8Array((await dst.mapReadAsync()));
  const result = expectContents(actual, exp);
  console.log(result);
  dst.destroy();
}



export async function texture_init() {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags / #enable - unsafe - webgpuflag.');
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  // const glslang = await glslangInit();
  // console.log(glslang);

  const data = new Uint32Array([0x01020304]);

  const [src, map] = device.createBufferMapped({
    size: 4,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  new Uint32Array(map).set(data);
  src.unmap();

  const dst = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const midDesc: GPUTextureDescriptor = {
    size: {width: 1, height: 1, depth: 1},
    format: 'rgba8uint',
    usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
  };
  const mid1 = device.createTexture(midDesc);
  const mid2 = device.createTexture(midDesc);

  const encoder = device.createCommandEncoder();
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
  device.defaultQueue.submit([encoder.finish()]);
  await checkContents(device, dst, data);
  return true;
}

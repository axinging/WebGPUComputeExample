import * as compute  from '@webgpu/compute';
//import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
//import glslangInit from '@webgpu/glslang/dist/web-devel-onefile/glslang';

(async () => {
  if (!navigator.gpu) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const glslang = await glslangInit();
  const texture = new compute.TextureOp(device, glslang);
  console.log(await texture.compileAndRun());
  const buffer = new compute.BufferOp(device, glslang);
  console.log(await buffer.compileAndRun());
})();

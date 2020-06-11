import * as compute  from '@webgpu/compute';
//import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

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

  const addTextureOp = new compute.CopyTextureOp(device, glslang);
  const loop = 1;
  for (var i = 0; i < loop; i++) {
    await addTextureOp.execute();
    //console.log("not staging: "+await addTextureOp.data());
  }
/*
  const buffer = new compute.AddBufferOp(device, glslang);
  for (var i = 0; i < loop; i++) {
    await buffer.execute()
    console.log("not staging: "+await buffer.data());
  }
  for (var i = 0; i < loop; i++) {
    await buffer.execute(1)
    console.log("staging: "+await buffer.data());
  }
*/
})();

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

  const copyTextureOp = new compute.CopyTextureRGBA8Op(device, glslang);
  const loop = 1;
  for (var i = 0; i < loop; i++) {
    await copyTextureOp.execute();
    console.log("Texture rgba8 not staging: "+await copyTextureOp.data());
  }

  {
    const copyTextureOp = new compute.CopyTextureRGBA32FOp(device, glslang);
    const loop = 1;
    for (var i = 0; i < loop; i++) {
      await copyTextureOp.execute();
      console.log("Texture rgba32f not staging: "+await copyTextureOp.data());
    }
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

import * as compute  from '@webgpu/compute';
//import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = i;
  }
  return matrix;
}

function createUint32Array(w, h) {
  let matrix = new Uint32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = i;
  }
  return matrix;
}

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
  /*
  {
    // First Matrix
    // Works: [256, 128];
    // Not work: [259, 127]; [7, 3];
    const firstMatrixSize = [15, 8];
    const firstMatrix =
      createUint32Array(firstMatrixSize[0], firstMatrixSize[1]);
    const shape = new Uint32Array([firstMatrixSize[0], firstMatrixSize[1]]);
    const copyTextureOp = new compute.CopyTextureOp(device, glslang, 'rgba8uint', 4);
    const loop = 1;
    for (var i = 0; i < loop; i++) {
      await copyTextureOp.execute(firstMatrix, shape);
      console.log("Texture rgba8 not staging: "+await copyTextureOp.data());
    }
  }
  */

  {
    // First Matrix
    // Works: [16, 8]; [32, 16];
    // Not work: [17, 9];[15, 7]; [15, 8];
    const firstMatrixSize = [15, 8];
    const firstMatrix =
      createFloat32Array(firstMatrixSize[0], firstMatrixSize[1]);
    const shape = new Uint32Array([firstMatrixSize[0], firstMatrixSize[1]]);

    const copyTextureOp = new compute.CopyTextureOp(device, glslang, 'rgba32float', 16);
    const loop = 1;
    for (var i = 0; i < loop; i++) {
      await copyTextureOp.execute(firstMatrix, shape);
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

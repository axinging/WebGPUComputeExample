import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] =
        Math.random() * 100;  // tf.randomUniform(shape, 0, 2.5);//0.01*i;
  }
  return matrix;
}

function compareAddFloat32Array(result, firstMatrix, secondMatrix, w, h) {
  // let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(result[i] - (firstMatrix[i] + secondMatrix[i])) > 0.01)
      return i;
  }
  return -1;
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
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const enableTimeStamp = false;
  const device = await adapter.requestDevice();
  const glslang = await glslangInit();
  const oldLog = console.log;
  let kernels = new Array();
  compute.startLog(kernels, oldLog);
  {
    const loop = 50;  
    for (var i = 0; i < loop; i++) {
      // First Matrix.
      const size_x = 4096;
      const size_y = 256;
      const firstMatrixSize = [size_x, size_y];
      const firstMatrix = createFloat32Array(size_x, size_y);
      // Second Matrix.
      const secondMatrixSize = [size_x, size_y];
      const secondMatrix = createFloat32Array(size_x, size_y);
      const shape = new Uint32Array([
        firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
        secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
      ]);

      const addBufferOP = new compute.AddBufferOp(device, glslang);
      await addBufferOP.execute(firstMatrix, secondMatrix, shape);

      const failItem = compareAddFloat32Array(
          await addBufferOP.data(), firstMatrix, secondMatrix, size_x, size_y);
      if (failItem != -1)
        console.log('compareAddFloat32Array Item ' + failItem + ' Fail');
    }
  }

  {
    const loop = 50;
    for (var i = 0; i < loop; i++) {
      // First Matrix.
      const size_x = 4096;
      const size_y = 256;
      const firstMatrixSize = [size_x, size_y];
      const firstMatrix = createFloat32Array(size_x, size_y);
      // Second Matrix.
      const secondMatrixSize = [size_x, size_y];
      const secondMatrix = createFloat32Array(size_x, size_y);
      const shape = new Uint32Array([
        firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
        secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
      ]);

      const addTextureOP =
          new compute.AddTextureOp(device, glslang, 'rgba32f', 16);

      await addTextureOP.execute(firstMatrix, secondMatrix, shape);
      const failItem = compareAddFloat32Array(
          await addTextureOP.data(), firstMatrix, secondMatrix, size_x, size_y);
      if (failItem != -1)
        console.log('compareAddFloat32Array Item ' + failItem + ' Fail');
    }
  }
  compute.endLog(kernels, oldLog);
})();

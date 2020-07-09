import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
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
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const enableTimeStamp = false;
  const device = await adapter.requestDevice();
  const glslang = await glslangInit();
  /*
  const device = await adapter.requestDevice({
    extensions: ['timestamp-query'],
  });
  */
 {
  // First Matrix.
  const size = 32;
  const firstMatrixSize = [size , size];
  const firstMatrix = createFloat32Array(size, size);
  // Second Matrix.
  const secondMatrixSize = [size, size];
  const secondMatrix = createFloat32Array(size, size);
  const shape = new Uint32Array([
    firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
    secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
  ]);
  const addBufferOP = new compute.AddBufferOp(device, glslang);
  const loop = 1;
  for (var i = 0; i < loop; i++) {
    await addBufferOP.execute(firstMatrix, secondMatrix, shape);
    console.log('addBufferOP: ' + await addBufferOP.data());
  }
}

 {
  // First Matrix.
  const size = 32;
  const firstMatrixSize = [size , size];
  const firstMatrix = createFloat32Array(size, size);
  // Second Matrix.
  const secondMatrixSize = [size, size];
  const secondMatrix = createFloat32Array(size, size);
  const shape = new Uint32Array([
    firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
    secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
  ]);
  const addTextureOP = new compute.AddTextureOp(device, glslang, 'rgba32f', 16);
  const loop = 1;
  for (var i = 0; i < loop; i++) {
    await addTextureOP.execute(firstMatrix, secondMatrix, shape);
    console.log("addTextureOP: "+await addTextureOP.data());
   }
 }

 {
  // First Matrix.
  const size = 32;
  const firstMatrixSize = [size , size];
  const firstMatrix = createFloat32Array(size, size);
  // Second Matrix.
  const secondMatrixSize = [size, size];
  const secondMatrix = createFloat32Array(size, size);
  const shape = new Uint32Array([
    firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
    secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
  ]);
  const matmulBufferOp = new compute.MatmulBufferOp(device, glslang);
  const loop = 1;
  for (var i = 0; i < loop; i++) {
    await matmulBufferOp.execute(firstMatrix, secondMatrix, shape);
    console.log("matmulBufferOp: "+await matmulBufferOp.data());
   }
 }

  {
    // First Matrix.
    const size = 32;
    const firstMatrixSize = [size , size];
    const firstMatrix = createFloat32Array(size, size);
    // Second Matrix.
    const secondMatrixSize = [size, size];
    const secondMatrix = createFloat32Array(size, size);
    const shape = new Uint32Array([
      firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
      secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
    ]);
    const matmulTextureOp = new compute.MatmulTextureOp(device, glslang, 'rgba32f', 16);
    const loop = 1;
    for (var i = 0; i < loop; i++) {
      await matmulTextureOp.execute(firstMatrix, secondMatrix, shape);
      console.log("MatmulTextureOp: "+await matmulTextureOp.data());
     }
   }
  /*
  {
    // First Matrix
    // Works: [256, 128];
    // Not work: [259, 127]; [7, 3];
    const firstMatrixSize = [15, 8];
    const firstMatrix =
      createUint32Array(firstMatrixSize[0], firstMatrixSize[1]);
    const shape = new Uint32Array([firstMatrixSize[0], firstMatrixSize[1]]);
    const copyTextureOp = new compute.CopyTextureOp(device, glslang,
    'rgba8uint', 4); const loop = 1; for (var i = 0; i < loop; i++) { await
    copyTextureOp.execute(firstMatrix, shape); console.log("Texture rgba8 not
    staging: "+await copyTextureOp.data());
    }
  }
  */
  /*
  {
    // First Matrix
    // Works: [16, 8]; [32, 16];
    // Not work: [17, 9];[15, 7]; [15, 8];
    const firstMatrixSize = [15, 8];
    const firstMatrix =
      createFloat32Array(firstMatrixSize[0], firstMatrixSize[1]);
    const shape = new Uint32Array([firstMatrixSize[0], firstMatrixSize[1]]);

    const copyTextureOp = new compute.CopyTextureOp(device, glslang,
  'rgba32float', 16); const loop = 1; for (var i = 0; i < loop; i++) { await
  copyTextureOp.execute(firstMatrix, shape); console.log("Texture rgba32f not
  staging: "+await copyTextureOp.data());
    }
  }
  */

/*
  const arrayProduct = (arr) => {
    let product = 1;
    for (let i = 0; i < arr.length; i++) {
      product *= arr[i];
    }
    return product;
  };
  {
    // First Matrix.
    const inChannels = 1;
    const filterHeight = 1;
    const filterWidth = 1;
    const strideHeight = 2;
    const strideWidth = 2;
    const dilationHeight = 1;
    const dilationWidth = 1;
    const pad = [0, 0];
    const xShape = [1, 4, 4, inChannels];
    const wShape = [filterHeight, filterWidth, inChannels, 3];
    const outputShape = [1, 2, 2, 3];  // ouputShape.length must be 4

    // First Matrix
    const xSize = arrayProduct(xShape);
    const firstMatrix = new Float32Array(xSize);
    for (var i = 0; i < xSize; i++) {
      firstMatrix[i] = Math.random();
    }

    // Second Matrix
    const wSize = arrayProduct(wShape);
    const secondMatrix = new Float32Array(wSize);
    for (var i = 0; i < wSize; i++) {
      secondMatrix[i] = Math.random();
    }


    const conv2dBufferOP = new compute.Conv2dBufferOp(device, glslang);
    const loop = 1;
    for (var i = 0; i < loop; i++) {
      await conv2dBufferOP.execute(firstMatrix, secondMatrix, null);
      console.log('conv2d: ' + await conv2dBufferOP.data());
    }
  }
*/

})();

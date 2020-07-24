import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';

function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] =
        i;  // Math.random();  // tf.randomUniform(shape, 0, 2.5);//0.01*i;
  }
  return matrix;
}

function createFloat32Array1(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] =
        1;  // i;//Math.random();  // tf.randomUniform(shape, 0, 2.5);//0.01*i;
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

function compareTwoFloat32Array(a, b, w, h) {
  // let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(a[i] - b[i]) > 0.01) {
      console.log('Mismatch at ' + i);
      return i;
    }
  }
  return -1;
}

function compareThreeFloat32Array(a, b, c, w, h) {
  // let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(a[i] - b[i]) > 0.01 || Math.abs(b[i] - c[i]) > 0.01 ||
        Math.abs(a[i] - c[i]) > 0.01) {
      console.log('Mismatch at ' + i);
      return i;
    }
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
  const trials = 50;
  const reps = 50;
  const resultCheck = false;
  const size_x = 256;
  const size_y = 256;
  // Result check.
  {

    const firstMatrixSize = [size_x, size_y];
    const firstMatrix = createFloat32Array(size_x, size_y);
    // Second Matrix.
    const secondMatrixSize = [size_x, size_y];
    const secondMatrix = createFloat32Array(size_x, size_y);
    const shape = new Uint32Array([
      firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
      secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
    ]);
    //
    // TFJS code:
    /*
    await tf.ready();
    var a = tf.tensor2d(firstMatrix, firstMatrixSize);
    var b = tf.tensor2d(secondMatrix, secondMatrixSize);

    var result = tf.matMul(a, b);
    */
    // console.log(await result.data());
    //

    const matmulBufferOp = new compute.MatmulBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    matmulBufferOp.executeSync();
    const matmulBufferOpData = await matmulBufferOp.data();

    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    matmulPackedBufferOp.executeSync();
    const matmulPackedBufferOpData = await matmulPackedBufferOp.data();

    const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float', 4);
    matmulTextureR32FOp.executeSync();
    const matmulTextureR32FOpData = await matmulPackedBufferOp.data();

    const compareResult = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpData, matmulTextureR32FOpData,
        size_x, size_y);
    const matmulPackedBufferOpWPT2 = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 2);
    matmulPackedBufferOpWPT2.executeSync();
    const matmulPackedBufferOpWPT2Data = await matmulPackedBufferOp.data();

    const compareResult2 = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpWPT2Data,
        matmulTextureR32FOpData, size_x, size_y);


    if (compareResult == -1 && compareResult2 == -1) {
      console.log('All results match!');
    }
    else 
    {
      console.log('Results mismatch!!!');
    }
  }

  {
    // const size_x = 32;
    // const size_y = 32;

    const firstMatrixSize = [size_x, size_y];
    const firstMatrix = createFloat32Array(size_x, size_y);
    // Second Matrix.
    const secondMatrixSize = [size_x, size_y];
    const secondMatrix = createFloat32Array(size_x, size_y);
    const shape = new Uint32Array([
      firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
      secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
    ]);
    const matmulBufferOp = new compute.MatmulBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);

    // const reps=100;
    const times = [];
    const trial = async () => {
      // let result;
      for (let r = 0; r < reps; ++r) {
        matmulBufferOp.executeSync();
      }
      await matmulBufferOp.data();
    };

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }

    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    const times2 = times.map(function(time){
      return Number(time.toFixed(2));
    });
    console.log('Sync buffer ' + times2);
    console.log(`Sync buffer Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
    console.log(
        `Sync buffer Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  }

  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const size_x = 256;
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

    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);

    const times = [];
    const trial = async () => {
      // let result;
      for (let r = 0; r < reps; ++r) {
        matmulPackedBufferOp.executeSync();
      }
      await matmulPackedBufferOp.data();
    };

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }

    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    const times2 = times.map(function(time){
      return Number(time.toFixed(2));
    });
    console.log('Sync packed buffer ' + times2);
    console.log(`Sync packed buffer Mean time: ${fmt(mean)} ms -> ${
        fmt(mean / reps)} / rep`);
    console.log(`Sync packed buffer  Min time: ${fmt(min)} ms -> ${
        fmt(min / reps)} / rep`);
  }

  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const size_x = 256;
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

    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape,2);

    const times = [];
    const trial = async () => {
      // let result;
      for (let r = 0; r < reps; ++r) {
        matmulPackedBufferOp.executeSync();
      }
      await matmulPackedBufferOp.data();
    };

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }

    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    const times2 = times.map(function(time){
      return Number(time.toFixed(2));
    });
    console.log('Sync packed buffer WPT2x2 ' + times2);
    console.log(`Sync packed buffer WPT2x2 Mean time: ${fmt(mean)} ms -> ${
        fmt(mean / reps)} / rep`);
    console.log(`Sync packed buffer WPT2x2 Min time: ${fmt(min)} ms -> ${
        fmt(min / reps)} / rep`);
  }

  {
    const size_x = 256;
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
    const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float', 4);

    const times = [];
    const trial = async () => {
      // let result;
      for (let r = 0; r < reps; ++r) {
        matmulTextureR32FOp.executeSync();
      }
      await matmulTextureR32FOp.data();
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    // await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    const times2 = times.map(function(time){
      return Number(time.toFixed(2));
    });
    console.log('Sync r32float  ' + times2);
    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    console.log(`Sync r32float texture Mean time: ${fmt(mean)} ms -> ${
        fmt(mean / reps)} / rep`);
    console.log(`Sync r32float texture Min time: ${fmt(min)} ms -> ${
        fmt(min / reps)} / rep`);
  }
  // TODO: RGBA32F not work!
  {
    /*
    const oldLog = console.log;
    let times = new Array();
    compute.startLog(times, oldLog);
    */
    const firstMatrixSize = [size_x, size_y];
    const firstMatrix = createFloat32Array(size_x, size_y);
    // Second Matrix.
    const secondMatrixSize = [size_x, size_y];
    const secondMatrix = createFloat32Array(size_x, size_y);
    const shape = new Uint32Array([
      firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
      secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
    ]);

    const matmulTextureOp2 = new compute.MatmulTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float', 16);
    for (var i = 0; i < trials; i++) {
      // First Matrix.
      const start = performance.now();
      matmulTextureOp2.executeSync();
      await matmulTextureOp2.data();
    }
    /*
    compute.endLog(times, oldLog);
    console.log(times);
    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    console.log(
        `Async texture mean time: ${fmt(mean)} ms -> ${fmt(mean / 1)} / rep`);
    console.log(`Async texture mime: ${fmt(min)} ms -> ${fmt(min / 1)} / rep`);
    */
  }
})();

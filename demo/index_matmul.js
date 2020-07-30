import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';

function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = i; //Math.random();
  }
  return matrix;
}

function compareThreeFloat32Array(a, b, c, w, h) {
  for (let i = 0; i < w * h; i++) {
    if (i == 0) {
      console.log('item 0=' + a[i] + ', ' + b[i] + ',' + c[i]);
    }
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
const trials = 1;
const reps = 1;
function logTimes(name,times) {
  const times2 = times.map(function(time) {
    return Number(time.toFixed(2));
  });
  console.log(name + times2);
  const mean = times.reduce((a, b) => a + b, 0) / trials;
  const min = Math.min(...times);
  const fmt = (n) => n.toFixed(2);
  console.log(name + ` Mean time: ${fmt(mean)} ms -> ${
      fmt(mean / reps)} / rep`);
  console.log(name + `Min time: ${fmt(min)} ms -> ${
      fmt(min / reps)} / rep`);
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

  const resultCheck = false;
  const size_x = 256;
  const size_y = size_x;
  console.log('Input size: ' + size_x + ',' + size_y);

  const firstMatrixSize = [size_x, size_y];
  const firstMatrix = createFloat32Array(size_x, size_y);
  // Second Matrix.
  const secondMatrixSize = [size_x, size_y];
  const secondMatrix = createFloat32Array(size_x, size_y);
  const shape = new Uint32Array([
    firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
    secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
  ]);
  // Result check
  {
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
    console.log("matmulPackedBufferOpData = "+ matmulPackedBufferOpData);

    const matmulBufferVec4Op = new compute.MatmulBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape, 8);
    matmulBufferVec4Op.executeSync();
    const matmulBufferVec4OpData = await matmulBufferVec4Op.data();

    var compareResult = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpData, matmulBufferVec4OpData,
        size_x, size_y);

    if (compareResult == -1) {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulBufferVec4Op results match!');
    } else {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulBufferVec4Op results mismatch!!!');
    }

    const matmulTextureRGBA32FV2Op = new compute.MatmulTextureRGBA32FV2Op(
      device, glslang, firstMatrix, secondMatrix, shape, 8, 'rgba32float',
      16);
    matmulTextureRGBA32FV2Op.executeSync();
    const matmulTextureRGBA32FV2OpData = await matmulTextureRGBA32FV2Op.data();
    console.log("matmulTextureRGBA32FV2OpData = "+ matmulTextureRGBA32FV2OpData);

    var compareResult = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpData,
        matmulTextureRGBA32FV2OpData, size_x, size_y);

    if (compareResult == -1) {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulTextureRGBA32FV2OpData results match!');
    } else {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulTextureRGBA32FV2OpData results mismatch!!!');
    }

    const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4, 'r32float', 4);
    matmulTextureR32FOp.executeSync();
    const matmulTextureR32FOpData = await matmulPackedBufferOp.data();

    compareResult = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpData, matmulTextureR32FOpData,
        size_x, size_y);

    if (compareResult == -1) {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulTextureR32FOp results match!');
    } else {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulTextureR32FOp results mismatch!!!');
    }

    const matmulTextureRGBA32FOp = new compute.MatmulTextureRGBA32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4, 'rgba32float',
        16);
    matmulTextureRGBA32FOp.executeSync();
    const matmulTextureRGBA32FOpData = await matmulTextureRGBA32FOp.data();

    compareResult = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpData,
        matmulTextureRGBA32FOpData, size_x, size_y);

    if (compareResult == -1) {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulTextureRGBA32FOp results match!');
    } else {
      console.log(
          'matmulBufferOp, matmulPackedBufferOp, matmulTextureRGBA32FOp results mismatch!!!');
    }

    const matmulPackedBufferOpWPT4 = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4);
    matmulPackedBufferOpWPT4.executeSync();
    const matmulPackedBufferOpWPT4Data = await matmulPackedBufferOp.data();

    const compareResult2 = compareThreeFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpWPT4Data,
        matmulTextureR32FOpData, size_x, size_y);

    if (compareResult2 == -1) {
      console.log(
          'matmulBufferOp, matmulPackedBufferOpWPT4, matmulTextureR32FOp results match!');
    } else {
      console.log(
          'matmulBufferOp, matmulPackedBufferOpWPT4, matmulTextureR32FOp results mismatch!!!');
    }

  }
  if (trials == 0) {
    return;
  }
  // Performance test
  {
    const matmulBufferOp = new compute.MatmulBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);

    const times = [];
    const trial = async () => {
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

    await trial();

    logTimes(" buffer  ", times);
  }

  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        matmulPackedBufferOp.executeSync();
      }
      await matmulPackedBufferOp.data();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(" packed buffer  ", times);
  }

  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 2);

    const times = [];
    const trial = async () => {
      // let result;
      for (let r = 0; r < reps; ++r) {
        matmulPackedBufferOp.executeSync();
      }
      await matmulPackedBufferOp.data();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(" packed buffer WPT2x2  ", times);
  }

  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const matmulBufferVec4Op = new compute.MatmulBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        matmulBufferVec4Op.executeSync();
      }
      await matmulBufferVec4Op.data();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(" buffer vec4  ", times);
  }

  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        matmulPackedBufferOp.executeSync();
      }
      await matmulPackedBufferOp.data();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(" packed buffer WPT4x4  ", times);
  }

  {
    const WPT = 4;
    const format = 'r32float';
    const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, WPT, format, 4);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        matmulTextureR32FOp.executeSync();
      }
      await matmulTextureR32FOp.data();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(" r32float texture WPT4x4 ", times);
  }
})();

import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';

function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = Math.random();
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

function compareFloat32Array(a, b, w, h, name) {
  for (let i = 0; i < w * h; i++) {
    if (i == 0) {
      console.log('item 0=' + a[i] + ', ' + b[i]);
    }
    if (Math.abs(a[i] - b[i]) > 0.01) {
      console.error(name + ' mismatch at ' + i);
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

const trials = 50;
const reps = 50;

const resultCheck = false;
const size_x = 2048;
const size_y = size_x;

function logTimes(name, times) {
  const times2 = times.map(function(time) {
    return Number(time.toFixed(2));
  });
  console.log(name + times2);
  const mean = times.reduce((a, b) => a + b, 0) / trials;
  const min = Math.min(...times);
  const fmt = (n) => n.toFixed(2);
  console.log(
      name + ` Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
  console.log(name + `Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
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
  if (resultCheck) {
    //
    // TFJS code:
    /*
    await tf.ready();
    var a = tf.tensor2d(firstMatrix, firstMatrixSize);
    var b = tf.tensor2d(secondMatrix, secondMatrixSize);

    var result = tf.matMul(a, b);
    console.log(await result.data());
    */

    const matmulBufferOp = new compute.MatmulBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    matmulBufferOp.executeSync();
    const matmulBufferOpData = await matmulBufferOp.data();

    const matmulPackedBufferOp = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    matmulPackedBufferOp.executeSync();
    const matmulPackedBufferOpData = await matmulPackedBufferOp.data();
    compareFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpData, size_x, size_y,
        ' matmulPackedBuffer ');

    const matmulBufferVec4Op = new compute.MatmulBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape, 8);
    matmulBufferVec4Op.executeSync();
    const matmulBufferVec4OpData = await matmulBufferVec4Op.data();

    compareFloat32Array(
        matmulBufferOpData, matmulBufferVec4OpData, size_x, size_y,
        ' matmulBufferVec4 ');

    const matmulPackedBufferOpWPT4 = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4);
    matmulPackedBufferOpWPT4.executeSync();
    const matmulPackedBufferOpWPT4Data = await matmulPackedBufferOp.data();

    compareFloat32Array(
        matmulBufferOpData, matmulPackedBufferOpWPT4Data, size_x, size_y,
        ' matmulPackedBufferOpWPT4Data ');

    const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4, 'r32float');
    matmulTextureR32FOp.executeSync();
    const matmulTextureR32FOpData = await matmulPackedBufferOp.data();

    compareFloat32Array(
        matmulBufferOpData, matmulTextureR32FOpData, size_x, size_y,
        ' matmulTextureR32FOp ');

    const matmulTextureRGBA32FV2Op = new compute.MatmulTextureRGBA32FV2Op(
        device, glslang, firstMatrix, secondMatrix, shape, 8, 'rgba32float');
    matmulTextureRGBA32FV2Op.executeSync();
    const matmulTextureRGBA32FV2OpData = await matmulTextureRGBA32FV2Op.data();
    // TODO: at first, I am trying to use matmulBufferOpData as reference.
    // Howerver, the first item of matmulBufferOpData turns into 8 after several
    // test. Possible reason is due to memory pressure so this data is gced.
    compareFloat32Array(
        matmulTextureR32FOpData, matmulTextureRGBA32FV2OpData, size_x, size_y,
        ' matmulTextureRGBA32FV2 ');
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
      matmulBufferOp.dispose();
    };

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }

    await trial();

    logTimes(' buffer  ', times);
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
      matmulPackedBufferOp.dispose();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' packed buffer  ', times);
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
      matmulPackedBufferOp.dispose();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' packed buffer WPT2x2  ', times);
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
      matmulPackedBufferOp.dispose();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' packed buffer WPT4x4  ', times);
  }

  {
    const WPT = 4;
    const format = 'r32float';
    const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, WPT, format);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        matmulTextureR32FOp.executeSync();
      }
      await matmulTextureR32FOp.data();
      matmulTextureR32FOp.dispose();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' r32float texture WPT4x4 ', times);
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
      matmulBufferVec4Op.dispose();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' buffer vec4 WPT8x8 ', times);
  }

  {
    const WPT = 8;
    const format = 'rgba32float';
    const matmulTextureR32FOp = new compute.MatmulTextureRGBA32FV2Op(
        device, glslang, firstMatrix, secondMatrix, shape, WPT, format);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        matmulTextureR32FOp.executeSync();
      }
      await matmulTextureR32FOp.data();
      matmulTextureR32FOp.dispose();
    };

    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' rgba32float texture WPT8x8 ', times);
  }


})();

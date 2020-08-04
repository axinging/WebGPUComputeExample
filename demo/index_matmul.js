import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';
import * as utils from './utils.js';

var errorStatus = false;
function compareFloat32Array(a, b, w, h, name) {
  for (let i = 0; i < w * h; i++) {
    if (i == 0) {
      console.log('item 0=' + a[i] + ', ' + b[i]);
    }
    if (Math.abs(a[i] - b[i]) > 0.01) {
      errorStatus = true;
      console.error(name + ' mismatch at ' + i + ', ' + a[i] + ',' + b[i]);
      return i;
    }
  }
  return -1;
}

const trials = 50;
const reps = 50;

const resultCheck = true;
const size_x = 256;
const size_y = size_x;

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
  const firstMatrix = utils.createFloat32Array(size_x, size_y);
  // Second Matrix.
  const secondMatrixSize = [size_x, size_y];
  const secondMatrix = utils.createFloat32Array(size_x, size_y);
  const shape = new Uint32Array([
    firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
    secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
  ]);
  // Result check
  if (resultCheck) {
    await checkCorrectness(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape);
  }

  if (trials == 0) {
    return;
  }
  // Performance test
  {
    // const oldLog = console.log;
    // let times = new Array();
    // compute.startLog(times, oldLog);
    const op = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4);
    await utils.time(
        op, utils.executeOp, ' packed buffer WPT4x4 ', trials, reps);
  }

  {
    const WPT = 4;
    const format = 'r32float';
    const op = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, WPT, format);
    await utils.time(
        op, utils.executeOp, ' r32float texture WPT4x4 ', trials, reps);
  }

  {
    const op = new compute.MatmulBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(op, utils.executeOp, ' buffer vec4 WPT8x8 ', trials, reps);
  }

  {
    const WPT = 8;
    const format = 'rgba32float';
    const op = new compute.MatmulTextureRGBA32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, WPT, format);
    await utils.time(
        op, utils.executeOp, ' rgba32float texture WPT8x8 ', trials, reps);
  }

  const testAll = false;
  if (testAll) {
    const op = new compute.MatmulBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(op, utils.executeOp, ' buffer ', trials, reps);
  }

  if (testAll) {
    const op = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(op, utils.executeOp, ' packed buffer ', trials, reps);
  }

  if (testAll) {
    const op = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 2);
    await utils.time(
        op, utils.executeOp, ' packed buffer WPT2x2 ', trials, reps);
  }
})();


async function checkCorrectness(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape) {
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
  matmulPackedBufferOp.dispose();

  const matmulBufferVec4Op = new compute.MatmulBufferVec4Op(
      device, glslang, firstMatrix, secondMatrix, shape, 8);
  matmulBufferVec4Op.executeSync();
  const matmulBufferVec4OpData = await matmulBufferVec4Op.data();

  compareFloat32Array(
      matmulBufferOpData, matmulBufferVec4OpData, size_x, size_y,
      ' matmulBufferVec4 ');
  matmulBufferVec4Op.dispose();

  const matmulPackedBufferOpWPT4 = new compute.MatmulPackedBufferOp(
      device, glslang, firstMatrix, secondMatrix, shape, 4);
  matmulPackedBufferOpWPT4.executeSync();
  const matmulPackedBufferOpWPT4Data = await matmulPackedBufferOp.data();

  compareFloat32Array(
      matmulBufferOpData, matmulPackedBufferOpWPT4Data, size_x, size_y,
      ' matmulPackedBufferOpWPT4Data ');
  matmulPackedBufferOpWPT4.dispose();

  const matmulTextureR32FOp = new compute.MatmulTextureR32FOp(
      device, glslang, firstMatrix, secondMatrix, shape, 4, 'r32float');
  matmulTextureR32FOp.executeSync();
  const matmulTextureR32FOpData = await matmulPackedBufferOp.data();

  compareFloat32Array(
      matmulBufferOpData, matmulTextureR32FOpData, size_x, size_y,
      ' matmulTextureR32FOp ');
  matmulTextureR32FOp.dispose();

  const matmulTextureRGBA32FOp = new compute.MatmulTextureRGBA32FOp(
      device, glslang, firstMatrix, secondMatrix, shape, 8, 'rgba32float');
  matmulTextureRGBA32FOp.executeSync();
  const matmulTextureRGBA32FOpData = await matmulTextureRGBA32FOp.data();
  // TODO: at first, I am trying to use matmulBufferOpData as reference.
  // Howerver, the first item of matmulBufferOpData turns into 8 after several
  // test. Possible reason is due to memory pressure so this data is gced.
  compareFloat32Array(
      matmulBufferOpData, matmulTextureRGBA32FOpData, size_x, size_y,
      ' matmulTextureRGBA32F ');
  matmulTextureRGBA32FOp.dispose();

  matmulBufferOp.dispose();

  if (errorStatus) {
    console.error('Error and exit!!!');
    return;
  } else {
    console.log('All test pass!!!');
  }
}
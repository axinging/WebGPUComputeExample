import * as compute from '@webgpu/compute';
import * as utils from './utils.js';

const resultCheck = true;

export async function runTestMatmul(device, glslang, size_x = 256, size_y = 256, trials = 50, reps = 50) {
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
    const error = await checkCorrectnessMatmul(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape);
    if (error > 0) return;
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
}

export async function checkCorrectnessMatmul(
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

  /*
  const matmulCPUOp = new compute.MatmulCPUOp(firstMatrix, secondMatrix, shape);
  matmulCPUOp.executeSync();
  const matmulReferenceData = matmulCPUOp.data();
  */
  let errorSummary = {error: 0};
  const matmulGPUOp = new compute.MatmulBufferOp(
      device, glslang, firstMatrix, secondMatrix, shape);

  matmulGPUOp.executeSync();
  const matmulReferenceData = await matmulGPUOp.data();

  {
    const op = new compute.MatmulBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.MatmulBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape, 8);
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.MatmulPackedBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4);
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.MatmulTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 4, 'r32float');
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }
  {
    const op = new compute.MatmulTextureRGBA32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 8, 'rgba32float');
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }
  return errorSummary.error;
}


export async function runTestAdd(device, glslang, size_x = 4096, size_y = 256, trials = 50, reps = 50) {
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

  if (resultCheck) {
    const error = await checkCorrectnessAdd(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape);
    if (error) return;
  }

  {
    const addOp = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(addOp, utils.executeOp, ' Add buffer ', trials, reps);
  }

  {
    const addOp = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float');
    await utils.time(
        addOp, utils.executeOp, ' Add texture r32float ', trials, reps);
  }

  {
    const addOp = new compute.AddBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(addOp, utils.executeOp, ' Add bufferVec4 ', trials, reps);
  }

  {
    const addOp = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');
    await utils.time(
        addOp, utils.executeOp, ' Add texture rgba32float ', trials, reps);
  }
}

export async function checkCorrectnessAdd(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape) {
  let errorSummary = {error: 0};
  {
    const op = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float');
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.AddBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }

  {
    const op = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }

  return errorSummary.error;
}

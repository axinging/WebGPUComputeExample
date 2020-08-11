import * as compute from '@webgpu/compute';
import * as utils from './utils.js';

const trials = 50;
const reps = 50;
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
    const error = await checkCorrectnessMatmulSameOp(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape);
    if (error > 0) return;
  }
}

export async function checkCorrectnessMatmulSameOp(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape) {
  let errorSummary = {error: 0};
  const matmulGPUOp = new compute.MatmulBufferVec4Op(
    device, glslang, firstMatrix, secondMatrix, shape, 8);

  matmulGPUOp.executeSync();
  const matmulReferenceData = await matmulGPUOp.data();
  {
    const op = new compute.MatmulBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape, 8);
    await utils.executeCompareAndDispose(
        op, matmulReferenceData, size_x, size_y, errorSummary);
  }
  return errorSummary.error;
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
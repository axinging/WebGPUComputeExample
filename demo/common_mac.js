import * as compute from '@webgpu/compute';
import * as utils from './utils.js';

const trials = 50;
const reps = 50;
const resultCheck = true;

export async function runTestMatmul(device, glslang) {
  const size_x = 512;
  const size_y = size_x;
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
}

export async function checkCorrectnessMatmul(
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

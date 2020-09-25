import * as compute from '@webgpu/compute';
import * as utils from './utils.js';

const resultCheck = true;
// Below case are for work group size test.
export async function checkCorrectnessAddBufferWGS(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
    workgroupsize) {
  let errorSummary = {error: 0};
  {
    const op = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, workgroupsize);
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }
  return errorSummary.error;
}

export async function runTestAddBufferWGS(
    device, glslang, size_x = 4096, size_y = 256, trials = 50, reps = 50,
    warmupTrials = 50, workgroupsize) {
  console.log(
      'Work group size: ' + workgroupsize +
      '---------------------------------------------------------');
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
    const error = await checkCorrectnessAddBufferWGS(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
        workgroupsize);
    if (error) return;
  }

  {
    const addOp = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape, workgroupsize);
    await utils.time(
        addOp, utils.executeOp, ' Add buffer(float) ', trials, reps, warmupTrials);
  }
}

export async function checkCorrectnessAddBufferVec4WGS(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
    workgroupsize) {
  let errorSummary = {error: 0};
  {
    const op = new compute.AddBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape, workgroupsize);
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }
  return errorSummary.error;
}

export async function runTestAddBufferVec4WGS(
    device, glslang, size_x = 4096, size_y = 256, trials = 50, reps = 50,
    warmupTrials = 50, workgroupsize) {
  console.log(
      'Work group size: ' + workgroupsize +
      '---------------------------------------------------------');
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
    const error = await checkCorrectnessAddBufferVec4WGS(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
        workgroupsize);
    if (error) return;
  }

  {
    const addOp = new compute.AddBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape, workgroupsize);
    await utils.time(
        addOp, utils.executeOp, ' Add buffer(vec4)', trials, reps,
        warmupTrials);
  }
}

export async function checkCorrectnessAddTextureR32FWGS(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
    workgroupsize) {
  let errorSummary = {error: 0};
  {
    const op = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float');
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }
  return errorSummary.error;
}

export async function runTestAddTextureR32FWGS(
    device, glslang, size_x = 4096, size_y = 256, trials = 50, reps = 50,
    warmupTrials = 50, workgroupsize) {
  console.log(
      'Work group size: ' + workgroupsize +
      '---------------------------------------------------------');
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
    const error = await checkCorrectnessAddTextureR32FWGS(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
        workgroupsize);
    if (error) return;
  }

  {
    const addOp = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float',
        workgroupsize);
    await utils.time(
        addOp, utils.executeOp, ' Add texture(r32float) ', trials, reps,
        warmupTrials);
  }
}

export async function checkCorrectnessAddTextureRGBA32FWGS(
    device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
    workgroupsize) {
  let errorSummary = {error: 0};

  {
    const op = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float',
        workgroupsize);
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }

  return errorSummary.error;
}

export async function runTestAddTextureRGBA32FWGS(
    device, glslang, size_x = 4096, size_y = 256, trials = 50, reps = 50,
    warmupTrials = 50, workgroupsize) {
  console.log(
      'Work group size: ' + workgroupsize +
      '---------------------------------------------------------');
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
    const error = await checkCorrectnessAddTextureRGBA32FWGS(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape,
        workgroupsize);
    if (error) return;
  }

  {
    const addOp = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float',
        workgroupsize);
    await utils.time(
        addOp, utils.executeOp, ' Add texture(rgba32float) ', trials, reps,
        warmupTrials);
  }
}
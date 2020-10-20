// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import * as compute from '@webgpu/compute';
import * as utils from './utils.js';

const resultCheck = true;
export async function runTestAdd(
    device, glslang, size_x = 4096, size_y = 256, trials = 50, reps = 50,
    warmupTrails = 50) {
  console.log(
      'Input size: ' + size_x + ',' + size_y +
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
    const error = await checkCorrectnessAdd(
        device, glslang, firstMatrix, secondMatrix, size_x, size_y, shape);
    if (error) return;
  }
  if (trials == 0) {
    return;
  }

  {
    const addOp = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(
        addOp, utils.executeOp, ' Add buffer(float) ', trials, reps, warmupTrails);
  }

  {
    const addOp = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float');
    await utils.time(
        addOp, utils.executeOp, ' Add texture(r32float) ', trials, reps,
        warmupTrails);
  }

  {
    /* Newly added, not work under rgba32f at any size
    const addOp = new compute.AddBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(
        addOp, utils.executeOp, ' Add buffer(vec4) ', trials, reps, warmupTrails);
    */
  }

  {
    const addOp = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');
    await utils.time(
        addOp, utils.executeOp, ' Add texture(rgba32float) ', trials, reps,
        warmupTrails);
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
    /* Newly added, not work under rgba32f at any size
    const op = new compute.AddBufferVec4Op(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
    */
  }

  {
    const op = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');
    await utils.executeCompareAndDisposeAdd(
        op, firstMatrix, secondMatrix, size_x, size_y, errorSummary);
  }

  return errorSummary.error;
}


(async () => {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  // GPURequestAdapterOptions can be: 'low-power' or 'high-performance'
  // Below code is not tested. And will fail on Windows.
  // const gpuOptions = {'high-performance'};
  // const adapter = await navigator.gpu.requestAdapter(gpuOptions);
  const enableTimeStamp = false;
  const device = await adapter.requestDevice();
  const glslang = await glslangInit();
  const trials = 1;
  const repeat = 1;
  const warmupTrials = 0;

  console.error(
      '**********************************************************Below is for input size test**********************************************************');
  await runTestAdd(
      device, glslang, 257, 31, trials, repeat, warmupTrials);
})();

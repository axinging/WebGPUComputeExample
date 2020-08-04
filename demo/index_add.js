import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

import * as utils from './utils.js';

var errorStatus = false;

function compareAddFloat32Array(result, firstMatrix, secondMatrix, w, h) {
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(result[i] - (firstMatrix[i] + secondMatrix[i])) > 0.01) {
      errorStatus = true;
      console.error(name + ' mismatch at ' + i);
      return i;
    }
  }
  return -1;
}

const trials = 50;
const reps = 50;
const resultCheck = true;
const size_x = 4096;
const size_y = 256;

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

  if (resultCheck) {
    {
      const addBufferOp = new compute.AddBufferOp(
          device, glslang, firstMatrix, secondMatrix, shape);
      addBufferOp.executeSync();
      compareAddFloat32Array(
          await addBufferOp.data(), firstMatrix, secondMatrix, size_x, size_y);
    }
    {
      const addTextureOp = new compute.AddTextureOp(
          device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');
      addTextureOp.executeSync();

      compareAddFloat32Array(
          await addTextureOp.data(), firstMatrix, secondMatrix, size_x, size_y);
    }
    {
      const addTextureOp = new compute.AddTextureR32FOp(
          device, glslang, firstMatrix, secondMatrix, shape, 'r32float');
      addTextureOp.executeSync();
      compareAddFloat32Array(
          await addTextureOp.data(), firstMatrix, secondMatrix, size_x, size_y);
    }
  }

  if (errorStatus) {
    console.error('Error and exit!!!');
    return;
  } else {
    console.log('All test pass!!!');
  }

  {
    const addOp = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    await utils.time(addOp, utils.executeOp, ' buffer ', trials, reps);
  }

  {
    const addOp = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float');
    await utils.time(
        addOp, utils.executeOp, ' texture r32float ', trials, reps);
  }

  {
    const addOp = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');
    await utils.time(
        addOp, utils.executeOp, ' texture rgba32float ', trials, reps);
  }
  /*
  {
    const oldLog = console.log;
    let times = new Array();
    compute.startLog(times, oldLog);
    const addTextureOp = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float', 16);
    for (var i = 0; i < trials; i++) {
      // First Matrix.
      await addTextureOp.execute();
      if (resultCheck) {
        const failItem = compareAddFloat32Array(
            await addTextureOp.data(), firstMatrix, secondMatrix, size_x,
            size_y);
        if (failItem != -1) {
          console.log('Test fail at item ' + failItem);
          return;
        }
      }
    }
    compute.endLog(times, oldLog);
    const times2 = times.map(function(time) {
      return Number(time.toFixed(2));
    });
    console.log(times2);
    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    console.log(`Async texture mean time: ${fmt(mean)} ms -> ${
        fmt(mean / reps)} / rep`);
    console.log(
        `Async texture mime: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  }
  */
  /*
  {
    const oldLog = console.log;
    let times = new Array();
    compute.startLog(times, oldLog);

    const addBufferOp = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);
    for (var i = 0; i < trials; i++) {
      // First Matrix.
      await addBufferOp.execute();
      // console.log(await addBufferOP.data());
      if (resultCheck) {
        const failItem = compareAddFloat32Array(
            await addBufferOp.data(), firstMatrix, secondMatrix, size_x,
            size_y);
        if (failItem != -1) {
          console.log('Test fail at item ' + failItem);
          return;
        }
      }
    }

    compute.endLog(times, oldLog);
    const times2 = times.map(function(time) {
      return Number(time.toFixed(2));
    });
    console.log(times2);
    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n) => n.toFixed(2);
    console.log(
        `Async buffer Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
    console.log(
        `Async buffer  Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  }
  */
})();

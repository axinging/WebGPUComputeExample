import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = Math.random();  // tf.randomUniform(shape, 0, 2.5);//0.01*i;
  }
  return matrix;
}

function compareAddFloat32Array(result, firstMatrix, secondMatrix, w, h) {
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(result[i] - (firstMatrix[i] + secondMatrix[i])) > 0.01) {
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
  {
    const addBufferOp = new compute.AddBufferOp(
        device, glslang, firstMatrix, secondMatrix, shape);

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        addBufferOp.executeSync();
      }
      await addBufferOp.data();
      addBufferOp.dispose();
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }

    logTimes(' buffer  ', times);
  }

  {
    const addTextureOp = new compute.AddTextureOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'rgba32float');

    const times = [];
    const trial = async () => {
      for (let r = 0; r < reps; ++r) {
        addTextureOp.executeSync();
      }
      await addTextureOp.data();
      addTextureOp.dispose();
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }

    logTimes(' texture rgba32float  ', times);
  }

  {
    const addTextureOp = new compute.AddTextureR32FOp(
        device, glslang, firstMatrix, secondMatrix, shape, 'r32float');

    const times = [];
    const trial = async () => {
      // let result;
      for (let r = 0; r < reps; ++r) {
        addTextureOp.executeSync();
      }
      await addTextureOp.data();
      if (resultCheck) {
        const failItem = compareAddFloat32Array(
            await addTextureOp.data(), firstMatrix, secondMatrix, size_x,
            size_y);
        if (failItem != -1) {
          console.log('AddTextureR32FOp Test fail at item ' + failItem);
          return;
        }
      }
      addTextureOp.dispose();
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    await trial();

    for (let t = 0; t < trials; ++t) {
      const start = performance.now();
      await trial();
      times.push(performance.now() - start);
    }
    logTimes(' texture r32float  ', times);
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

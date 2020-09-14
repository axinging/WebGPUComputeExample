import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';
import * as utils from './utils.js';
import * as common from './common.js';

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

  const trials = 50, reps = 50, warmupTrails = 50;
  var size = 256;
  await common.runTestMatmul(device, glslang, size, size, trials, reps, warmupTrails);
  size = 512;
  await common.runTestMatmul(device, glslang, size, size, trials, reps, warmupTrails);
  size = 1024;
  await common.runTestMatmul(device, glslang, size, size, trials, reps, warmupTrails);
  size = 2048;
  await common.runTestMatmul(device, glslang, size, size, trials, reps, warmupTrails);
  await common.runTestAdd(device, glslang, 4096, 256, trials, reps, warmupTrails);
  await common.runTestAdd(device, glslang, 4096, 1024, trials, reps, warmupTrails);

})();
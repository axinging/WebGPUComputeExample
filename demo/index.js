import * as compute from '@webgpu/compute';
// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';
import * as utils from './utils.js';
import * as common from './common.js';
import * as commonmac from './common_mac.js';

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
  await common.runTestMatmul(device, glslang);
  await common.runTestAdd(device, glslang);
})();
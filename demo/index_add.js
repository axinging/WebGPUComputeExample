// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

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
  await common.runTestAdd(device, glslang, 10, 3, 0,0);
  // await common.runTestAdd(device, glslang, 10, 3, 0,0);
  await common.runTestAdd(device, glslang, 35, 15, 0,0);
  // await common.runTestAdd(device, glslang, 10, 20, 0,0);
  //await common.runTestAdd(device, glslang, 200, 100, 0,0);
  await common.runTestAdd(device, glslang, 4096, 128, 0,0);
  // await common.runTestAdd(device, glslang, 4096, 256, 0,0);
  // await common.runTestAdd(device, glslang, 4096, 512, 0,0);
})();
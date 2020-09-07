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
  const trials = 50;
  const repeat = 50;
  const warmupTrials = 50;

  for (var y = 1; y <= 8; y = y*2) 
  {
    for (var x = 1; x <= 8; x = x*2) 
    {
      await common.runTestAdd(device, glslang, 256*y,256*x, trials,repeat, warmupTrials);
    }
  }

})();
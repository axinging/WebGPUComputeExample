// import glslangModule from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';

import * as common from './common.js';
import * as commonwgs from './common_wgs.js';
import * as utils from './utils.js';

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

  console.error(
      '**********************************************************Below is for input size test**********************************************************');
  for (var y = 1; y <= 8; y = y * 2) {
    for (var x = 1; x <= 8; x = x * 2) {
      await common.runTestAdd(
          device, glslang, 256 * y, 256 * x, trials, repeat, warmupTrials);
    }
  }

  console.error(
      '**********************************************************Below is for work group size test**********************************************************');
  const inputSizeWGS = 1024;
  await commonwgs.runTestAddBufferWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [64, 1, 1]);
  await commonwgs.runTestAddBufferWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [128, 1, 1]);
  await commonwgs.runTestAddBufferWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [256, 1, 1]);
  await commonwgs.runTestAddBufferVec4WGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [64, 1, 1]);
  await commonwgs.runTestAddBufferVec4WGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [128, 1, 1]);
  await commonwgs.runTestAddBufferVec4WGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [256, 1, 1]);
  await commonwgs.runTestAddTextureR32FWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [8, 8, 1]);
  await commonwgs.runTestAddTextureR32FWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [16, 16, 1]);
  await commonwgs.runTestAddTextureR32FWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [32, 32, 1]);
  await commonwgs.runTestAddTextureRGBA32FWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [8, 8, 1]);
  await commonwgs.runTestAddTextureRGBA32FWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [16, 16, 1]);
  await commonwgs.runTestAddTextureRGBA32FWGS(
      device, glslang, inputSizeWGS, inputSizeWGS, trials, repeat, warmupTrials,
      [32, 32, 1]);
})();
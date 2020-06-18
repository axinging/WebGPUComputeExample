// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {CopyTextureOp} from './copy_texture';

export class CopyTextureRGBA32FOp extends CopyTextureOp {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  times: [];
  gpuTextureFirstMatrix: GPUTexture;
  shape: Uint32Array;
  format: GPUTextureFormat;
  kBytesPerTexel: number;
  constructor(device: GPUDevice, glslang: Glslang) {
    super(device, glslang, 'rgba32float', 16);
  }

  createArray(w: number, h: number) {
    let matrix = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      matrix[i] = i;
    }
    return matrix;
  }

  async execute(mode = 0) {
    // First Matrix
    // Works: [15, 7]; [16, 8]; [32, 16];
    // Not work: [17, 9];
    const firstMatrixSize = [16, 8];
    const firstMatrix =
        this.createArray(firstMatrixSize[0], firstMatrixSize[1]);
    const shape = new Uint32Array([firstMatrixSize[0], firstMatrixSize[1]]);
    const result = await this.compileAndRun(firstMatrix, null, shape, '', mode);
    return result;
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    return new Float32Array(arrayBuffer);
  }
}

import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class AddBufferOp extends BufferOp {
  constructor(device: GPUDevice, glslang: Glslang) {
    super(device, glslang);
  }
}
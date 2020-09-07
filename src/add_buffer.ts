import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class AddBufferOp extends BufferOp {
  workGroupSize: [number, number, number];
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      workGroupSize: [number, number, number] = [128, 1, 1]) {
    // Compute shader code (GLSL)
    super(device, glslang);
    // const TS = 32;
    this.workGroupSize = workGroupSize;
    this.compile(firstMatrix, secondMatrix, shape, this.getShader());
  }

  async execute() {
    const result = await this.compileAndRun(this.workGroupSize);
    return result;
  }

  executeSync() {
    const result = this.compileAndRunSync(this.workGroupSize);
    return result;
  }

  getShader() {
    // Compute shader code (GLSL)
    const computeShaderCode = `#version 450
        layout(set = 0, binding = 0) uniform Uniforms {
          int inputWidth;
          int inputHeight;
          int filterWidth;
          int filterHeight;
          int outputWidth;
          int outputHeight;
        } uniforms;

        layout(set = 0, binding = 1) readonly buffer FirstMatrix {
            float firstMatrix[];
        } ;

        layout(set = 0, binding = 2) readonly buffer SecondMatrix {
            float secondMatrix[];
        } ;

        layout(set = 0, binding = 3) buffer ResultMatrix {
            float resultMatrix[];
        } ;

        layout(local_size_x = ${this.workGroupSize[0]}, local_size_y = ${
        this.workGroupSize[1]}, local_size_z = 1) in;

        void main() {
          uint index = gl_GlobalInvocationID.x;
          resultMatrix[index] = firstMatrix[index]+secondMatrix[index];
        }
          `;
    return computeShaderCode;
  }
}
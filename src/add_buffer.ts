import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class AddBufferOp extends BufferOp {
  constructor(device: GPUDevice, glslang: Glslang) {
    super(device, glslang);
  }
  async execute() {
    // First Matrix
    const firstMatrix = new Float32Array(
        [4 /* rows */, 2 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8]);

    // Second Matrix
    const secondMatrix = new Float32Array(
        [4 /* rows */, 2 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8]);
    const result =
        await this.compileAndRun(firstMatrix, secondMatrix, this.getShader());
    return result;
  }
  async executeStaging() {
    // First Matrix
    const firstMatrix = new Float32Array(
        [4 /* rows */, 2 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8]);

    // Second Matrix
    const secondMatrix = new Float32Array(
        [4 /* rows */, 2 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8]);
    const result = await this.compileAndRunStaging(
        firstMatrix, secondMatrix, this.getShader());
    return result;
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    return new Float32Array(arrayBuffer);
  }
  getShader() {
    // Compute shader code (GLSL)
    const computeShaderCode = `#version 450
        layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
            vec2 size;
            float numbers[];
        } firstMatrix;
      
        layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
            vec2 size;
            float numbers[];
        } secondMatrix;
      
        layout(std430, set = 0, binding = 2) buffer ResultMatrix {
            vec2 size;
            float numbers[];
        } resultMatrix;
      
        void main() {
          resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);
          ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
      
          int index = resultCell.y + resultCell.x * int(secondMatrix.size.y);
          resultMatrix.numbers[index] = firstMatrix.numbers[index]+secondMatrix.numbers[index];
        }
        `;
    return computeShaderCode;
  }
}
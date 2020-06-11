import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class AddBufferOp extends BufferOp {
  constructor(device: GPUDevice, glslang: Glslang) {
    super(device, glslang);
  }

  createArray(w: number, h: number) {
    let matrix = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      matrix[i] = i;
    }
    return matrix;
  }

  async execute(mode = 0) {
    // First Matrix.
    const firstMatrixSize = [4, 8];
    const firstMatrix = this.createArray(4, 8);

    // Second Matrix.
    const secondMatrixSize = [4, 8];
    const secondMatrix = this.createArray(4, 8);
    const shape = new Int32Array([
      firstMatrixSize[0], firstMatrixSize[1], secondMatrixSize[0],
      secondMatrixSize[1], firstMatrixSize[0], firstMatrixSize[1]
    ]);
    let result;
    result = await this.compileAndRun(
        firstMatrix, secondMatrix, shape, this.getShader(), mode);
    return result;
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    return new Float32Array(arrayBuffer);
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
            //vec2 size;
            float numbers[];
        } firstMatrix;
      
        layout(set = 0, binding = 2) readonly buffer SecondMatrix {
            //vec2 size;
            float numbers[];
        } secondMatrix;
      
        layout(set = 0, binding = 3) buffer ResultMatrix {
            //vec2 size;
            float numbers[];
        } resultMatrix;
      
        void main() {
          //resultMatrix.size = vec2(inputWidth, inputHeight);
          ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
      
          int index = resultCell.y + resultCell.x * int(uniforms.inputHeight);
          resultMatrix.numbers[index] = firstMatrix.numbers[index]+secondMatrix.numbers[index];
        }
        `;
    return computeShaderCode;
  }
}
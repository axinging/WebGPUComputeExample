import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class AddBufferOp extends BufferOp {
  workGroupSize: [number, number, number];
  constructor(device: GPUDevice, glslang: Glslang, firstMatrix: Float32Array|Uint32Array,
    secondMatrix: Float32Array|Uint32Array, shape: Uint32Array) {
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

       layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
        layout(std430,set = 1, binding = 0) readonly buffer FirstMatrix {
            float firstMatrix[];
        } ;
      
        layout(std430,set = 2, binding = 1) readonly buffer SecondMatrix {
            float secondMatrix[];
        } ;
      
        layout(std430,set = 3, binding = 2) buffer ResultMatrix {
            float resultMatrix[];
        } ;



        void main() {
          // resultMatrix.size = vec2(inputWidth, inputHeight);
          // ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
      
          // int index = resultCell.y + resultCell.x * int(uniforms.inputHeight);
          uint index = gl_GlobalInvocationID.x;////resultCell.x;// + resultCell.y * int(uniforms.inputWidth);
          resultMatrix[index] = firstMatrix[index]+secondMatrix[index];
        }
        `;
    super(device, glslang,firstMatrix, secondMatrix,  shape, computeShaderCode);
    // const TS = 32;
    this.workGroupSize = [128, 1, 1];
  }

  async execute() {
    const result = await this.compileAndRun(this.workGroupSize);
    return result;
  }
  executeSync() {
    const result = this.compileAndRunSync(this.workGroupSize);
    return result;
  }
/*
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

        layout(local_size_x = ${this.workGroupSize[0]}, local_size_y = ${
        this.workGroupSize[1]}, local_size_z = 1) in;

        void main() {
          //resultMatrix.size = vec2(inputWidth, inputHeight);
          ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
      
          int index = resultCell.y + resultCell.x * int(uniforms.inputHeight);
          resultMatrix.numbers[index] = firstMatrix.numbers[index]+secondMatrix.numbers[index];
        }
        `;
    return computeShaderCode;
  }
  */
}
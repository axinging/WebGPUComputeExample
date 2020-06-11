import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class AddTextureOp extends TextureOp {
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
    // First Matrix
    const firstMatrixSize = [2, 4];
    const firstMatrix = this.createArray(2, 4);

    // Second Matrix
    const secondMatrixSize = [2, 4];
    const secondMatrix = this.createArray(2, 4);
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

        layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D outputValues;

        layout(set = 0, binding = 2, rgba32f) uniform readonly image2D values;
        // readonly
        layout(set = 0, binding = 3, rgba32f) uniform readonly image2D filterValues;
        
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        

        void main() {
          uint row = (gl_GlobalInvocationID.x);
          uint col = (gl_GlobalInvocationID.y);
          vec4 x = imageLoad(values, ivec2(row, col));
          vec4 w = imageLoad(filterValues, ivec2(row, col));
          vec4 res = x + w;
          imageStore(outputValues, ivec2(gl_GlobalInvocationID.xy), res);
        }
        
        `;
    return computeShaderCode;
  }
}
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class AddTextureOp extends TextureOp {
  constructor(device: GPUDevice, glslang: Glslang) {
    super(device, glslang);
  }
  async execute(mode = 0) {
    // First Matrix
    const firstMatrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

    // Second Matrix
    const secondMatrix = new Float32Array([1, 9, 3, 4, 5, 6, 7, 8]);
    const shape = new Int32Array([2, 4, 2, 4, 2, 4]);
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
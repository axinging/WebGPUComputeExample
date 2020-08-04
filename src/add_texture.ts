import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class AddTextureOp extends TextureOp {
  workGroupSize: [number, number, number];
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      format: GPUTextureFormat) {
    super(device, glslang, format);
    const TS = 16;
    this.workGroupSize = [TS, TS, 1];
    this.compile(firstMatrix, secondMatrix, shape, this.getShader());
  }

  async execute() {
    const result = await this.compileAndRun(this.workGroupSize);
    return result;
  }

  executeSync() {
    this.compileAndRunSync(this.workGroupSize);
    return;
  }

  private getShader() {
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

          layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D
    outputValues;

          layout(set = 0, binding = 2, rgba32f) uniform readonly image2D values;
          // readonly
          layout(set = 0, binding = 3, rgba32f) uniform readonly image2D
    filterValues;

          layout(local_size_x = ${this.workGroupSize[0] / 4}, local_size_y = ${
        this.workGroupSize[1]}, local_size_z = 1) in;

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
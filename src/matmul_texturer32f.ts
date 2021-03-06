import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';

import * as tex_util from './tex_util';
import {TextureOp} from './texture';

export class MatmulTextureR32FOp extends TextureOp {
  workGroupSize: [number, number, number];
  workPerThread: [number, number, number];
  scalarFormt: string;
  vectorFormat: string;
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      workPerThread: [number, number, number], format: GPUTextureFormat) {
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    /// super(device, glslang, firstMatrix, secondMatrix,
    /// shape,computeShaderCode, format, kBytesPerTexel);
    super(device, glslang, format);
    const TS = 16;
    this.workGroupSize = [TS, TS, 1];
    this.workPerThread = workPerThread;
    this.scalarFormt = tex_util.getVarType(format);
    this.vectorFormat = tex_util.getVectorType(format);
    this.compile(firstMatrix, secondMatrix, shape, this.getShader());
  }

  executeSync() {
    const result =
        this.compileAndRunSync(this.workGroupSize, this.workPerThread);
    return result;
  }

  // Experimental. DO not USE!
  async execute(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array, mode = 0) {
    const result = await this.compileAndRun(this.workGroupSize);
    return result;
  }

  private getShader() {
    // Compute shader code (GLSL)
    // https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgpu/src/kernels/matmul_packed_webgpu.ts
    const computeShaderCode = `#version 450

    layout(local_size_x = ${this.workGroupSize[0]}, local_size_y = ${
        this.workGroupSize[1]}, local_size_z = 1) in;
    
    /* TODO.
    layout(std140, set = 0, binding = 0) uniform Uniforms {
        ivec3 aShape; ivec3 bShape; ivec3 outShape; 
    };
    */
    layout(set = 0, binding = 0) uniform Uniforms {
      int inputWidth;
      int inputHeight;
      int filterWidth;
      int filterHeight;
      int outputWidth;
      int outputHeight;
    };     
  
    layout(set = 0, binding = 1, ${
        tex_util.getShaderFormat(this.format)}) uniform writeonly ${
        tex_util.getShaderImageType(this.format)} result;

    layout(set = 0, binding = 2, ${
        tex_util.getShaderFormat(this.format)}) uniform readonly ${
        tex_util.getShaderImageType(this.format)} A;
    // readonly
    layout(set = 0, binding = 3, ${
        tex_util.getShaderFormat(this.format)}) uniform readonly ${
        tex_util.getShaderImageType(this.format)} B;

    // TODO. Make this works with rectangle.
    int dimAOuter = inputWidth;   // aShape[1];
    int dimInner = filterWidth;   // aShape[2];
    int dimBOuter = outputWidth;  // bShape[2];

    ${this.scalarFormt} mm_readA(int row, int col);
    ${this.scalarFormt}  mm_readB(int row, int col);
    void mm_write(int row, int col, ${this.scalarFormt}  value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int RowPerThread = ${this.workPerThread[0]};
    const int ColPerThread = ${this.workPerThread[1]};
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

    shared ${this.scalarFormt}  mm_Asub[TileAOuter][TileInner];
    shared ${this.scalarFormt}  mm_Bsub[TileInner][TileBOuter];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

      int numTiles = (dimInner - 1) / TileInner + 1;

      ${this.scalarFormt}  acc[RowPerThread][ColPerThread];
      ${this.scalarFormt}  ACached;
      ${this.scalarFormt}  BCached[ColPerThread];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = ${this.scalarFormt}(0.0);
        }
      }

      const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        //
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] =
                mm_readA(globalRow + innerRow, t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] =
                mm_readB(t * TileInner + inputRow, globalCol + innerCol);
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner; k++) {
          for (int inner = 0; inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          if ((globalCol + innerCol) < dimBOuter &&
              (globalRow + innerRow) < dimAOuter) {
            mm_write(
                globalRow + innerRow, globalCol + innerCol,
                acc[innerRow][innerCol]);
          }
        }
      }
    }

    ${this.scalarFormt} mm_readA(int row, int col) {
      return  ${this.scalarFormt}(imageLoad(A, ivec2(col, row)).r);
    }

    ${this.scalarFormt} mm_readB(int row, int col) {
      return ${this.scalarFormt} (imageLoad(B, ivec2(col, row)).r);
    }

    void mm_write(int row, int col, ${this.scalarFormt} value) {
      // TODO: Figure out why need vec4 here.
      imageStore(result, ivec2(col, row), ${
        this.vectorFormat}(value, 0.0, 0.0, 0.0));
    }

    void main() {
      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
    `;
    return computeShaderCode;
  }
}

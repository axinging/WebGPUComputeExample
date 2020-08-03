import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class MatmulTextureRGBA32FV2Op extends TextureOp {
  workGroupSize: [number, number, number];
  workPerThread: number;
  outputShape: number[];
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      workPerThread: number, format: GPUTextureFormat) {
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    /// super(device, glslang, firstMatrix, secondMatrix,
    /// shape,computeShaderCode, format, kBytesPerTexel);
    super(device, glslang, format);
    const TS = 16;
    const TS_Y = 16;
    this.workGroupSize = [TS, TS_Y / 4, 1];
    this.outputShape = [shape[0], shape[1], shape[1]];
    this.workPerThread = workPerThread;
    this.compile(firstMatrix, secondMatrix, shape, this.getShader());
  }

  async execute(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array, mode = 0) {
    const result = await this.compileAndRun(this.workGroupSize);
    return result;
  }

  executeSync() {
    const result =
        this.compileAndRunSync(this.workGroupSize, this.workPerThread);
    return result;
  }

  private getShader() {
    // Compute shader code (GLSL)
    // https://github.com/tensorflow/tfjs/pull/2048
    // https://github.com/qjia7/tfjs/commit/10ab492faabb031dc56ef2e6d8f789e719edc94e
    // https://www.ibiblio.org/e-notes/webgl/gpu/mul/slm.htm
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
  
    layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D result;

    layout(set = 0, binding = 2, rgba32f) uniform readonly image2D A;
    // readonly
    layout(set = 0, binding = 3, rgba32f) uniform readonly image2D B;


    const int TILE_M = 32;
    const int TILE_N = 128;
    const int TILE_K = 64;
    const int VEC_SIZE = 4;
    const int ROWS_PER_WI = 8;
    shared vec4 atile[TILE_M * TILE_K / VEC_SIZE];
    int dimAOuter = inputWidth;
    int dimInner = inputHeight / VEC_SIZE;
    int dimBOuter = filterHeight / VEC_SIZE;
    // TODO: Fix shape.
    ivec3 aShape = ivec3(inputWidth, inputHeight, inputHeight);
    ivec3 bShape = ivec3(inputWidth, inputHeight, inputHeight);
    // Consider compiling a different version of the shader that doesn't care
    // about boundary conditions. May slightly improve performance.

    vec4 mm_readA(int row, int col) {
        return imageLoad(A, ivec2(col, row));
    }
    vec4 mm_readB(int row, int col) {
        return imageLoad(B, ivec2(col, row));
    }
    void mm_write(int row, int col, vec4 value) {
        imageStore(result, ivec2(col, row), value);
    }
    
    void main() {
    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);
    int local_y = int(gl_LocalInvocationID.y);
    // Result ctile is M rows x N columns
    // M = 32, we have 4 rows of work-items, so we need 32/4 8 results down
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8
    // results across = 2 float4s across
    vec4 dot00,dot01,dot02,dot03,dot04,dot05,dot06,dot07;
    vec4 dot10,dot11,dot12,dot13,dot14,dot15,dot16,dot17;
    // Src0 is used to load atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int globalRow = ( group_y * TILE_M ) + ROWS_PER_WI * local_y;
    int globalColA = local_x;
    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    // K = 64, we'll process four rows at a time
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 floats
    // across = 2 float4s across
    int globalCol0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );
    int globalCol1 = globalCol0 + ( TILE_N / 2 / VEC_SIZE );
    int rowB0 = 0, rowB1 = 0;
    int slm = local_y * ( ROWS_PER_WI * TILE_K / VEC_SIZE );
    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do{
      // We want to load atile, which is M rows x K columns
      // M = 32, and we have 4 rows of work-items, so each work-item must load
      // 32/4 = 8 rows.
      // K = 64, and we have 16 columns of work-items, so each work-item must
      // load 64/16 = 4 columns = 1 float4.
      atile[slm + local_x + 0 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow, globalColA);
      atile[slm + local_x + 1 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 1, globalColA);
      atile[slm + local_x + 2 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 2, globalColA);
      atile[slm + local_x + 3 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 3, globalColA);
      atile[slm + local_x + 4 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 4, globalColA);
      atile[slm + local_x + 5 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 5, globalColA);
      atile[slm + local_x + 6 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 6, globalColA);
      atile[slm + local_x + 7 * TILE_K / VEC_SIZE] =
          mm_readA(globalRow + 7, globalColA);
      globalColA += TILE_K / VEC_SIZE;
      barrier();
      int i = 0;
      do{
          // We get better performance by loading btile first.
          vec4 brow00 = mm_readB(rowB0, globalCol0); rowB0++;
          vec4 brow01 = mm_readB(rowB0, globalCol0); rowB0++;
          vec4 brow02 = mm_readB(rowB0, globalCol0); rowB0++;
          vec4 brow03 = mm_readB(rowB0, globalCol0); rowB0++;
          vec4 brow10 = mm_readB(rowB1, globalCol1); rowB1++;
          vec4 brow11 = mm_readB(rowB1, globalCol1); rowB1++;
          vec4 brow12 = mm_readB(rowB1, globalCol1); rowB1++;
          vec4 brow13 = mm_readB(rowB1, globalCol1); rowB1++;
          vec4 a0 = atile[slm + i + 0 * TILE_K / VEC_SIZE ];
          dot00 = brow00*a0.x + dot00;
          dot00 = brow01*a0.y + dot00;
          dot00 = brow02*a0.z + dot00;
          dot00 = brow03*a0.w + dot00;
          dot10 = brow10*a0.x + dot10;
          dot10 = brow11*a0.y + dot10;
          dot10 = brow12*a0.z + dot10;
          dot10 = brow13*a0.w + dot10;
          vec4 a1 = atile[slm + i + 1 * TILE_K / VEC_SIZE ];
          dot01 = brow00*a1.x + dot01;
          dot01 = brow01*a1.y + dot01;
          dot01 = brow02*a1.z + dot01;
          dot01 = brow03*a1.w + dot01;
          dot11 = brow10*a1.x + dot11;
          dot11 = brow11*a1.y + dot11;
          dot11 = brow12*a1.z + dot11;
          dot11 = brow13*a1.w + dot11;
          vec4 a2 = atile[slm + i + 2 * TILE_K / VEC_SIZE ];
          dot02 = brow00*a2.x + dot02;
          dot02 = brow01*a2.y + dot02;
          dot02 = brow02*a2.z + dot02;
          dot02 = brow03*a2.w + dot02;
          dot12 = brow10*a2.x + dot12;
          dot12 = brow11*a2.y + dot12;
          dot12 = brow12*a2.z + dot12;
          dot12 = brow13*a2.w + dot12;
          vec4 a3 = atile[slm + i + 3 * TILE_K / VEC_SIZE ];
          dot03 = brow00*a3.x + dot03;
          dot03 = brow01*a3.y + dot03;
          dot03 = brow02*a3.z + dot03;
          dot03 = brow03*a3.w + dot03;
          dot13 = brow10*a3.x + dot13;
          dot13 = brow11*a3.y + dot13;
          dot13 = brow12*a3.z + dot13;
          dot13 = brow13*a3.w + dot13;
          vec4 a4 = atile[slm + i + 4 * TILE_K / VEC_SIZE ];
          dot04 = brow00*a4.x + dot04;
          dot04 = brow01*a4.y + dot04;
          dot04 = brow02*a4.z + dot04;
          dot04 = brow03*a4.w + dot04;
          dot14 = brow10*a4.x + dot14;
          dot14 = brow11*a4.y + dot14;
          dot14 = brow12*a4.z + dot14;
          dot14 = brow13*a4.w + dot14;
          vec4 a5 = atile[slm + i + 5 * TILE_K / VEC_SIZE ];
          dot05 = brow00*a5.x + dot05;
          dot05 = brow01*a5.y + dot05;
          dot05 = brow02*a5.z + dot05;
          dot05 = brow03*a5.w + dot05;
          dot15 = brow10*a5.x + dot15;
          dot15 = brow11*a5.y + dot15;
          dot15 = brow12*a5.z + dot15;
          dot15 = brow13*a5.w + dot15;
          vec4 a6 = atile[slm + i + 6 * TILE_K / VEC_SIZE ];
          dot06 = brow00*a6.x + dot06;
          dot06 = brow01*a6.y + dot06;
          dot06 = brow02*a6.z + dot06;
          dot06 = brow03*a6.w + dot06;
          dot16 = brow10*a6.x + dot16;
          dot16 = brow11*a6.y + dot16;
          dot16 = brow12*a6.z + dot16;
          dot16 = brow13*a6.w + dot16;
          vec4 a7 = atile[slm + i + 7 * TILE_K / VEC_SIZE ];
          dot07 = brow00*a7.x + dot07;
          dot07 = brow01*a7.y + dot07;
          dot07 = brow02*a7.z + dot07;
          dot07 = brow03*a7.w + dot07;
          dot17 = brow10*a7.x + dot17;
          dot17 = brow11*a7.y + dot17;
          dot17 = brow12*a7.z + dot17;
          dot17 = brow13*a7.w + dot17;
          i++;
      }
      while( i < TILE_K / VEC_SIZE );
      barrier();
      w += TILE_K / VEC_SIZE;
    }
    while( w < dimInner );
    mm_write(globalRow, globalCol0, dot00);
    mm_write(globalRow + 1, globalCol0, dot01);
    mm_write(globalRow + 2, globalCol0, dot02);
    mm_write(globalRow + 3, globalCol0, dot03);
    mm_write(globalRow + 4, globalCol0, dot04);
    mm_write(globalRow + 5, globalCol0, dot05);
    mm_write(globalRow + 6, globalCol0, dot06);
    mm_write(globalRow + 7, globalCol0, dot07);
    mm_write(globalRow, globalCol1, dot10);
    mm_write(globalRow + 1, globalCol1, dot11);
    mm_write(globalRow + 2, globalCol1, dot12);
    mm_write(globalRow + 3, globalCol1, dot13);
    mm_write(globalRow + 4, globalCol1, dot14);
    mm_write(globalRow + 5, globalCol1, dot15);
    mm_write(globalRow + 6, globalCol1, dot16);
    mm_write(globalRow + 7, globalCol1, dot17);
    }
        `;
    return computeShaderCode;
  }
}

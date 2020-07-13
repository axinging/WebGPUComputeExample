import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class MatmulTextureOp extends TextureOp {
  workGroupSize: [number, number, number];
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      format: GPUTextureFormat, kBytesPerTexel: number) {
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    /// super(device, glslang, firstMatrix, secondMatrix,
    /// shape,computeShaderCode, format, kBytesPerTexel);
    super(device, glslang, format, kBytesPerTexel);
    const TS = 32;
    this.workGroupSize = [TS, TS, 1];
    this.compile(firstMatrix, secondMatrix, shape, this.getShader());
  }

  async execute(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array, mode = 0) {
    const result = await this.compileAndRun(this.workGroupSize);
    return result;
  }

  executeSync() {
    this.compileAndRunSync(this.workGroupSize);
    return;
  }

  private getShader() {
    // Compute shader code (GLSL)
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    const computeShaderCode = `#version 450
        layout(set = 0, binding = 0) uniform Uniforms {
          int inputWidth;
          int inputHeight;
          int filterWidth;
          int filterHeight;
          int outputWidth;
          int outputHeight;
        } uniforms;

        layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D C;

        layout(set = 0, binding = 2, rgba32f) uniform readonly image2D A;
        // readonly
        layout(set = 0, binding = 3, rgba32f) uniform readonly image2D B;
        //#define TS 32u
        //layout (local_size_x = TS/4, local_size_y = TS, local_size_z = 1) in;
        layout(local_size_x = ${this.workGroupSize[0]}, local_size_y = ${
        this.workGroupSize[1]}, local_size_z = 1) in;
        const uint TS =  ${this.workGroupSize[0]};

        // uniform uvec3 MNK;
        shared vec4 Asub[TS/4][TS];  // Local memory to fit a tile of
        shared vec4 Bsub[TS/4][TS];  // TS*TS elements of A and B
      void main() {
          //uint M = MNK.x, N = MNK.y, K = MNK.z;
          // TODO: change this to INPUT SIZE.
          // uint M = 32, N = 32, K = 32;
          uint M = uniforms.inputWidth;
          uint N = uniforms.inputWidth;
          uint K = uniforms.inputWidth;

          // Thread identifiers
          uint row = gl_LocalInvocationID.x; // Local row ID (max: TS)
          uint col = gl_LocalInvocationID.y; // Local col ID (max: TS)
          uint globalRow = TS/4*gl_WorkGroupID.x + row; // Row ID of C (0..M)
          uint globalCol = TS*gl_WorkGroupID.y + col; // Col ID of C (0..N)

          // Initialise the accumulation register
          vec4 acc = vec4(0.0);
          // Loop over all tiles
          uint numTiles = K/TS; //4
          for (uint t=0u; t < numTiles; t++) {
              // Load one tile of A and B into local memory
              uint tiledRow = TS/4*t + row;
              uint tiledCol = TS*t + col;
              Asub[col][row] = imageLoad(A, ivec2(tiledCol*M + globalRow));
              // A[tiledCol*M + globalRow];
              Bsub[col][row] = imageLoad(B,  ivec2(globalCol*K + tiledRow));
              // B[globalCol*K + tiledRow];

              // Synchronise to make sure the tile is loaded
              memoryBarrierShared();
              barrier();

              // Perform the computation for a single tile
              for (uint k=0u; k < TS/4; k++) {
                  acc += Asub[k][row] * Bsub[col][k];
              }
              // Synchronise before loading the next tile
              barrier();
          }
          // Store the final result in C
          // C[globalCol*M + globalRow] = acc;
          imageStore(C, ivec2(globalRow,globalCol), acc);
          // imageStore(C, ivec2(globalRow,globalCol), vec4(3,80,90,100));
      }
        `;
    return computeShaderCode;
  }
}
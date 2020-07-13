import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class MatmulBufferOp extends BufferOp {
  workGroupSize: [number, number, number];
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array) {
    super(device, glslang);
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
    const result = this.compileAndRunSync(this.workGroupSize);
    return result;
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

        layout (set = 0, binding = 1) readonly buffer ssbA {
            float A[];
          };
          layout (set = 0, binding = 2) readonly buffer ssbB {
            float B[];
          };
          layout (set = 0, binding = 3) writeonly buffer ssbC {
            float C[];
          };

        //#define TS 32u
        //layout (local_size_x = TS, local_size_y = TS, local_size_z = 1) in;
        layout(local_size_x = ${this.workGroupSize[0]}, local_size_y = ${
        this.workGroupSize[1]}, local_size_z = 1) in;

        // uniform uvec3 MNK;
        const uint TS =  ${this.workGroupSize[0]};
        // const uint TS_Y =  ${this.workGroupSize[1]};
        shared float Asub[TS][TS];  // Local memory to fit a tile of
        shared float Bsub[TS][TS];  // TS*TS elements of A and B
      void main() {
          //uint M = MNK.x, N = MNK.y, K = MNK.z;
          // TODO: change this to INPUT SIZE.
          uint M = uniforms.inputWidth;
          uint N = uniforms.inputWidth;
          uint K = uniforms.inputWidth;

          // Thread identifiers
          uint row = gl_LocalInvocationID.x; // Local row ID (max: TS)
          uint col = gl_LocalInvocationID.y; // Local col ID (max: TS)
          uint globalRow = TS*gl_WorkGroupID.x + row; // Row ID of C (0..M)
          uint globalCol = TS*gl_WorkGroupID.y + col; // Col ID of C (0..N)

          // Initialise the accumulation register
          float acc = 0.0;

          // Loop over all tiles
          uint numTiles = K/TS;
          for (uint t=0u; t < numTiles; t++) {

              // Load one tile of A and B into local memory
              uint tiledRow = TS*t + row;
              uint tiledCol = TS*t + col;
              Asub[col][row] = A[tiledCol*M + globalRow];
              Bsub[col][row] = B[globalCol*K + tiledRow];

              // Synchronise to make sure the tile is loaded
              memoryBarrierShared();
              barrier();

              // Perform the computation for a single tile
              for (uint k=0u; k < TS; k++) {
                  acc += Asub[k][row] * Bsub[col][k];
              }

              // Synchronise before loading the next tile
              barrier();
          }
          // Store the final result in C
          C[globalCol*M + globalRow] = acc;
          // imageStore(C, ivec2(globalRow,globalCol), acc);
          // imageStore(C, ivec2(globalRow,globalCol), vec4(3,80,90,100));
      }

        `;
    return computeShaderCode;
  }
}
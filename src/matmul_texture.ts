import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class MatmulTextureOp extends TextureOp {
  constructor(
      device: GPUDevice, glslang: Glslang, format: GPUTextureFormat,
      kBytesPerTexel: number) {
    super(device, glslang, format, kBytesPerTexel);
  }

  async execute(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shapeMatmul: Uint32Array,
      mode = 0) {
    const result = await this.compileAndRun(
        firstMatrix, secondMatrix, this.shape, this.getShader(), mode);
    return result;
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

        layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D C;

        layout(set = 0, binding = 2, rgba32f) uniform readonly image2D A;
        // readonly
        layout(set = 0, binding = 3, rgba32f) uniform readonly image2D B;
        #define TS 32u
        layout (local_size_x = TS, local_size_y = TS, local_size_z = 1) in;
        

        // uniform uvec3 MNK;
        shared float Asub[TS][TS];  // Local memory to fit a tile of
        shared float Bsub[TS][TS];  // TS*TS elements of A and B
      void main() {
          //uint M = MNK.x, N = MNK.y, K = MNK.z;
          uint M = 64, N = 64, K = 64;
      
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
      }
        
        `;
    return computeShaderCode;
  }
}
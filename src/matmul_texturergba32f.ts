import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class MatmulTextureRGBA32FOp extends TextureOp {
  workGroupSize: [number, number, number];
  workPerThread: number;
  workPerThread2: [number, number];
  outputShape: number[];
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      workPerThread: number, format: GPUTextureFormat, kBytesPerTexel: number) {
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    /// super(device, glslang, firstMatrix, secondMatrix,
    /// shape,computeShaderCode, format, kBytesPerTexel);
    super(device, glslang, format, kBytesPerTexel);
    const TS = 16;
    // const TSK = 16;
    const WPT = 4;
    const [TSM, TSN] = [TS, TS];
    const [WPTM, WPTN] = [WPT, WPT];
    // const LPTB = TSK * WPTM * WPTN / TSM;
    const [RTSM, RTSN] = [TSM / WPTM, TSN / WPTN];
    // this.workGroupSize = [TS, TS, 1];
    this.workPerThread = WPTN;  // workPerThread;
    this.workGroupSize = [RTSN, RTSM, 1];
    this.workPerThread2 = [WPTN, WPTM];
    this.outputShape = [shape[0], shape[1], shape[1]];
    this.compile(firstMatrix, secondMatrix, shape, this.getShader(shape));
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

  private getShader(shape: Uint32Array) {
    // Compute shader code (GLSL)
    // https://github.com/qjia7/tfjs-core/blob/compileAndRunCS/src/kernels/webgl/mulmat_packed_gpu_cs_v4.ts
    const transposeA = false;
    const transposeB = false;
    const sharedDim =
        shape[0];  // const sharedDim = transposeA ? aShape[1] : aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 2);
    const TS = 16;
    const TSK = 16;
    const WPT = 4;
    const [TSM, TSN] = [TS, TS];
    const [WPTM, WPTN] = [WPT, WPT];
    const LPTA = TSK * WPTM * WPTN / TSN;
    // const LPTB = TSK * WPTM * WPTN / TSM;
    const [RTSM, RTSN] = [TSM / WPTM, TSN / WPTN];

    const aSample = transposeA ? `tiledIndex * 2, (offsetM + row) * 2` :
                                 `(offsetM + row) * 2, tiledIndex * 2`;
    const bSample = transposeB ? `(offsetN + row) * 2, tiledIndex * 2` :
                                 `tiledIndex * 2, (offsetN + row) * 2`;
    const aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
    const bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];
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

    // TODO. Make this works with rectangle.
    int dimAOuter = inputWidth; // aShape[1];
    int dimInner = filterWidth; // aShape[2];
    int dimBOuter = outputWidth; // bShape[2];

    int imod(int x, int y) {
        return x - y * (x / y);
    }
  
    vec4 getMatrixA(int row, int col, int z) {
      return imageLoad(A, ivec2(col, z));
    }
  
    vec4 getMatrixB(int row, int col, int z) {
      return imageLoad(B, ivec2(col, z));
    }
    ivec3 getOutputCoords() {
      int tileRow = int(gl_LocalInvocationID.x) * ${WPTN};
      int tileCol = int(gl_LocalInvocationID.y) * ${WPTM};
      int globalRow = int(gl_GlobalInvocationID.x) * ${WPTN};
      int globalCol = int(gl_GlobalInvocationID.y) * ${WPTM};
      vec2 resTexRC = vec2(globalRow, globalCol);
      return ivec3(0,globalRow, globalCol);
    }

    shared vec4 Asub[${TSM}][${TSK}];
    shared vec4 Bsub[${TSK}][${TSN}];
    void main() {
      ivec3 rc = getOutputCoords();
      int tidm = int(gl_LocalInvocationID.y);
      int tidn = int(gl_LocalInvocationID.x);
      int offsetM = ${TSM} * int(gl_WorkGroupID.y);
      int offsetN = ${TSN} * int(gl_WorkGroupID.x);
      vec4 Breg[${WPTN}];
      vec4 acc[${WPTM}][${WPTN}];
      for (int wm = 0; wm < ${WPTM}; wm++) {
        for (int wn = 0; wn < ${WPTN}; wn++) {
          acc[wm][wn] = vec4(0);
        }
      }
      // Loop over all tiles
      int numTiles = ${Math.ceil(sharedDimensionPacked / TSK)};
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into local memory
        for (int i = 0; i < ${LPTA}; i++) {
          int tid = tidm * ${RTSN} + tidn;
          int id = i * ${RTSN} * ${RTSM} + tid;
          int row = id / ${TSK};
          int col = imod(id, ${TSK});
          int tiledIndex = ${TSK} * t + col;
          Asub[row][col] = getMatrixA(rc.x, ${aSample});
          Bsub[col][row] = getMatrixB(rc.x, ${bSample});
        }
        //memoryBarrierShared();
        barrier();
        // Loop over the values of a single tile
        int sizeTS = (t == (numTiles - 1) &&
                      ${sharedDimensionPacked % TSK} != 0) ?
                      ${sharedDimensionPacked % TSK} : ${TSK};
        for (int k = 0; k < sizeTS; k++) {
          for (int wn = 0; wn < ${WPTN}; wn++) {
            int col = tidn + wn * ${RTSN};
            Breg[wn] = Bsub[k][col];
          }
          for (int wm = 0; wm < ${WPTM}; wm++) {
            int row = tidm + wm * ${RTSM};
            vec4 a = Asub[row][k];
            for (int wn = 0; wn < ${WPTN}; wn++) {
              vec4 b = Breg[wn];
              acc[wm][wn] += (${aSwizzle[0]} * ${bSwizzle[0]}) +
                             (${aSwizzle[1]} * ${bSwizzle[1]});
            }
          }
        }
        // Synchronize before loading the next tile.
        barrier();
      }
      // Store the final result
      for (int wm = 0; wm < ${WPTM}; wm++) {
        int globalRow = offsetM + tidm + wm * ${RTSM};
        if (globalRow >= ${Math.ceil(this.outputShape[1] / 2)}) {
          continue;
        }
        for (int wn = 0; wn < ${WPTN}; wn++) {
          int globalCol = offsetN + tidn + wn * ${RTSN};
          if (globalCol >= ${Math.ceil(this.outputShape[2] / 2)}) {
            continue;
          }
          imageStore(result, ivec2(globalCol, globalRow),
                     acc[wm][wn]);
        }
      }
    }
        `;
    return computeShaderCode;
  }
}

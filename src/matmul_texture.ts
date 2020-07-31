import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {TextureOp} from './texture';

export class MatmulTextureOp extends TextureOp {
  workGroupSize: [number, number, number];
  workPerThread: number;
  constructor(
      device: GPUDevice, glslang: Glslang,
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      format: GPUTextureFormat) {
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    /// super(device, glslang, firstMatrix, secondMatrix,
    /// shape,computeShaderCode, format, kBytesPerTexel);
    super(device, glslang, format);
    const TS = 32;
    this.workPerThread = 4;
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
      layout(
        local_size_x = ${this.workGroupSize[0]},
        local_size_y = ${
        this.workGroupSize[1] / this.workPerThread}, local_size_z = 1) in ;

    // TODO.
    // layout(std140, set = 0, binding = 0) uniform Uniforms {
    //    ivec3 aShape; ivec3 bShape; ivec3 outShape;
    //};
    //
    layout(set = 0, binding = 0) uniform Uniforms {
      int inputWidth;
      int inputHeight;
      int filterWidth;
      int filterHeight;
      int outputWidth;
      int outputHeight;
    };

    layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D result;

    layout(set = 0, binding = 2, rgba32f) uniform readonly image2D matrixA;
    // readonly
    layout(set = 0, binding = 3, rgba32f) uniform readonly image2D matrixB;

      vec4 getMatrixA(int row, int col) {
        // vec2 uv = (vec2(col, row) + halfCR) / vec2(32.0, 32.0);
        // return texture2D(matrixA, uv);
        return imageLoad(matrixA, ivec2(row, col));
      }
  
      vec4 getMatrixB(int row, int col) {
        // vec2 uv = (vec2(col, row) + halfCR) / vec2(32.0, 32.0);
        // return texture2D(matrixB, uv);
        return imageLoad(matrixB, ivec2(row, col));
      }
      void setOutput(int row, int col, vec4 values) {
        imageStore(result, ivec2(row, col), values);
        // setOutput(row * dimBOuter + col, value);
      }
      // TODO.
      int halfsize = 256 / 4;
      int K4 = 256 / 4;
      int dimAOuter = inputWidth;   // aShape[1];
      int dimInner = filterWidth;   // aShape[2];
      int dimBOuter = outputWidth;  // bShape[2];
      const int RowPerThread = 1;
      const int ColPerThread = 1;
  
      ivec2 getOutputCoords() {
        int tileRow = int(gl_LocalInvocationID.x) * RowPerThread;
        int tileCol = int(gl_LocalInvocationID.y) * ColPerThread;
        int globalRow = int(gl_GlobalInvocationID.x) * RowPerThread;
        int globalCol = int(gl_GlobalInvocationID.y) * ColPerThread;
        vec2 resTexRC = vec2(globalRow, globalCol);
        return ivec2(globalRow, globalCol);
      }

      void main(void) {
        vec4 acc = vec4(0.);
        ivec2 rc = getOutputCoords();
        // https://www.ibiblio.org/e-notes/webgl/gpu/mul/mul32_4.htm
        for (int k = 0; k < K4; k++) {

          vec4 a = getMatrixA(rc.x, k);
          int rcy = rc.y*4;
          
          acc.x += dot(a, getMatrixB(  rcy,k));
          acc.y += dot(a, getMatrixB(rcy+1,k));
          acc.z += dot(a, getMatrixB(rcy+2,k));
          acc.w += dot(a, getMatrixB(rcy+3,k));
          /*
          acc.x += dot(a, getMatrixB(k, rc.x));
          acc.y += dot(a, getMatrixB(k, rc.x+1));
          acc.z += dot(a, getMatrixB(k, rc.x+2));
          acc.w += dot(a, getMatrixB(k, rc.x+3));
          */
        }
        // setOutput(rc.x, rc.y,vec4(rc.x,rc.y,0.0,0.0));
        // rc.x 0 ... 255;
        // rc.y 0 ... 63;
        setOutput(rc.x, rc.y,acc);//vec4(rc.x,rc.y,1.0,2.0));
      }
      `;
    return computeShaderCode;
  }

  /*
  private getShader() {
    // Compute shader code (GLSL)
    // view-source:https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm2.htm
    const computeShaderCode = ` #version 450

    layout(
        local_size_x = ${this.workGroupSize[0]},
        local_size_y = ${this.workGroupSize[1] / 4}, local_size_z = 1) in ;

    // TODO.
    // layout(std140, set = 0, binding = 0) uniform Uniforms {
    //    ivec3 aShape; ivec3 bShape; ivec3 outShape;
    //};
    //
    layout(set = 0, binding = 0) uniform Uniforms {
      int inputWidth;
      int inputHeight;
      int filterWidth;
      int filterHeight;
      int outputWidth;
      int outputHeight;
    };

    layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D result;

    layout(set = 0, binding = 2, rgba32f) uniform readonly image2D matrixA;
    // readonly
    layout(set = 0, binding = 3, rgba32f) uniform readonly image2D matrixB;
    const int RowPerThread = 1;
    const int ColPerThread = 1;
    void setOutput(int flatIndex, float value) {
      // result[flatIndex] = value;
    }

    void setOutput(int row, int col, vec4 values) {
      imageStore(result, ivec2(row, col), values);
      // setOutput(row * dimBOuter + col, value);
    }
    // TODO.
    int halfsize = 256 / 4;
    int dimAOuter = inputWidth;   // aShape[1];
    int dimInner = filterWidth;   // aShape[2];
    int dimBOuter = outputWidth;  // bShape[2];

    ivec3 getOutputCoords() {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;
      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;
      vec2 resTexRC = vec2(globalRow, globalCol);
      return ivec3(0, globalRow, globalCol);
    }

    vec4 getMatrixA(int row, int col) {
      // vec2 uv = (vec2(col, row) + halfCR) / vec2(32.0, 32.0);
      // return texture2D(matrixA, uv);
      return imageLoad(matrixA, ivec2(row, col));
    }

    vec4 getMatrixB(int row, int col) {
      // vec2 uv = (vec2(col, row) + halfCR) / vec2(32.0, 32.0);

      // return texture2D(matrixB, uv);
      return imageLoad(matrixB, ivec2(row, col));
    }

    vec4 dot2x2ARowBCol(ivec3 rc) {
      vec4 result = vec4(0);
      for (int i = 0; i < halfsize; i++) {
        vec4 a = getMatrixA(rc.y, i);
        vec4 b = getMatrixB(i, rc.z);

        // These swizzled products need to be separately added.
        // See: https://github.com/tensorflow/tfjs/issues/1735
        result += (a.xxzz * b.xyxy);
        result += (a.yyww * b.zwzw);
      }
      return result;
    }

    void main() {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;
      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;
      ivec3 rc = getOutputCoords();
      vec4 result = dot2x2ARowBCol(rc);
      setOutput(globalRow, globalCol, result);
    }
    `;
    return computeShaderCode;
  }
}
  */
}

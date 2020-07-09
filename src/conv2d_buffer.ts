import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {BufferOp} from './buffer';

export class Conv2dBufferOp extends BufferOp {
  inChannels = 1;
  filterHeight = 1;
  filterWidth = 1;
  strideHeight = 2;
  strideWidth = 2;
  dilationHeight = 1;
  dilationWidth = 1;
  pad = [0, 0];
  localSizeX = 16;
  localSizeY = 16;
  workPerThread = [2, 2];

  xShape = [1, 4, 4, this.inChannels];
  wShape = [this.filterHeight, this.filterWidth, this.inChannels, 3];
  outputShape = [1, 2, 2, 3];  // ouputShape.length must be 4
  workGroupSize: [number, number, number];

  constructor(device: GPUDevice, glslang: Glslang) {
    super(device, glslang);
    const TS = 32;
    this.workGroupSize = [TS, TS, 1];
  }

  async execute(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array, mode = 0) {
    const dispatchLayout = {x: [3], y: [1, 2], z: [0]};

    const dispatch = this.computeDispatch(
        dispatchLayout, this.outputShape, [this.localSizeX, this.localSizeY, 1],
        [this.workPerThread[0], this.workPerThread[1], 1]);

    const result = await this.compileAndRun2(
        firstMatrix, secondMatrix, this.getShape(), dispatch,
        this.workGroupSize, this.getShader(), mode);
    return result;
  }

  getShape() {
    let dimUniforms: number[] = [];
    const bufferShapes = [this.xShape, this.wShape, this.outputShape];
    let currentOffset = 0;
    bufferShapes.forEach((d, i) => {
      // Uniforms.
      if (d.length === 0) {
        d = [1];
      }
      // Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      let baseAlignment;
      switch (d.length) {
        case 0:
          baseAlignment = 1;
          break;
        case 1:
          baseAlignment = 1;
          break;
        case 2:
          baseAlignment = 2;
          break;
        case 3:
          baseAlignment = 4;
          break;
        case 4:
          baseAlignment = 4;
          break;
        default:
          console.log('unsupported shape');
      }

      const padding = Math.ceil(currentOffset / baseAlignment) * baseAlignment -
          currentOffset;
      for (let p = 0; p < padding; ++p) {
        dimUniforms.push(0);
      }
      dimUniforms.push(...d);
      currentOffset += d.length + padding;
    });

    const dimensions = [
      this.filterHeight, this.filterWidth, ...this.pad, this.strideHeight,
      this.strideWidth, this.dilationHeight, this.dilationWidth
    ];

    dimUniforms = dimUniforms.concat(dimensions);

    // Uniform Buffer
    const uniformData = new Uint32Array(dimUniforms);
    return uniformData;
  }


  computeDispatch(
      layout: {x: any[]; y: any[]; z: any[];}, outputShape: {[x: string]: any;},
      workGroupSize = [1, 1, 1], elementsPerThread = [1, 1, 1]) {
    const arrayProduct = (arr: string|any[]) => {
      let product = 1;
      for (let i = 0; i < arr.length; i++) {
        product *= arr[i];
      }
      return product;
    };

    return [
      Math.ceil(
          arrayProduct(layout.x.map((d: string|number) => outputShape[d])) /
          (workGroupSize[0] * elementsPerThread[0])),
      layout.y ?
          Math.ceil(
              arrayProduct(layout.y.map((d: string|number) => outputShape[d])) /
              (workGroupSize[1] * elementsPerThread[1])) :
          1,
      layout.z ?
          Math.ceil(
              arrayProduct(layout.z.map((d: string|number) => outputShape[d])) /
              (workGroupSize[2] * elementsPerThread[2])) :
          1
    ];
  }


  getShader() {
    // Compute shader code (GLSL)
    const sampleA =
        `coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0`;
    const sampleB =
        `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ? W[row * dimBOuter + col] : 0`;
    const computeShaderCode = `#version 450
    layout (local_size_x = ${this.localSizeX},
    local_size_y = ${this.localSizeY},
    local_size_z = 1) in;
  
    // Checks whether coordinates lie within the bounds of the shape.
    bool coordsInBounds(ivec4 coord, ivec4 shape) {
      return all(greaterThanEqual(coord, ivec4(0))) &&
          all(lessThan(coord, shape));
    }

    bool coordsInBounds(ivec2 coord, ivec2 shape) {
      return all(greaterThanEqual(coord, ivec2(0))) &&
          all(lessThan(coord, shape));
    }
  
    int getFlatIndex(int coord, int shape) {
      return coord;
    }

    int getFlatIndex(ivec2 coords, ivec2 shape) {
      return int(dot(coords, ivec2(shape.y, 1.)));
    }

    int getFlatIndex(ivec3 coords, ivec3 shape) {
      return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
    }

    int getFlatIndex(ivec4 coords, ivec4 shape) {
      return int(dot(coords, ivec4(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
    }
    
    layout(std140, set = 0, binding = 0) uniform Uniforms {
      ivec4 xShape; ivec4 wShape; ivec4 outShape; ivec2 filterDims, pad, stride, dilation;
    };

    layout(std430, set = 0, binding = 1) writeonly buffer ssbOut {
      float result[];
    };
  
  
    layout(std430, set = 0, binding = 2) readonly buffer ssbx {
      float x[];
    };
  
  
    layout(std430, set = 0, binding = 3) readonly buffer ssbW {
      float W[];
    };
  
  

  
    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];
    int dimInner = filterDims[0] * filterDims[1] * xShape[3];
  
    float mm_readA(int row, int col) {
      int r = int(row), c = int(col);
      int outRow = r / outShape[2];
      int outCol = r % outShape[2];
  
      int WRow = c / (filterDims[1] * xShape[3]);
      int WCol = (c / xShape[3]) % filterDims[1];
  
      ivec4 coord = ivec4(
          batch,
          outRow * stride[0] + dilation[0] * WRow - pad[0],
          outCol * stride[1] + dilation[1] * WCol - pad[1],
          c % xShape[3]);
      return ${sampleA};
    }
  
    float mm_readB(int row, int col) {
      return ${sampleB};
    }
  
    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / outShape[2],
          row % outShape[2],
          col);
      result[getFlatIndex(outCoord, outShape)] = value;
    }
  
    const int RowPerThread = ${this.workPerThread[1]};
    const int ColPerThread = ${this.workPerThread[0]};
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;
  
    shared float mm_Asub[TileAOuter][TileInner];
    shared float mm_Bsub[TileInner][TileBOuter];
  
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;
  
      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;
  
      int numTiles = (dimInner - 1) / TileInner + 1;
  
      float acc[RowPerThread][ColPerThread];
      float ACached;
      float BCached[ColPerThread];
  
      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
  
      const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
  
      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileColA + innerCol;
  
            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol + innerCol;
  
            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol + innerCol);;
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
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol]);
          }
        }
      }
    }
  
    void main() {
      batch = int(gl_GlobalInvocationID.z);
  
      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
    `;
    return computeShaderCode;
  }
}
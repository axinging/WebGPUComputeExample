export class MatmulCPUOp {
  resultMatrixBuffer: Float32Array;
  resultMatrixBufferSize: number;
  shape: Uint32Array;
  blockSize: number;
  constructor(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array) {
    this.blockSize = 48;
    this.resultMatrixBuffer = new Float32Array(shape[0] * shape[1]);
    this.compile(firstMatrix, secondMatrix, shape);
  }
  compile(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array) {
    // This is from: backend_cpu/batchMatmul.
    const sharedDim = shape[1];
    const leftDim = shape[0];
    const rightDim = shape[3];
    const batchDim = shape[2];

    const stridesA = [shape[0] * shape[1], shape[1]];
    const [aBatch, aOuterStep, aInnerStep] = [stridesA[0], stridesA[1], 1];
    const stridesB = [shape[0] * shape[1], shape[0]];
    const [bInnerStep, bOuterStep, bBatch] = [stridesB[1], 1, stridesB[0]];

    const size = leftDim * rightDim;
    const blockSize = this.blockSize;

    for (let b = 0; b < batchDim; b++) {
      for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
        for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
          for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
            // for when blockSize doesn't evenly divide the input
            const iBlock = Math.min(i0 + blockSize, leftDim);
            const jBlock = Math.min(j0 + blockSize, rightDim);
            const kBlock = Math.min(k0 + blockSize, sharedDim);

            for (let i = i0; i < iBlock; i++) {
              for (let j = j0; j < jBlock; j++) {
                let sum = 0.0;

                for (let k = k0; k < kBlock; k++) {
                  sum +=
                      firstMatrix[b * aBatch + i * aOuterStep + k * aInnerStep] *
                      secondMatrix[k * bInnerStep + j * bOuterStep + b * bBatch];
                }
                this.resultMatrixBuffer[b * size + (i * rightDim + j)] += sum;
              }
            }
          }
        }
      }
    }
    // return result;
  }

  executeSync() {
    return;
  }

  data() {
    return this.resultMatrixBuffer;
  }
}

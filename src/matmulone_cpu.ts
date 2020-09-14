export class MatmulOneCPUOp {
  resultMatrixBuffer: Float32Array;
  resultMatrixBufferSize: number;
  shape: Uint32Array;
  blockSize: number;
  row: number;
  col: number;
  constructor(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array, row: number,
      col: number) {
    this.resultMatrixBuffer = new Float32Array(shape[0] * shape[1]);
    this.row = row;
    this.col = col;
    this.compile(firstMatrix, secondMatrix, shape);
  }

  compile(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array) {
    this.getOneFromMatrixmul(
        firstMatrix, secondMatrix, shape, this.row, this.col);
  }

  getOneFromMatrixmul(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array, row: number,
      col: number) {
    var aNumCols = shape[1], bNumRows = shape[2], bNumCols = shape[3];
    for (var i = 0; i < aNumCols; ++i) {
      this.resultMatrixBuffer[this.row * bNumCols + this.col] +=
          firstMatrix[this.row * bNumCols + i] *
          secondMatrix[this.col + i * bNumRows];
    }
  }

  executeSync() {
    return;
  }

  data() {
    return this.resultMatrixBuffer;
  }
}

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


export enum PackingScheme {
  /**
   * All values in a single texel are densely packed without any constraints.
   *
   * This is how the shader encodes a tensor with shape = [2, 3, 4]
   * (indices are [batch, row, col]).
   *
   * 000|001   010|011   020|021
   * -------   -------   -------
   * 002|003   012|013   022|023
   *
   * 100|101   110|111   120|121
   * -------   -------   -------
   * 102|103   112|113   122|123
   *
   */
  DENSE,

  /**
   * Single texels contain only values from the same batch, and from adjacent
   * rows and columns.
   *
   * This is how the shader encodes a tensor with shape = [2, 3, 5]
   * (indices are [batch, row, col]).
   *
   * 000|001   002|003   004|xxx   020|021   022|023   024|xxx
   * -------   -------   -------   -------   -------   -------
   * 010|011   012|013   014|xxx   xxx|xxx   xxx|xxx   xxx|xxx
   *
   * 100|101   102|103   104|xxx   120|121   122|123   124|xxx
   * -------   -------   -------   -------   -------   -------
   * 110|111   112|113   114|xxx   xxx|xxx   xxx|xxx   xxx|xxx
   *
   */
  SHARED_BATCH
}

export enum TextureUsage {
  RENDER,
  UPLOAD,
  PIXELS,
  DOWNLOAD
}

export enum PhysicalTextureType {
  UNPACKED_FLOAT16,
  UNPACKED_FLOAT32,
  PACKED_4X1_UNSIGNED_BYTE,
  PACKED_2X2_FLOAT32,
  PACKED_2X2_FLOAT16
}

export function getUnpackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [columns, rows];
}

export function getUnpackedArraySizeFromMatrixSize(
    matrixSize: number, channelsPerTexture: number): number {
  return matrixSize * channelsPerTexture;
}

export function getColorMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [columns * 4, rows];
}

export function getMatrixSizeFromUnpackedArraySize(
    unpackedSize: number, channelsPerTexture: number): number {
  if (unpackedSize % channelsPerTexture !== 0) {
    throw new Error(
        `unpackedSize (${unpackedSize}) must be a multiple of ` +
        `${channelsPerTexture}`);
  }
  return unpackedSize / channelsPerTexture;
}

export function decodeMatrixFromUnpackedColorRGBAArray(
    unpackedArray: Float32Array, matrix: Float32Array, channels: number) {
  const requiredSize = unpackedArray.length * channels / 4;
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }
  let dst = 0;
  for (let src = 0; src < unpackedArray.length; src += 4) {
    for (let c = 0; c < channels; c++) {
      matrix[dst++] = unpackedArray[src + c];
    }
  }
}

/*
export function getPackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [
    Math.max(1, Math.ceil(columns / 2)), Math.max(1, Math.ceil(rows / 2))
  ];
}
*/

export function getPackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number, format: GPUTextureFormat): [number, number] {
  // kBytesPerTexel = 4;
  if (format == 'rgba32float')
    return [Math.max(1, Math.ceil(rows)), Math.max(1, Math.ceil(columns / 4))];
  else if (format == 'rgba8uint')
    return [rows, columns];
  else
    return [rows, columns];
}

export function getPackedRGBAArraySizeFromMatrixShape(
    rows: number, columns: number, format: GPUTextureFormat): number {
  const [w, h] = getPackedMatrixTextureShapeWidthHeight(rows, columns, format);
  return w * h * 4;
}

// BufferSpec MinimumBufferSpec(uint32_t width, uint32_t height) {
//  uint32_t bytesPerRow = Align(width * kBytesPerTexel,
//  kTextureBytesPerRowAlignment); return {bytesPerRow * (height - 1) + width *
//  kBytesPerTexel, 0, bytesPerRow};
// }
export function getBytesPerRow(width: number, kBytesPerTexel = 16) {
  const kTextureBytesPerRowAlignment = 256;
  const alignment = kTextureBytesPerRowAlignment;
  // const kBytesPerTexel = 16;
  const value = kBytesPerTexel * width;
  // const bytesPerRow = (value + (alignment - 1)) & ~(alignment - 1);
  const bytesPerRow =
      ((value + (alignment - 1)) & ((~(alignment - 1)) >>> 0)) >>> 0;
  return bytesPerRow;
}

export function getBytesPerTexel(format: GPUTextureFormat): number {
  // kBytesPerTexel = 4;
  if (format == 'rgba32float')
    return 16;
  else if (format == 'r32float')
    return 4;
  else {
    console.error('Unsupported format ' + format);
    return 4;
  }
}

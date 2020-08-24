// import {Glslang} from '@webgpu/glslang/dist/web-devel-onefile/glslang';
import {Glslang} from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import * as utils from './fixture';

export class BufferOp {
  device: GPUDevice;
  queue: GPUQueue;
  glslang: Glslang;
  commandQueue: GPUCommandEncoder[];
  times: [];
  resultMatrixBuffer: GPUBuffer;
  resultMatrixBufferSize: number;
  // TODO; This workaround memory usage. But not work.
  // gpuReadBuffer: GPUBuffer;
  shape: Uint32Array;
  computePipeline: any;
  bindGroup: any;
  bufferID: number;
  freeBuffers: Map<number, GPUBuffer[]> = new Map();
  // enableTimeStamp: boolean;
  constructor(device: GPUDevice, glslang: Glslang) {
    this.device = device;
    this.queue = device.defaultQueue;
    this.glslang = glslang;
    this.commandQueue = [];
    this.bufferID = 0;
    // this.enableTimeStamp = false;
  }

  /*
    var x = new Int32Array(1);
    x[0] = 17;
    console.log(x[0]);
    console.log(x.length);
    this.uploadToGPU(x, 4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  */
  /*
  private uploadToGPU(values: ArrayBufferView, byteSize: number, usage: any) {
    const buffer = this.device.createBuffer({size: byteSize, usage});

    if (values) {
      buffer.setSubData(0, values as ArrayBufferView);
      values = null;
    }
    return buffer;
  }
  */
  // TIMESTAMP
  /*
  async getQueryTime(dstBuffer: GPUBuffer) {
    const dstStagingBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(dstBuffer, 0, dstStagingBuffer, 0, 16);
    this.device.defaultQueue.submit([commandEncoder.finish()]);

    // @ts-ignore
    const arrayBuf = new BigUint64Array(await dstStagingBuffer.mapReadAsync());

    // Time delta is a gpu ticks, we need convert to time using gpu frequency
    const timeDelta = arrayBuf[1] - arrayBuf[0];
    // gpu frequency can be got from console in chromium. For Winodws on
    // UHD630/RX560 = 25000000;
    const frequency = 25000000;
    // 1 ms = 1000 000 ns
    const timeInMS = Number(timeDelta) * 1000 / frequency;
    console.log('timestamp: ' + timeInMS + 'ms');
    dstStagingBuffer.destroy();
    dstBuffer.destroy();
    return timeInMS;
  }
  */

  compile(
      firstMatrix: Float32Array|Uint32Array,
      secondMatrix: Float32Array|Uint32Array, shape: Uint32Array,
      computeShaderCode: any) {
    this.shape = shape;
    /*
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] =
        this.device.createBufferMapped({
          size: (firstMatrix as Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE
        });
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
    */
    const gpuBufferFirstMatrix = this.device.createBuffer({
      mappedAtCreation: true,
      size: (firstMatrix as Float32Array).byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    // TODO: turn this into type of firstMatrix.
    new Float32Array(gpuBufferFirstMatrix.getMappedRange()).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();

    /*
    const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] =
        this.device.createBufferMapped({
          size: (secondMatrix as Float32Array).byteLength,
          usage: GPUBufferUsage.STORAGE
        });
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();
    */
    const gpuBufferSecondMatrix = this.device.createBuffer({
      mappedAtCreation: true,
      size: (secondMatrix as Float32Array).byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    // TODO: turn this into type of secondMatrix.
    new Float32Array(gpuBufferSecondMatrix.getMappedRange()).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();

    // Result Matrix
    this.resultMatrixBufferSize =
        Float32Array.BYTES_PER_ELEMENT * (shape[4] * shape[5]);
    this.resultMatrixBuffer = this.device.createBuffer({
      size: this.resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    // console.log(this.resultMatrixBufferSize);

    // This works.
    /*
    const [shapeBuffer, shapeMapping] = this.device.createBufferMapped({
      size: shape.byteLength,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Uint32Array(shapeMapping).set(shape);
    shapeBuffer.unmap();
    */
    const shapeBuffer = this.device.createBuffer({
      mappedAtCreation: true,
      size: shape.byteLength,
      usage: GPUBufferUsage.UNIFORM
    });
    // TODO: turn this into type of shape.
    new Uint32Array(shapeBuffer.getMappedRange()).set(shape);
    shapeBuffer.unmap();

    // This works too.
    /*
     const shapeBuffer = this.uploadToGPU(
         shape, shape.byteLength,
         GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC |
             GPUBufferUsage.COPY_DST);
     */

    return this.createLayout(
        gpuBufferFirstMatrix, gpuBufferSecondMatrix, shapeBuffer,
        computeShaderCode);
  }

  private createLayout(
      gpuBufferFirstMatrix: GPUBuffer, gpuBufferSecondMatrix: GPUBuffer,
      shapeBuffer: GPUBuffer, computeShaderCode: any) {
    // Use old layout:
    // Bind group layout and bind group
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          type: 'uniform-buffer'
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          type: 'readonly-storage-buffer'
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          type: 'readonly-storage-buffer'
        },
        {binding: 3, visibility: GPUShaderStage.COMPUTE, type: 'storage-buffer'}
      ]
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: shapeBuffer}},
        {binding: 1, resource: {buffer: gpuBufferFirstMatrix}},
        {binding: 2, resource: {buffer: gpuBufferSecondMatrix}},
        {binding: 3, resource: {buffer: this.resultMatrixBuffer}}
      ]
    });
    // Old layout end.

    // Pipeline setup
    const result =
        this.glslang.compileGLSLZeroCopy(computeShaderCode, 'compute', false);
    if (result.data.length === 0) {
      throw new Error('Shader compilation failed');
    }
    this.computePipeline = this.device.createComputePipeline({
      // For new layout, remove this line.
      layout: this.device.createPipelineLayout(
          {bindGroupLayouts: [bindGroupLayout]}),
      computeStage: {
        module: this.device.createShaderModule({code: result.data}),
        entryPoint: 'main'
      }
    });

    /* New layout
    const bindGroup = this.device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {binding: 0, resource: {buffer: shapeBuffer}},
        {binding: 1, resource: {buffer: gpuBufferFirstMatrix}},
        {binding: 2, resource: {buffer: gpuBufferSecondMatrix}},
        {binding: 3, resource: {buffer: this.resultMatrixBuffer}}
      ]
    });
        */
  }

  compileAndRunSync(
      workGroupSize: [number, number, number], workPerThread = 1) {
    // TODO: figure out how to return non const two values.
    // if (mode == 0) {
    return this.dispatchAndSubmit(
        this.computePipeline, this.bindGroup, this.shape[0], this.shape[1],
        workGroupSize, workPerThread);
  }

  private dispatchAndSubmit(
      computePipeline: any, bindGroup: any, dispatchX: number,
      dispatchY: number, workGroupSize: [number, number, number],
      workPerThread = 1) {
    // TIMESTAMP
    // TODO: necessary to destroy querySet?
    /*
    let querySet: any;
    let dstBuffer: GPUBuffer;
    if (this.enableTimeStamp) {
      querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
      dstBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    */
    // Commands submission
    const commandEncoder = this.device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    /*
    if (this.enableTimeStamp) {
      passEncoder.writeTimestamp(querySet, 0);
    }
    */
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // passEncoder.dispatch(dispatchX, dispatchY);
    if (workGroupSize[1] == 1 && workGroupSize[2] == 1) {
      passEncoder.dispatch(
          dispatchX * dispatchY / workGroupSize[0] / workPerThread /
              workPerThread,
          1);
    } else {
      passEncoder.dispatch(
          dispatchX / workGroupSize[0] / workPerThread,
          dispatchY / workGroupSize[1] / workPerThread);
    }
    /*
    if (this.enableTimeStamp) {
      passEncoder.writeTimestamp(querySet, 1);
    }
    */
    passEncoder.endPass();
    /*
    if (this.enableTimeStamp) {
      commandEncoder.resolveQuerySet(querySet, 0, 2, dstBuffer, 0);
    }
    */
    // Submit GPU commands.

    const gpuCommands = commandEncoder.finish();
    this.device.defaultQueue.submit([gpuCommands]);
    /*
    if (this.enableTimeStamp) {
      await this.getQueryTime(dstBuffer);
    }
    */

    // return (this.now() - start);
    return true;
  }

  async data() {
    const arrayBuffer = await this.getBufferData();
    return new Float32Array(arrayBuffer);
  }

  async getBufferData() {
    // TODO: this is only used in readback.
    const gpuReadBuffer = this.device.createBuffer({
      size: this.resultMatrixBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Commands submission
    const commandEncoder = this.device.createCommandEncoder();
    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
        this.resultMatrixBuffer /* source buffer */, 0 /* source offset */,
        gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
        this.resultMatrixBufferSize /* size */
    );

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.device.defaultQueue.submit([gpuCommands]);
    // Read buffer.
    // const mapped = await gpuReadBuffer.mapReadAsync();
    // const arrayBuffer = mapped.slice(0);
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange().slice(0);
    gpuReadBuffer.unmap();
    gpuReadBuffer.destroy();
    return arrayBuffer;
  }

  // ---------Below code is not used!-------------------
  // Do not USE!
  private async dispatchAndSubmitWithFence(
      computePipeline: any, bindGroup: any, dispatchX: number,
      dispatchY: number, workGroupSize: [number, number, number]) {
    const start = utils.now();
    this.dispatchAndSubmit(
        this.computePipeline, this.bindGroup, this.shape[0], this.shape[1],
        workGroupSize);
    const fence = this.queue.createFence();
    this.queue.signal(fence, 1);
    await fence.onCompletion(1);
    console.log((utils.now() - start).toFixed(2));
  }

  // TODO: Float32Array is bad. And buffer is bad.
  async compileAndRun(workGroupSize: [number, number, number]) {
    // TODO: figure out how to return non const two values.
    // if (mode == 0) {
    return await this.dispatchAndSubmitWithFence(
        this.computePipeline, this.bindGroup, this.shape[0], this.shape[1],
        workGroupSize);
  }

  getBufferKey() {
    return this.bufferID++;
  }

  // Below is not used currently.
  releaseBuffer(buffer: GPUBuffer) {
    if (this.freeBuffers == null) {
      return;
    }

    const key = this.getBufferKey();
    if (!this.freeBuffers.has(key)) {
      this.freeBuffers.set(key, []);
    }

    this.freeBuffers.get(key).push(buffer);
  }
  // Call this after execute.
  disposeReadBackBuffer() {
    if (this.freeBuffers == null) {
      return;
    }

    this.freeBuffers.forEach((buffers, key) => {
      buffers.forEach(buff => {
        // console.log(' freeBuffers destroy key = ' + key);
        buff.unmap();
        buff.destroy();
      });
    });
  }

  dispose() {
    this.disposeReadBackBuffer();
  }
}

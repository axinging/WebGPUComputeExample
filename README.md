## Introduction
This project is used for demo Texture and Buffer difference on WebGPU Compute.

The draft idea is from here: [TFJS Issue](https://github.com/tensorflow/tfjs/issues/3132).

This is the WebGPU version of [VulkanCompute](https://github.com/math3d/VulkanCompute).
## Build
```
yarn & yarn build-npm & yarn link
cd demo
yarn link "@webgpu/compute"
yarn & yarn build
yarn watch
```

## Status
Add support any size except rgba32f. Matmul only support 256 aligned.
```
add_buffer: Pass; Any size;
add_texture(rgba32f): Pass; Must be 256 bytes aligned.
add_texturer32f: Pass; Any size;
matmul_buffer: Pass
matmul_buffervec4: Pass
matmul_packedbuffer: Pass
matmul_texturer32f: Pass
matmul_texturergba32f: Pass
```


## Other usage
Branch TypeScriptStarter demos how to write a TypeScript application from scratch.

## Error fix

In demo, yarn build will complains:
```
Cannot resolve dependency '@webgpu/compute'
```
Fix:
```
yarn build-npm (with rollup)
```

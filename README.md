## Introduction
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
```
add_buffer: Pass
add_texture: Fail
```


## Other usage
Brach TypeScriptStarter demos how to write a TypeScript application from scratch.

## Error fix

In demo, yarn build will complains:
```
Cannot resolve dependency '@webgpu/compute'
```
Fix:
```
yarn build-npm (with rollup)
```

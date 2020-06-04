## Build
```
yarn
yarn build
yarn link
```

cd demo
```
yarn link "@webgpu/compute"
```


## Error fix

In demo, yarn build will complains:
Cannot resolve dependency '@webgpu/compute'
Fix:
yarn build-npm (with rollup)

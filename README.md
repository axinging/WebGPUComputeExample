## Build
```
yarn & yarn build-npm & yarn link
cd demo
yarn link "@webgpu/compute"
yarn & yarn build
yarn watch
```


## Error fix

In demo, yarn build will complains:
```
Cannot resolve dependency '@webgpu/compute'
```
Fix:
```
yarn build-npm (with rollup)
```

// Below code is from:
// https://github.com/axinging/tfjs-examples/commit/fa4b53e06d90b06711e39e7eac13794f4bcd8147
// let oldLog: any;
export function startLog(kernels: any, oldLog: any) {
  console.log = (msg: any) => {
    kernels.push(parseFloat(msg));
  }
}

export async function endLog(kernels: any, oldLog: any, mode = 0) {
  let i = 0;
  if (mode == 1) {
    kernels.forEach((msg: any) => {
      i = i + 1;
      oldLog.call(oldLog, i + ' ' + msg);
    });
  }
  console.log = oldLog;
}

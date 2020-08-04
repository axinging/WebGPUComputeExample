export function createFloat32Array(w, h) {
  let matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = Math.random();  // tf.randomUniform(shape, 0, 2.5);//0.01*i;
  }
  return matrix;
}

export function logTimes(name, times, trials, reps) {
  const times2 = times.map(function(time) {
    return Number(time.toFixed(2));
  });
  console.log(name + times2);
  const mean = times.reduce((a, b) => a + b, 0) / trials;
  const min = Math.min(...times);
  const fmt = (n) => n.toFixed(2);
  console.log(
      name + `Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
  console.log(name + `Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
}

export function createUint32Array(w, h) {
  let matrix = new Uint32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = i;
  }
  return matrix;
}

export async function time(op, execute, opName, doRep, trials = 50, reps = 50) {
  const times = [];

  const trial = async () => {
    for (let r = 0; r < reps; ++r) {
      execute(op);
    }
    await op.data();
    op.dispose();
  };

  // Warm-up. Specifically, this pre-allocates enough memory for an entire
  // trial, ensuring that no allocations happen when timing a trial (if the
  // backend reuses allocations).
  await trial();

  for (let t = 0; t < trials; ++t) {
    const start = performance.now();
    await trial();
    times.push(performance.now() - start);
  }

  logTimes(opName, times, trials, reps);
}

export function executeOp(op) {
  op.executeSync();
}

function compareThreeFloat32Array(a, b, c, w, h) {
    for (let i = 0; i < w * h; i++) {
      if (i == 0) {
        console.log('item 0=' + a[i] + ', ' + b[i] + ',' + c[i]);
      }
      if (Math.abs(a[i] - b[i]) > 0.01 || Math.abs(b[i] - c[i]) > 0.01 ||
          Math.abs(a[i] - c[i]) > 0.01) {
        console.log('Mismatch at ' + i);
        return i;
      }
    }
    return -1;
}
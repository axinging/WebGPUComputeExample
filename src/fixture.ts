
export function now(): number {
  return performance.now();
}

export function expectContents(actual: any, exp: any) {
  const size = exp.byteLength;

  if (actual.byteLength !== size) {
    return 'size mismatch';
  }

  const lines = [];
  let failedPixels = 0;

  for (let i = 0; i < size; ++i) {
    if (actual[i] !== exp[i]) {
      if (failedPixels > 4) {
        lines.push('... and more');
        break;
      }

      failedPixels++;
      lines.push(`at [${i}], expected ${exp[i]}, got ${actual[i]}`);
    }
  }  // TODO: Could make a more convenient message, which could look like e.g.:
  //
  //   Starting at offset 48,
  //              got 22222222 ABCDABCD 99999999
  //     but expected 22222222 55555555 99999999
  //
  // or
  //
  //   Starting at offset 0,
  //              got 00000000 00000000 00000000 00000000 (... more)
  //     but expected 00FF00FF 00FF00FF 00FF00FF 00FF00FF (... more)
  //
  // Or, maybe these diffs aren't actually very useful (given we have the prints
  // just above here), and we should remove them. More important will be logging
  // of texture data in a visual format.


  if (size <= 256 && failedPixels > 0) {
    const expHex =
        Array.from(exp).map(x => x.toString().padStart(2, '0')).join('');
    const actHex =
        Array.from(actual).map(x => x.toString().padStart(2, '0')).join('');
    lines.push('EXPECT: ' + expHex);
    lines.push('ACTUAL: ' + actHex);
  }

  if (failedPixels) {
    return lines.join('\n');
  }

  return undefined;
}

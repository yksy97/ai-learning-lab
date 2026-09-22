// 生成サンプルの良し悪しを測る指標

/** 各点から k 番目に近い（自分以外の）点までの距離 */
function knnRadii(X, n, k) {
  const r = new Float64Array(n);
  const best = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    best.fill(Infinity);
    const xi = X[2 * i], yi = X[2 * i + 1];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = (X[2 * j] - xi) ** 2 + (X[2 * j + 1] - yi) ** 2;
      if (d < best[k - 1]) {
        let m = k - 1;
        while (m > 0 && best[m - 1] > d) { best[m] = best[m - 1]; m--; }
        best[m] = d;
      }
    }
    r[i] = best[k - 1];
  }
  return r; // 距離の2乗
}

/**
 * 改良版 Precision / Recall（Kynkäänniemi et al., 2019）
 *   precision（品質）: 生成点のうち「本物の近く」に落ちた割合
 *   recall（網羅性）  : 本物の点のうち「生成点の近く」にある割合
 * 「近く」は各点の k 近傍までの距離を半径とする球で定義する。
 */
export function precisionRecall(real, fake, n, k = 3) {
  const rr = knnRadii(real, n, k);
  const rf = knnRadii(fake, n, k);
  let p = 0, r = 0;
  for (let j = 0; j < n; j++) {
    const x = fake[2 * j], y = fake[2 * j + 1];
    for (let i = 0; i < n; i++) {
      if ((real[2 * i] - x) ** 2 + (real[2 * i + 1] - y) ** 2 <= rr[i]) { p++; break; }
    }
  }
  for (let i = 0; i < n; i++) {
    const x = real[2 * i], y = real[2 * i + 1];
    for (let j = 0; j < n; j++) {
      if ((fake[2 * j] - x) ** 2 + (fake[2 * j + 1] - y) ** 2 <= rf[j]) { r++; break; }
    }
  }
  return { precision: p / n, recall: r / n };
}

/** 混合ガウスの山のうち、十分な数のサンプルが落ちている山の数 */
export function modesCovered(fake, n, modes) {
  if (!modes) return null;
  const { centers, radius } = modes;
  const counts = new Array(centers.length).fill(0);
  for (let i = 0; i < n; i++) {
    const x = fake[2 * i], y = fake[2 * i + 1];
    for (let m = 0; m < centers.length; m++) {
      if (Math.hypot(x - centers[m][0], y - centers[m][1]) < radius) { counts[m]++; break; }
    }
  }
  const need = Math.max(2, (n / centers.length) * 0.2);
  return { covered: counts.filter((c) => c >= need).length, total: centers.length };
}

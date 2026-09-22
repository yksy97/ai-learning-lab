// 自己回帰モデル用のデータ空間の描画
import { cssVar, hexToRgb, fitCanvas } from './viz.js';
import { RANGE, BINS, V, CELL, binLo } from './ar.js';

export class ARSpaceView {
  constructor(canvas) {
    this.canvas = canvas;
    this.img = document.createElement('canvas');
    this.img.width = this.img.height = BINS;
    this.hasDensity = false;
  }

  toPx(x, y) {
    const s = this.size / (2 * RANGE);
    return [(x + RANGE) * s, (RANGE - y) * s];
  }
  toWorld(px, py) {
    const s = this.size / (2 * RANGE);
    return [px / s - RANGE, RANGE - py / s];
  }
  /** ビン範囲 [[x0,x1],[y0,y1]] → ピクセルの矩形 */
  rectOf(r) {
    const [x0, y1] = this.toPx(binLo(r[0][0]), binLo(r[1][0]));
    const [x1, y0] = this.toPx(binLo(r[0][0]) + (r[0][1] - r[0][0]) * CELL, binLo(r[1][0]) + (r[1][1] - r[1][0]) * CELL);
    return [x0, y0, x1 - x0, y1 - y0];
  }

  /** 各マスの確率を対数スケールの濃淡にする */
  updateDensity(density) {
    const ctx = this.img.getContext('2d');
    const im = ctx.createImageData(BINS, BINS);
    const bg = hexToRgb(cssVar('--surface'));
    const fg = hexToRgb(cssVar('--text'));
    let mx = 0;
    for (const p of density) if (p > mx) mx = p;
    const lo = Math.log(mx * 1e-4), hi = Math.log(mx);
    for (let i1 = 0; i1 < BINS; i1++) {
      for (let i2 = 0; i2 < BINS; i2++) {
        const p = density[i1 * BINS + i2];
        const v = Math.max(0, Math.min(1, (Math.log(Math.max(p, 1e-300)) - lo) / (hi - lo)));
        const t = Math.pow(v, 1.6) * 0.8;
        const k = ((BINS - 1 - i2) * BINS + i1) * 4;
        im.data[k] = bg[0] + (fg[0] - bg[0]) * t;
        im.data[k + 1] = bg[1] + (fg[1] - bg[1]) * t;
        im.data[k + 2] = bg[2] + (fg[2] - bg[2]) * t;
        im.data[k + 3] = 255;
      }
    }
    ctx.putImageData(im, 0, 0);
    this.hasDensity = true;
  }

  draw(s) {
    const { ctx, w } = fitCanvas(this.canvas);
    this.size = w;
    const surface = cssVar('--surface');
    const text = cssVar('--text');
    const ar = cssVar('--ar');
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, w, w);
    if (s.show.density && this.hasDensity) {
      ctx.imageSmoothingEnabled = false; // マス目（量子化）をそのまま見せる
      ctx.drawImage(this.img, 0, 0, w, w);
    }

    // 目盛り
    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    ctx.globalAlpha = s.show.density ? 0.5 : 1;
    ctx.beginPath();
    for (let v = -2; v <= 2; v++) {
      const [px] = this.toPx(v, 0);
      const [, py] = this.toPx(0, v);
      ctx.moveTo(px, 0); ctx.lineTo(px, w);
      ctx.moveTo(0, py); ctx.lineTo(w, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = cssVar('--muted');
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    for (let v = -2; v <= 2; v++) {
      if (v === 0) continue;
      const [px, py0] = this.toPx(v, 0);
      ctx.fillText(String(v), px + 3, py0 + 3);
      const [px0, py] = this.toPx(0, v);
      ctx.fillText(String(v), px0 + 3, py + 3);
    }

    // 粗いマス目（最初のトークンが選ぶ単位）
    if (s.show.grid) {
      ctx.strokeStyle = text;
      ctx.globalAlpha = 0.18;
      ctx.beginPath();
      for (let c = 0; c <= V; c++) {
        const p = (c / V) * w;
        ctx.moveTo(p, 0); ctx.lineTo(p, w);
        ctx.moveTo(0, p); ctx.lineTo(w, p);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const r = Math.max(2, w / 260);
    if (s.show.real && s.real) {
      ctx.fillStyle = cssVar('--real');
      ctx.globalAlpha = 0.5;
      for (let i = 0; i < s.real.length / 2; i++) {
        const [px, py] = this.toPx(s.real[2 * i], s.real[2 * i + 1]);
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    if (s.show.samples && s.samples) {
      ctx.fillStyle = ar;
      ctx.strokeStyle = surface;
      ctx.lineWidth = 1;
      for (let i = 0; i < s.samples.length / 2; i++) {
        const [px, py] = this.toPx(s.samples[2 * i], s.samples[2 * i + 1]);
        ctx.beginPath(); ctx.arc(px, py, r + 0.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }

    // 生成の途中経過：次のトークンの候補 8 個と、確定した領域
    const g = s.gen;
    if (g) {
      if (g.candidates) {
        const mx = Math.max(...g.probs);
        g.candidates.forEach((reg, j) => {
          const [x, y, cw, ch] = this.rectOf(reg);
          const a = g.probs[j] / mx;
          ctx.fillStyle = ar;
          ctx.globalAlpha = 0.08 + 0.5 * a;
          ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
          ctx.globalAlpha = 1;
          ctx.strokeStyle = ar;
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
          if (Math.min(cw, ch) > 26) {
            ctx.fillStyle = text;
            ctx.font = '11px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const label = `${j}: ${(g.probs[j] * 100).toFixed(0)}%`;
            if (cw >= ch) ctx.fillText(label, x + cw / 2, y + 12);
            else {
              ctx.save();
              ctx.translate(x + cw / 2, y + 12);
              ctx.fillText(String(j), 0, 0);
              ctx.fillText(`${(g.probs[j] * 100).toFixed(0)}%`, 0, 14);
              ctx.restore();
            }
          }
        });
      }
      if (g.region) {
        const [x, y, cw, ch] = this.rectOf(g.region);
        ctx.strokeStyle = text;
        ctx.lineWidth = 2;
        ctx.setLineDash(g.complete ? [] : [5, 4]);
        ctx.strokeRect(x, y, cw, ch);
        ctx.setLineDash([]);
        if (g.complete) {
          ctx.beginPath();
          ctx.arc(x + cw / 2, y + ch / 2, 11, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
    // これまでに1点ずつ生成した点
    if (s.trail) {
      ctx.strokeStyle = text;
      ctx.lineWidth = 1.5;
      for (const [x, y] of s.trail) {
        const [px, py] = this.toPx(x, y);
        ctx.beginPath(); ctx.arc(px, py, r + 2.5, 0, Math.PI * 2); ctx.stroke();
      }
    }
    if (this.hover) {
      ctx.strokeStyle = text;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(this.hover[0], this.hover[1], 6, 0, Math.PI * 2); ctx.stroke();
    }
  }
}

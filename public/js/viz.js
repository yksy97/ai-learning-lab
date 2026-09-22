// Canvas 描画ユーティリティ：データ空間・潜在空間・時系列チャート

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const v = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** キャンバスを CSS サイズ × devicePixelRatio に合わせる。論理サイズを返す */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

/** z の先頭2次元から色を決める（角度→色相、半径→明るさ）。対応関係を追うための識別用 */
export function zColor(z0, z1) {
  const hue = ((Math.atan2(z1, z0) * 180) / Math.PI + 360) % 360;
  const r = Math.min(1, Math.hypot(z0, z1) / 2.5);
  return `hsl(${hue.toFixed(0)} 75% ${(68 - r * 22).toFixed(0)}%)`;
}

// ---------------------------------------------------------------------------
// データ空間
// ---------------------------------------------------------------------------
export class SpaceView {
  constructor(canvas, range = 2.2) {
    this.canvas = canvas;
    this.range = range;
    this.heatRes = 56;
    this.heat = document.createElement('canvas');
    this.heat.width = this.heat.height = this.heatRes;
    this.heatPoints = new Float64Array(this.heatRes * this.heatRes * 2);
    for (let j = 0; j < this.heatRes; j++) {
      for (let i = 0; i < this.heatRes; i++) {
        const k = j * this.heatRes + i;
        this.heatPoints[2 * k] = -range + ((i + 0.5) / this.heatRes) * 2 * range;
        this.heatPoints[2 * k + 1] = range - ((j + 0.5) / this.heatRes) * 2 * range;
      }
    }
  }

  toPx(x, y) {
    const s = this.size / (2 * this.range);
    return [(x + this.range) * s, (this.range - y) * s];
  }
  toWorld(px, py) {
    const s = this.size / (2 * this.range);
    return [px / s - this.range, this.range - py / s];
  }

  /** D(x) の確率グリッドから背景画像を作る（0=橙, 0.5=地の色, 1=青） */
  updateHeat(probs) {
    const res = this.heatRes;
    const ctx = this.heat.getContext('2d');
    const img = ctx.createImageData(res, res);
    const mid = hexToRgb(cssVar('--heat-mid'));
    const real = hexToRgb(cssVar('--real'));
    const fake = hexToRgb(cssVar('--fake'));
    for (let k = 0; k < res * res; k++) {
      const p = probs[k];
      const t = Math.pow(Math.min(1, Math.abs(p - 0.5) * 2), 1.3) * 0.5;
      const c = p >= 0.5 ? real : fake;
      img.data[4 * k] = mid[0] + (c[0] - mid[0]) * t;
      img.data[4 * k + 1] = mid[1] + (c[1] - mid[1]) * t;
      img.data[4 * k + 2] = mid[2] + (c[2] - mid[2]) * t;
      img.data[4 * k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    this.heatProbs = probs;
  }

  draw(s) {
    const { ctx, w } = fitCanvas(this.canvas);
    this.size = w;
    const surface = cssVar('--surface');
    const grid = cssVar('--grid');
    const muted = cssVar('--muted');
    const real = cssVar('--real');
    const fake = cssVar('--fake');

    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, w, w);
    if (s.show.heat && this.heatProbs) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.heat, 0, 0, w, w);
    }

    // 座標軸と目盛り（控えめに）
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let v = -2; v <= 2; v++) {
      const [px] = this.toPx(v, 0);
      const [, py] = this.toPx(0, v);
      ctx.moveTo(px, 0); ctx.lineTo(px, w);
      ctx.moveTo(0, py); ctx.lineTo(w, py);
    }
    ctx.stroke();
    ctx.fillStyle = muted;
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

    // 潜在空間の格子を G で写したもの
    if (s.show.grid && s.gridLines) {
      ctx.strokeStyle = fake;
      ctx.globalAlpha = 0.28;
      ctx.lineWidth = 1;
      for (const line of s.gridLines) {
        ctx.beginPath();
        for (let i = 0; i < line.length / 2; i++) {
          const [px, py] = this.toPx(line[2 * i], line[2 * i + 1]);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // 本物のサンプル
    const r = Math.max(2, w / 260);
    if (s.show.real && s.real) {
      ctx.fillStyle = real;
      ctx.globalAlpha = 0.55;
      const n = s.real.length / 2;
      for (let i = 0; i < n; i++) {
        const [px, py] = this.toPx(s.real[2 * i], s.real[2 * i + 1]);
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // 生成サンプル（表面色のリングで重なりを見分けやすく）
    if (s.show.fake && s.fake) {
      const n = s.fake.length / 2;
      let outside = 0;
      for (let i = 0; i < n; i++) {
        let [px, py] = this.toPx(s.fake[2 * i], s.fake[2 * i + 1]);
        const color = s.show.colorByZ ? s.fakeColors[i] : fake;
        if (px < 0 || py < 0 || px > w || py > w) {
          // 表示範囲外の点は縁に中抜きの印で示す
          outside++;
          px = Math.min(w - 5, Math.max(5, px));
          py = Math.min(w - 5, Math.max(5, py));
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(px, py, r + 1, 0, Math.PI * 2); ctx.stroke();
          continue;
        }
        ctx.lineWidth = 1;
        ctx.strokeStyle = surface;
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(px, py, r + 0.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      if (outside > 0) {
        const msg = `${s.fakeLabel || '生成点'}の ${Math.round((outside / n) * 100)}% が表示範囲外（縁の○）`;
        ctx.font = '12px system-ui, sans-serif';
        const tw = ctx.measureText(msg).width;
        ctx.fillStyle = surface;
        ctx.globalAlpha = 0.9;
        ctx.fillRect(10, w - 34, tw + 16, 24);
        ctx.globalAlpha = 1;
        ctx.fillStyle = cssVar('--text');
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(msg, 18, w - 22);
      }
    }

    // G への勾配の矢印
    if (s.show.grad && s.grad) {
      const n = s.grad.length / 2;
      let maxLen = 1e-12;
      for (let i = 0; i < n; i++) maxLen = Math.max(maxLen, Math.hypot(s.grad[2 * i], s.grad[2 * i + 1]));
      const scale = (w * 0.07) / maxLen;
      ctx.strokeStyle = cssVar('--text');
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1.2;
      for (let i = 0; i < n; i++) {
        const [x0, y0] = this.toPx(s.fake[2 * i], s.fake[2 * i + 1]);
        const gx = s.grad[2 * i] * scale, gy = -s.grad[2 * i + 1] * scale;
        const len = Math.hypot(gx, gy);
        if (len < 1.5) continue;
        const x1 = x0 + gx, y1 = y0 + gy;
        const ux = gx / len, uy = gy / len, hl = Math.min(5, len * 0.4);
        ctx.beginPath();
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        ctx.moveTo(x1, y1); ctx.lineTo(x1 - hl * (ux - uy * 0.5), y1 - hl * (uy + ux * 0.5));
        ctx.moveTo(x1, y1); ctx.lineTo(x1 - hl * (ux + uy * 0.5), y1 - hl * (uy - ux * 0.5));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // ホバー位置
    if (this.hover) {
      ctx.strokeStyle = cssVar('--text');
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(this.hover[0], this.hover[1], 6, 0, Math.PI * 2); ctx.stroke();
    }
  }
}

// ---------------------------------------------------------------------------
// 潜在空間
// ---------------------------------------------------------------------------
export function drawLatent(canvas, s) {
  const { ctx, w } = fitCanvas(canvas);
  const R = 3;
  const toPx = (a, b) => [((a + R) / (2 * R)) * w, ((R - b) / (2 * R)) * w];
  ctx.fillStyle = cssVar('--surface-2');
  ctx.fillRect(0, 0, w, w);

  const fake = cssVar('--fake');
  // 格子
  ctx.strokeStyle = fake;
  ctx.globalAlpha = 0.28;
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (s.noiseDim >= 2) {
    for (const v of s.gridVals) {
      let [x0, y0] = toPx(v, -s.gridMax); let [x1, y1] = toPx(v, s.gridMax);
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      [x0, y0] = toPx(-s.gridMax, v); [x1, y1] = toPx(s.gridMax, v);
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
    }
  } else {
    const [x0, y0] = toPx(-s.gridMax, 0); const [x1, y1] = toPx(s.gridMax, 0);
    ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // 点
  const d = s.noiseDim;
  const n = s.z.length / d;
  const r = Math.max(1.6, w / 150);
  for (let i = 0; i < n; i++) {
    const a = s.z[i * d];
    const b = d >= 2 ? s.z[i * d + 1] : (s.jitter[i] - 0.5) * 0.6;
    const [px, py] = toPx(a, b);
    ctx.fillStyle = s.show.colorByZ ? s.fakeColors[i] : fake;
    ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = cssVar('--muted');
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  ctx.fillText(d >= 2 ? 'z₁ →' : 'z →', w - 34, w / 2 + 3);
  if (d >= 2) ctx.fillText('↑ z₂', w / 2 + 4, 4);
}

// ---------------------------------------------------------------------------
// 時系列チャート（1本の y 軸、2系列、終端ラベル、十字線ツールチップ）
// ---------------------------------------------------------------------------
export class LineChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{series: {key:string,label:string,color:string}[], yMin?:number, yMax?:number, refs?: {y:number,label:string}[], fmt?:(v:number)=>string}} opt
   */
  constructor(canvas, opt) {
    this.canvas = canvas;
    this.tip = canvas.parentElement.querySelector('.tip');
    this.opt = opt;
    this.data = null;
    this.hoverX = null;
    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      this.hoverX = e.clientX - r.left;
      this.draw(this.data);
    });
    canvas.addEventListener('pointerleave', () => {
      this.hoverX = null;
      this.tip.hidden = true;
      this.draw(this.data);
    });
  }

  draw(data) {
    this.data = data;
    const { ctx, w, h } = fitCanvas(this.canvas);
    const { series, refs = [], fmt = (v) => v.toFixed(3) } = this.opt;
    const pad = { l: 34, r: 44, t: 8, b: 20 };
    const surface = cssVar('--surface');
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, w, h);
    const muted = cssVar('--muted');
    const grid = cssVar('--grid');
    ctx.font = '11px system-ui, sans-serif';

    const pts = data ? data.points : [];
    let yMin = this.opt.yMin ?? Infinity, yMax = this.opt.yMax ?? -Infinity;
    if (this.opt.yMin == null || this.opt.yMax == null) {
      for (const p of pts) for (const s of series) {
        const v = p[s.key];
        if (this.opt.yMin == null) yMin = Math.min(yMin, v);
        if (this.opt.yMax == null) yMax = Math.max(yMax, v);
      }
      for (const r of refs) { if (this.opt.yMax == null) yMax = Math.max(yMax, r.y); }
      if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
      if (this.opt.yMin == null) yMin = Math.max(0, yMin - 0.05 * (yMax - yMin));
      if (this.opt.yMax == null) yMax = yMax + 0.08 * (yMax - yMin || 1);
    }
    const x0 = pts.length ? pts[0].step : 0;
    const x1 = pts.length ? Math.max(pts[pts.length - 1].step, x0 + 1) : 1;
    const X = (s) => pad.l + ((s - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);

    // 目盛り
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const ticks = niceTicks(yMin, yMax, 3);
    ctx.beginPath();
    for (const t of ticks) {
      const y = Math.round(Y(t)) + 0.5;
      ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y);
      ctx.fillText(formatTick(t), pad.l - 5, y);
    }
    ctx.stroke();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('ステップ', pad.l, h - 5);
    ctx.textAlign = 'right';
    ctx.fillText(String(Math.round(x1)), w - pad.r, h - 5);

    // 参照線（理論的な均衡値）
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = muted;
    for (const r of refs) {
      if (r.y < yMin || r.y > yMax) continue;
      const y = Math.round(Y(r.y)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    if (pts.length < 2) {
      ctx.fillStyle = muted;
      ctx.textAlign = 'center';
      ctx.fillText('学習を開始すると表示されます', (w + pad.l - pad.r) / 2, h / 2);
      return;
    }

    // 系列
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    const ends = [];
    for (const s of series) {
      ctx.strokeStyle = cssVar(s.color);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = X(p.step), y = Y(Math.min(yMax, Math.max(yMin, p[s.key])));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const last = pts[pts.length - 1];
      ends.push({ y: Y(Math.min(yMax, Math.max(yMin, last[s.key]))), text: fmt(last[s.key]), color: cssVar(s.color) });
    }
    // 終端ラベル（重なりを避ける）
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (const e of ends) {
      ctx.fillStyle = e.color;
      ctx.fillRect(w - pad.r + 4, e.y - 1, 5, 2);
      ctx.fillStyle = cssVar('--text-2');
      ctx.fillText(e.text, w - pad.r + 11, e.y);
    }

    // 十字線 + ツールチップ
    if (this.hoverX != null && this.hoverX >= pad.l && this.hoverX <= w - pad.r) {
      const stepAt = x0 + ((this.hoverX - pad.l) / (w - pad.l - pad.r)) * (x1 - x0);
      let best = pts[0];
      for (const p of pts) if (Math.abs(p.step - stepAt) < Math.abs(best.step - stepAt)) best = p;
      const x = X(best.step);
      ctx.strokeStyle = cssVar('--text-2');
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      for (const s of series) {
        ctx.fillStyle = cssVar(s.color);
        ctx.strokeStyle = surface;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, Y(Math.min(yMax, Math.max(yMin, best[s.key]))), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      this.tip.innerHTML = `<div class="k">ステップ ${Math.round(best.step).toLocaleString()}</div>` +
        series.map((s) => `<div><i style="background:${cssVar(s.color)}"></i>${s.label} <b>${fmt(best[s.key])}</b></div>`).join('');
      this.tip.hidden = false;
      const tw = this.tip.offsetWidth;
      this.tip.style.left = `${x + 10 + tw > w ? x - tw - 10 : x + 10}px`;
      this.tip.style.top = '6px';
    } else if (this.tip) {
      this.tip.hidden = true;
    }
  }
}

function niceTicks(min, max, count) {
  const span = max - min;
  if (span <= 0) return [min];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function formatTick(v) {
  return Math.abs(v) >= 10 ? v.toFixed(0) : +v.toFixed(2) + '';
}

// VAE ページの描画：データ空間（密度・再構成）と潜在空間（楕円）
import { cssVar, hexToRgb, fitCanvas } from './viz.js';

const RANGE = 2.2;
const HEAT_RES = 48;

export class VAESpaceView {
  constructor(canvas) {
    this.canvas = canvas;
    this.res = HEAT_RES;
    this.heat = document.createElement('canvas');
    this.heat.width = this.heat.height = HEAT_RES;
    this.points = new Float64Array(HEAT_RES * HEAT_RES * 2);
    for (let j = 0; j < HEAT_RES; j++) {
      for (let i = 0; i < HEAT_RES; i++) {
        const k = j * HEAT_RES + i;
        this.points[2 * k] = -RANGE + ((i + 0.5) / HEAT_RES) * 2 * RANGE;
        this.points[2 * k + 1] = RANGE - ((j + 0.5) / HEAT_RES) * 2 * RANGE;
      }
    }
  }

  toPx(x, y) {
    const s = this.size / (2 * RANGE);
    return [(x + RANGE) * s, (RANGE - y) * s];
  }
  toWorld(px, py) {
    const s = this.size / (2 * RANGE);
    return [px / s - RANGE, RANGE - py / s];
  }

  /** デコーダから推定した p(x) を濃淡にする */
  updateDensity(dens) {
    const ctx = this.heat.getContext('2d');
    const im = ctx.createImageData(HEAT_RES, HEAT_RES);
    const bg = hexToRgb(cssVar('--surface'));
    const fg = hexToRgb(cssVar('--text'));
    let mx = 0;
    for (const p of dens) if (p > mx) mx = p;
    const lo = Math.log(Math.max(mx, 1e-12) * 1e-3), hi = Math.log(Math.max(mx, 1e-12));
    for (let k = 0; k < dens.length; k++) {
      const v = Math.max(0, Math.min(1, (Math.log(Math.max(dens[k], 1e-300)) - lo) / (hi - lo)));
      const t = Math.pow(v, 1.6) * 0.75;
      im.data[4 * k] = bg[0] + (fg[0] - bg[0]) * t;
      im.data[4 * k + 1] = bg[1] + (fg[1] - bg[1]) * t;
      im.data[4 * k + 2] = bg[2] + (fg[2] - bg[2]) * t;
      im.data[4 * k + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    this.hasDensity = true;
  }

  draw(s) {
    const { ctx, w } = fitCanvas(this.canvas);
    this.size = w;
    const surface = cssVar('--surface');
    const vae = cssVar('--vae');
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, w, w);
    if (s.show.density && this.hasDensity) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.heat, 0, 0, w, w);
    }

    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let v = -2; v <= 2; v++) {
      const [px] = this.toPx(v, 0);
      const [, py] = this.toPx(0, v);
      ctx.moveTo(px, 0); ctx.lineTo(px, w);
      ctx.moveTo(0, py); ctx.lineTo(w, py);
    }
    ctx.stroke();
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

    // デコーダが潜在空間の格子をどう写しているか
    if (s.show.grid && s.gridLines) {
      ctx.strokeStyle = vae;
      ctx.globalAlpha = 0.3;
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

    const r = Math.max(2, w / 260);
    // 再構成：元の点 → 復元された点
    if (s.show.recon && s.recon) {
      ctx.strokeStyle = vae;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      const n = s.recon.length / 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const [x0, y0] = this.toPx(s.real[2 * i], s.real[2 * i + 1]);
        const [x1, y1] = this.toPx(s.recon[2 * i], s.recon[2 * i + 1]);
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (s.show.real && s.real) {
      ctx.fillStyle = cssVar('--real');
      ctx.globalAlpha = 0.5;
      for (let i = 0; i < s.real.length / 2; i++) {
        const [px, py] = this.toPx(s.real[2 * i], s.real[2 * i + 1]);
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    // 生成サンプルは「中抜きの丸」で、本物（塗りつぶし）と形でも区別する
    if (s.show.samples && s.samples) {
      ctx.strokeStyle = vae;
      ctx.lineWidth = 1.6;
      let outside = 0;
      for (let i = 0; i < s.samples.length / 2; i++) {
        let [px, py] = this.toPx(s.samples[2 * i], s.samples[2 * i + 1]);
        if (px < 0 || py < 0 || px > w || py > w) {
          outside++;
          px = Math.min(w - 5, Math.max(5, px));
          py = Math.min(w - 5, Math.max(5, py));
        }
        ctx.beginPath(); ctx.arc(px, py, r + 0.5, 0, Math.PI * 2); ctx.stroke();
      }
      if (outside > 0) {
        const msg = `生成点の ${Math.round((outside / (s.samples.length / 2)) * 100)}% が表示範囲外`;
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
    // 選んだ 1 点（元の点と、その復元）
    if (s.pick) {
      const [x0, y0] = this.toPx(s.pick.x[0], s.pick.x[1]);
      ctx.strokeStyle = cssVar('--text');
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x0, y0, 7, 0, Math.PI * 2); ctx.stroke();
      if (s.pick.xh) {
        const [x1, y1] = this.toPx(s.pick.xh[0], s.pick.xh[1]);
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(x1, y1, 5, 0, Math.PI * 2); ctx.stroke();
      }
    }
  }
}

/**
 * 潜在空間。各データ点は「中心 μ と広がり σ の楕円」として置かれる。
 * 事前分布 N(0, I) の 1σ・2σ の円も描く。
 */
export function drawLatent(canvas, s) {
  const { ctx, w } = fitCanvas(canvas);
  const R = 3.2;
  const toPx = (a, b) => [((a + R) / (2 * R)) * w, ((R - b) / (2 * R)) * w];
  const scale = w / (2 * R);
  ctx.fillStyle = cssVar('--surface-2');
  ctx.fillRect(0, 0, w, w);

  // 事前分布の目安
  ctx.strokeStyle = cssVar('--muted');
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  for (const rad of [1, 2]) {
    ctx.beginPath();
    ctx.arc(...toPx(0, 0), rad * scale, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = cssVar('--muted');
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const [cx, cy] = toPx(0, 0);
  ctx.fillText('1σ', cx + scale + 3, cy);
  ctx.fillText('2σ', cx + 2 * scale + 3, cy);

  const vae = cssVar('--vae');
  const L = s.latentDim;
  // 各点の q(z|x)：中心と 1σ の楕円
  if (s.mu) {
    const n = s.mu.length / L;
    ctx.strokeStyle = vae;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const a = s.mu[i * L], b = L >= 2 ? s.mu[i * L + 1] : 0;
      const sa = s.sd[i * L], sb = L >= 2 ? s.sd[i * L + 1] : 0.02;
      const [px, py] = toPx(a, b);
      ctx.beginPath();
      ctx.ellipse(px, py, Math.max(1, sa * scale), Math.max(1, sb * scale), 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = cssVar('--real');
    for (let i = 0; i < n; i++) {
      const [px, py] = toPx(s.mu[i * L], L >= 2 ? s.mu[i * L + 1] : 0);
      ctx.beginPath(); ctx.arc(px, py, 1.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
  // 選んだ点の楕円を強調
  if (s.pick && s.pick.mu) {
    const [px, py] = toPx(s.pick.mu[0], L >= 2 ? s.pick.mu[1] : 0);
    ctx.strokeStyle = cssVar('--text');
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(px, py, Math.max(2, s.pick.sd[0] * scale), Math.max(2, (L >= 2 ? s.pick.sd[1] : 0.05) * scale), 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = cssVar('--muted');
  ctx.fillText(L >= 2 ? 'z₁ →' : 'z →', w - 30, w / 2 - 8);
}

// RAG の地図：資料と質問の「近さ」を2次元に置いて見せる
import { cssVar, fitCanvas } from './viz.js';

export class RagMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.hover = null;
  }

  toPx(x, y) {
    const p = this.pad;
    return [p + ((x + 1) / 2) * (this.w - 2 * p), p + ((1 - y) / 2) * (this.h - 2 * p)];
  }

  /** @param {{coords, chunks, query, picked:Set<number>, scores:number[]}} s */
  draw(s) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    this.w = w; this.h = h;
    this.pad = 26;
    ctx.fillStyle = cssVar('--surface');
    ctx.fillRect(0, 0, w, h);
    const muted = cssVar('--muted');
    const text = cssVar('--text');
    const accent = cssVar('--accent');

    ctx.fillStyle = muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('近いものどうしが近くに置かれた「資料の地図」（2次元に押しつぶした近似）', 8, 6);

    // 質問から上位の資料へ線を引く
    if (s.query) {
      const [qx, qy] = this.toPx(s.query[0], s.query[1]);
      ctx.strokeStyle = accent;
      for (const i of s.picked) {
        const [px, py] = this.toPx(s.coords[i][0], s.coords[i][1]);
        ctx.lineWidth = 1 + 3 * Math.min(1, s.scores[i] / 0.5);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(qx, qy); ctx.lineTo(px, py);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // 資料
    s.coords.forEach((c, i) => {
      const [px, py] = this.toPx(c[0], c[1]);
      const on = s.picked.has(i);
      const sc = s.scores ? s.scores[i] : 0;
      ctx.beginPath();
      ctx.arc(px, py, on ? 7 : 4.5, 0, Math.PI * 2);
      if (on) {
        ctx.fillStyle = accent;
        ctx.fill();
        ctx.strokeStyle = cssVar('--surface');
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.fillStyle = muted;
        ctx.globalAlpha = 0.35 + 0.5 * Math.min(1, sc / 0.4);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    });

    // 質問（ひし形）
    if (s.query) {
      const [qx, qy] = this.toPx(s.query[0], s.query[1]);
      ctx.save();
      ctx.translate(qx, qy);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = text;
      ctx.fillRect(-7, -7, 14, 14);
      ctx.restore();
      ctx.fillStyle = text;
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('質問', qx, qy - 11);
    }

    if (this.hover != null && s.coords[this.hover]) {
      const [px, py] = this.toPx(s.coords[this.hover][0], s.coords[this.hover][1]);
      ctx.strokeStyle = text;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, 10, 0, Math.PI * 2); ctx.stroke();
    }
  }

  /** マウス位置にいちばん近い資料の番号 */
  pick(px, py, coords) {
    let best = -1, bd = 18;
    coords.forEach((c, i) => {
      const [x, y] = this.toPx(c[0], c[1]);
      const d = Math.hypot(x - px, y - py);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }
}

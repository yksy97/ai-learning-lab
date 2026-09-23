// 第0層：数式に出てくる「道具」を、スライダーで動かして体感する。
// log・確率の積と対数の和・交差エントロピー・内積とコサイン類似度の4つ。
import { fitCanvas, cssVar } from './viz.js';
import { mountHeader, enableGlossary } from './common.js';
import { addReviewTerms } from './progress.js';

mountHeader({ id: 'math0', mode: 'detail', sub: '式に出てくる道具を、動かして体に入れる' });

const $ = (id) => document.getElementById(id);
const fix = (v, n = 3) => v.toFixed(n);
const redraw = [];
function onResize() {
  for (const f of redraw) f();
}
window.addEventListener('resize', onResize);
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', onResize);

// ---------------------------------------------------------------------------
// 道具1：−log p は「驚きの大きさ」
// ---------------------------------------------------------------------------
{
  const canvas = $('cLog');
  const pIn = $('logP');
  const draw = () => {
    const p = +pIn.value;
    const { ctx, w, h } = fitCanvas(canvas);
    const pad = { l: 34, r: 10, t: 10, b: 20 };
    const YMAX = 5;
    const X = (v) => pad.l + v * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - (Math.min(v, YMAX) / YMAX) * (h - pad.t - pad.b);
    ctx.clearRect(0, 0, w, h);

    // 目盛り
    ctx.strokeStyle = cssVar('--grid');
    ctx.fillStyle = cssVar('--muted');
    ctx.font = '10px system-ui, sans-serif';
    ctx.lineWidth = 1;
    for (let v = 0; v <= YMAX; v++) {
      ctx.beginPath();
      ctx.moveTo(pad.l, Y(v));
      ctx.lineTo(w - pad.r, Y(v));
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(String(v), pad.l - 5, Y(v) + 3);
    }
    ctx.textAlign = 'center';
    for (const t of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(t === 0 ? '0' : String(t), X(t), h - 6);

    // 曲線 −log p
    ctx.strokeStyle = cssVar('--accent');
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= 300; i++) {
      const q = 0.002 + (i / 300) * 0.998;
      const y = -Math.log(q);
      if (y > YMAX) { started = false; continue; }
      const px = X(q), py = Y(y);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // 現在地
    const y = -Math.log(p);
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = cssVar('--text-2');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(p), h - pad.b);
    ctx.lineTo(X(p), Y(y));
    ctx.lineTo(pad.l, Y(y));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar('--accent');
    ctx.beginPath();
    ctx.arc(X(p), Y(y), 4.5, 0, Math.PI * 2);
    ctx.fill();

    $('logPv').textContent = fix(p, 2);
    $('logOut').textContent = fix(y);
    $('logBits').textContent = fix(y / Math.LN2, 1);
    const ex = p >= 0.99 ? '「明日も太陽が昇る」くらい当たり前。知っても情報はほぼゼロ'
      : p >= 0.4 ? 'コイン投げくらい。「へえ」で済む'
      : p >= 0.1 ? 'サイコロで狙った目が出たくらい。少し驚く'
      : p >= 0.02 ? 'トランプ52枚から1枚を当てたくらい。かなり驚く'
      : 'ほぼ起きないはずのことが起きた。とても驚く';
    $('logEx').textContent = ex;
  };
  pIn.addEventListener('input', draw);
  redraw.push(draw);
  draw();
}

// ---------------------------------------------------------------------------
// 道具2：確率の掛け算は、log で足し算になる
// ---------------------------------------------------------------------------
{
  const canvas = $('cChain');
  const pIn = $('chainP'), nIn = $('chainN');
  const draw = () => {
    const p = +pIn.value, n = +nIn.value;
    const logp = Math.log(p);
    const total = n * logp;              // log p^n（足し算で求めた値）
    const naive = Math.pow(p, n);        // 素直に掛け算した値
    const log10 = total / Math.LN10;

    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const gap = 26;
    const ph = (h - gap) / 2;
    const pad = { l: 44, r: 10 };
    const X = (k) => pad.l + (k / n) * (w - pad.l - pad.r);

    const panel = (top, title, valAt, lo, hi, fmt) => {
      ctx.fillStyle = cssVar('--muted');
      ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(title, pad.l, top + 9);
      const y0 = top + 16, y1 = top + ph - 12;
      const Y = (v) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);
      ctx.strokeStyle = cssVar('--grid');
      ctx.lineWidth = 1;
      ctx.textAlign = 'right';
      for (let i = 0; i <= 2; i++) {
        const v = lo + ((hi - lo) * i) / 2;
        ctx.beginPath();
        ctx.moveTo(pad.l, Y(v));
        ctx.lineTo(w - pad.r, Y(v));
        ctx.stroke();
        ctx.fillStyle = cssVar('--muted');
        ctx.fillText(fmt(v), pad.l - 5, Y(v) + 3);
      }
      ctx.strokeStyle = cssVar('--accent');
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let k = 0; k <= n; k++) {
        const py = Y(Math.max(lo, Math.min(hi, valAt(k))));
        if (k === 0) ctx.moveTo(X(k), py); else ctx.lineTo(X(k), py);
      }
      ctx.stroke();
    };

    panel(0, 'ふつうの目盛り：p を n 回掛けた値', (k) => Math.pow(p, k), 0, 1, (v) => v.toFixed(1));
    panel(ph + gap, '対数の目盛り：log p を n 個足した値', (k) => k * logp, Math.min(total, -1), 0,
      (v) => v.toFixed(0));

    $('chainPv').textContent = fix(p, 2);
    $('chainNv').textContent = String(n);
    $('chainMul').textContent = naive === 0 ? '0 になった' : naive.toExponential(2);
    $('chainSum').textContent = fix(total, 2);
    $('chainBack').textContent = naive >= 1e-4
      ? `およそ ${naive.toPrecision(3)}`
      : `小数点以下に 0 が ${Math.floor(-log10)} 個ならんだ先に、やっと数字が出てくる大きさ`;
    $('chainWarn').hidden = naive !== 0;
  };
  pIn.addEventListener('input', draw);
  nIn.addEventListener('input', draw);
  redraw.push(draw);
  draw();
}

// ---------------------------------------------------------------------------
// 道具3：softmax と交差エントロピー
// ---------------------------------------------------------------------------
{
  const LABELS = ['ねこ', 'いぬ', 'とり'];
  const zIn = [$('z0'), $('z1'), $('z2')];
  let answer = 0;

  const draw = () => {
    const z = zIn.map((el) => +el.value);
    const m = Math.max(...z);
    const ex = z.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    const p = ex.map((v) => v / s);
    const loss = -Math.log(p[answer]);

    $('ceBars').innerHTML = p.map((v, i) => `
      <div class="ce-row${i === answer ? ' on' : ''}">
        <span class="ce-lbl">${LABELS[i]}${i === answer ? '（正解）' : ''}</span>
        <span class="ce-bar"><i style="width:${(v * 100).toFixed(1)}%"></i></span>
        <span class="ce-val">${fix(v)}</span>
      </div>`).join('');
    for (let i = 0; i < 3; i++) $(`z${i}v`).textContent = z[i].toFixed(1);
    $('ceLoss').textContent = fix(loss);
    $('ceP').textContent = fix(p[answer]);
    $('ceNote').textContent =
      loss < 0.2 ? '正解にほぼ全部の確率を置けている。損失はほぼ 0。'
      : loss < 1.1 ? 'まだ迷っている。損失は「正解の確率の驚き」ぶんだけ残る。'
      : '正解に置いた確率が小さい。モデルは「まさか」と驚いていて、損失が大きい。';
  };

  for (const el of zIn) el.addEventListener('input', draw);
  $('ceAns').innerHTML = LABELS.map((l, i) => `<button data-i="${i}" aria-pressed="${i === 0}">${l}</button>`).join('');
  $('ceAns').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    answer = +b.dataset.i;
    for (const x of $('ceAns').querySelectorAll('button')) x.setAttribute('aria-pressed', String(+x.dataset.i === answer));
    draw();
  });
  draw();
}

// ---------------------------------------------------------------------------
// 道具4：内積とコサイン類似度（向きが同じなら「似ている」）
// ---------------------------------------------------------------------------
{
  const canvas = $('cVec');
  const R = 1.35;               // 表示範囲
  let a = { x: 1.0, y: 0.35 };
  let b = { x: 0.45, y: 0.95 };
  let drag = null;

  const toPx = (v, size) => [(v.x + R) * (size / (2 * R)), (R - v.y) * (size / (2 * R))];
  const toWorld = (px, py, size) => ({ x: (px / (size / (2 * R))) - R, y: R - py / (size / (2 * R)) });

  const arrow = (ctx, size, v, color, label) => {
    const [ox, oy] = toPx({ x: 0, y: 0 }, size);
    const [px, py] = toPx(v, size);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(px, py);
    ctx.stroke();
    const ang = Math.atan2(py - oy, px - ox);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - 10 * Math.cos(ang - 0.4), py - 10 * Math.sin(ang - 0.4));
    ctx.lineTo(px - 10 * Math.cos(ang + 0.4), py - 10 * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, px + 14 * Math.cos(ang), py + 14 * Math.sin(ang) + 4);
  };

  const draw = () => {
    const { ctx, w, h } = fitCanvas(canvas);
    const size = Math.min(w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate((w - size) / 2, (h - size) / 2);

    // 目盛り
    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    for (let i = -1; i <= 1; i++) {
      const [gx] = toPx({ x: i, y: 0 }, size);
      const [, gy] = toPx({ x: 0, y: i }, size);
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(size, gy); ctx.stroke();
    }
    const [ox, oy] = toPx({ x: 0, y: 0 }, size);
    ctx.strokeStyle = cssVar('--border');
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(size, oy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, size); ctx.stroke();

    // なす角
    const th1 = Math.atan2(a.y, a.x), th2 = Math.atan2(b.y, b.x);
    ctx.strokeStyle = cssVar('--muted');
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(ox, oy, size * 0.13, -Math.max(th1, th2), -Math.min(th1, th2));
    ctx.stroke();

    arrow(ctx, size, a, cssVar('--real'), 'a');
    arrow(ctx, size, b, cssVar('--fake'), 'b');
    ctx.restore();

    const dot = a.x * b.x + a.y * b.y;
    const la = Math.hypot(a.x, a.y), lb = Math.hypot(b.x, b.y);
    const cos = dot / (la * lb || 1);
    $('vDot').textContent = fix(dot);
    $('vLa').textContent = fix(la);
    $('vLb').textContent = fix(lb);
    $('vCos').textContent = fix(cos);
    $('vDeg').textContent = `${((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI).toFixed(0)}°`;
    $('vNote').textContent =
      cos > 0.9 ? 'ほぼ同じ向き。RAG ならこの2つは「同じことを言っている資料」と判定される。'
      : cos > 0.4 ? '少し向きがずれている。関係はあるが、ぴったりではない。'
      : cos > -0.2 ? 'ほぼ直角。共通する成分がない＝無関係、という扱いになる。'
      : '逆向き。言葉の世界では、ここまで負になることはあまりない。';
  };

  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    const size = Math.min(r.width, r.height);
    const px = e.clientX - r.left - (r.width - size) / 2;
    const py = e.clientY - r.top - (r.height - size) / 2;
    return { p: toWorld(px, py, size), size };
  };
  canvas.addEventListener('pointerdown', (e) => {
    const { p } = pick(e);
    const da = Math.hypot(p.x - a.x, p.y - a.y), db = Math.hypot(p.x - b.x, p.y - b.y);
    drag = da < db ? 'a' : 'b';
    canvas.setPointerCapture(e.pointerId);
    const v = drag === 'a' ? a : b;
    v.x = Math.max(-R, Math.min(R, p.x));
    v.y = Math.max(-R, Math.min(R, p.y));
    draw();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const { p } = pick(e);
    const v = drag === 'a' ? a : b;
    v.x = Math.max(-R, Math.min(R, p.x));
    v.y = Math.max(-R, Math.min(R, p.y));
    draw();
  });
  const stop = () => { drag = null; };
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);

  $('vDouble').addEventListener('click', () => {
    b = { x: b.x * 2, y: b.y * 2 };
    if (Math.hypot(b.x, b.y) > R) { const s = (R * 0.95) / Math.hypot(b.x, b.y); b = { x: b.x * s, y: b.y * s }; }
    draw();
  });
  $('vSame').addEventListener('click', () => {
    const la = Math.hypot(a.x, a.y);
    b = { x: (a.x / la) * 0.6, y: (a.y / la) * 0.6 };
    draw();
  });
  $('vOrth').addEventListener('click', () => {
    const la = Math.hypot(a.x, a.y);
    b = { x: (-a.y / la) * 0.8, y: (a.x / la) * 0.8 };
    draw();
  });
  redraw.push(draw);
  draw();
}

// この4つは復習キューに入れておく
addReviewTerms(['log', 'prob', 'ce', 'softmax', 'dot', 'cosine']);
// 用語リンクは本文だけに付ける（数値の読み取り欄に付くと読みにくい）
enableGlossary(['.hero', '.tool-body > p', '.tool-body > .note']);

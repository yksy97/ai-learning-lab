// CNN：型紙（カーネル）を画像じゅうで使い回して形を見つける。
import { CNN, CLASSES, makeImage, S, K, SP } from './cnn.js';
import { mulberry32 } from './nn.js';
import { cssVar } from './viz.js';
import { mountHeader, mountGoal, enableGlossary } from './common.js';
import { renderQuiz } from './quiz.js';
import { renderExplain } from './explain.js';
import { codeBox } from './codebox.js';

mountHeader({ id: 'cnn', mode: 'detail', sub: '3×3 の型紙を画像の上でずらしながら当てて、形を見つけます' });
mountGoal('cnn');

const $ = (id) => document.getElementById(id);
const TARGET = 400;

let net = new CNN();
let job = null;
let current = null;          // いま見ている1枚
const drawn = new Float64Array(S * S);

$('params').textContent = net.paramCount().toLocaleString();
$('nf').textContent = String(net.F);

// ---------------------------------------------------------------------------
// 描画ユーティリティ
// ---------------------------------------------------------------------------
/** 0〜1 の値をグレーで描く */
function drawGray(canvas, data, n, px = 8) {
  canvas.width = canvas.height = n * px * (window.devicePixelRatio || 1);
  canvas.style.width = canvas.style.height = `${n * px}px`;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bg = cssVar('--surface');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, n * px, n * px);
  let max = 0;
  for (const v of data) max = Math.max(max, v);
  const scale = max > 0 ? 1 / max : 1;
  const dark = document.documentElement.getAttribute('data-theme') === 'dark'
    || matchMedia('(prefers-color-scheme: dark)').matches;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const v = Math.max(0, Math.min(1, data[y * n + x] * scale));
      const c = dark ? Math.round(v * 235) : Math.round(255 - v * 235);
      ctx.fillStyle = `rgb(${c} ${c} ${c})`;
      ctx.fillRect(x * px, y * px, px, px);
    }
  }
}

/** 正負のある重みを、青（負）〜白〜赤（正）で描く */
function drawSigned(canvas, data, n, px = 14) {
  canvas.width = canvas.height = n * px * (window.devicePixelRatio || 1);
  canvas.style.width = canvas.style.height = `${n * px}px`;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  let max = 1e-6;
  for (const v of data) max = Math.max(max, Math.abs(v));
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const v = data[y * n + x] / max;
      // 正は赤、負は青。彩度で強さを表す
      const t = Math.abs(v);
      const [r, g, b] = v >= 0 ? [220, 90, 60] : [50, 110, 210];
      const base = cssVar('--surface-2') || '#eee';
      ctx.fillStyle = base;
      ctx.fillRect(x * px, y * px, px, px);
      ctx.globalAlpha = t;
      ctx.fillStyle = `rgb(${r} ${g} ${b})`;
      ctx.fillRect(x * px, y * px, px, px);
      ctx.globalAlpha = 1;
    }
  }
}

function bars(el, probs, pred) {
  el.innerHTML = CLASSES.map((c, i) => `
    <div class="lm-row">
      <span class="lm-ch">${c.label}</span>
      <span class="lm-bar"><i class="hot" style="width:${(probs[i] * 100).toFixed(1)}%;${i === pred ? '' : 'opacity:.45'}"></i></span>
      <span class="lm-p">${(probs[i] * 100).toFixed(1)}%</span>
    </div>`).join('');
}

// ---------------------------------------------------------------------------
// 材料のサンプル
// ---------------------------------------------------------------------------
function renderSamples() {
  const r = mulberry32(3);
  $('samples').innerHTML = CLASSES.map((c) => `
    <div class="cnn-cls">
      <div class="cnn-cls-name">${c.label}</div>
      <div class="cnn-row">${[0, 1, 2, 3].map(() => '<canvas class="cnn-img"></canvas>').join('')}</div>
    </div>`).join('');
  const cls = [...$('samples').querySelectorAll('.cnn-cls')];
  cls.forEach((el, ci) => {
    for (const cv of el.querySelectorAll('canvas')) drawGray(cv, makeImage(CLASSES[ci].id, r), S, 5);
  });
}

// ---------------------------------------------------------------------------
// カーネル・混同行列
// ---------------------------------------------------------------------------
function renderKernels() {
  $('kernels').innerHTML = Array.from({ length: net.F }, (_, f) =>
    `<figure><canvas></canvas><figcaption>${f + 1}</figcaption></figure>`).join('');
  [...$('kernels').querySelectorAll('canvas')].forEach((cv, f) => {
    drawSigned(cv, net.w.slice(f * K * K, (f + 1) * K * K), K, 16);
  });
}

function renderConf(ev) {
  const head = `<tr><th></th>${CLASSES.map((c) => `<th>${c.label}</th>`).join('')}</tr>`;
  const rows = CLASSES.map((c, i) => `
    <tr><th>${c.label}</th>${ev.conf[i].map((n, j) => {
      const on = n > 0 ? Math.min(1, n / 20) : 0;
      const good = i === j;
      return `<td style="background:color-mix(in oklab, ${good ? 'var(--ar)' : 'var(--fake)'} ${Math.round(on * 55)}%, transparent)">${n || ''}</td>`;
    }).join('')}</tr>`).join('');
  $('conf').innerHTML = head + rows;
}

// ---------------------------------------------------------------------------
// 1枚を通してみる
// ---------------------------------------------------------------------------
function showOne(img) {
  current = img;
  drawGray($('inImg'), img, S, 11);
  const r = net.predict(img);
  $('maps').innerHTML = Array.from({ length: net.F }, (_, f) =>
    `<figure><canvas></canvas><figcaption>${f + 1}</figcaption></figure>`).join('');
  [...$('maps').querySelectorAll('canvas')].forEach((cv, f) => {
    drawGray(cv, r.relu.slice(f * S * S, (f + 1) * S * S), S, 4);
  });
  $('pools').innerHTML = Array.from({ length: net.F }, (_, f) =>
    `<figure><canvas></canvas><figcaption>${f + 1}</figcaption></figure>`).join('');
  [...$('pools').querySelectorAll('canvas')].forEach((cv, f) => {
    drawGray(cv, r.pool.slice(f * SP * SP, (f + 1) * SP * SP), SP, 8);
  });
  bars($('probs'), r.probs, r.pred);
}

$('pick').innerHTML = CLASSES.map((c) => `<button data-id="${c.id}">${c.label}を1枚</button>`).join('');
$('pick').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  showOne(makeImage(b.dataset.id, Math.random));
});

// ---------------------------------------------------------------------------
// 学習
// ---------------------------------------------------------------------------
function refreshStats() {
  $('stStep').textContent = net.step.toLocaleString();
  $('stLoss').textContent = net.last ? net.last.loss.toFixed(3) : '–';
  const ev = net.evaluate(80);
  $('stAcc').textContent = `${Math.round(ev.acc * 100)}%`;
  renderConf(ev);
  renderKernels();
  if (current) showOne(current);
  renderDrawPrediction();
}

function tick() {
  if (!job) return;
  const t0 = performance.now();
  while (performance.now() - t0 < 24 && job.done < TARGET) {
    net.trainStep();
    job.done++;
  }
  $('prog').querySelector('i').style.width = `${Math.round((job.done / TARGET) * 100)}%`;
  $('prog').querySelector('span').textContent = `${job.done} / ${TARGET} ステップ`;
  $('stStep').textContent = net.step.toLocaleString();
  $('stLoss').textContent = net.last ? net.last.loss.toFixed(3) : '–';
  renderKernels();
  if (job.done >= TARGET) {
    job = null;
    $('prog').hidden = true;
    $('btnTrain').disabled = false;
    refreshStats();
  } else {
    requestAnimationFrame(tick);
  }
}

$('btnTrain').addEventListener('click', () => {
  job = { done: 0 };
  $('prog').hidden = false;
  $('btnTrain').disabled = true;
  requestAnimationFrame(tick);
});

$('btnReset').addEventListener('click', () => {
  net = new CNN({ seed: Math.floor(Math.random() * 1000) + 1 });
  refreshStats();
});

// ---------------------------------------------------------------------------
// 自分で描く
// ---------------------------------------------------------------------------
{
  const cv = $('draw');
  const PX = 16;
  const paint = (e) => {
    const rect = cv.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * S;
    const y = ((e.clientY - rect.top) / rect.height) * S;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const xi = Math.floor(x) + dx, yi = Math.floor(y) + dy;
        if (xi < 0 || yi < 0 || xi >= S || yi >= S) continue;
        const d = Math.hypot(dx, dy);
        drawn[yi * S + xi] = Math.min(1, drawn[yi * S + xi] + (d === 0 ? 0.9 : d < 1.5 ? 0.35 : 0.12));
      }
    }
    drawGray(cv, drawn, S, PX);
    renderDrawPrediction();
  };
  let down = false;
  cv.addEventListener('pointerdown', (e) => { down = true; cv.setPointerCapture(e.pointerId); paint(e); });
  cv.addEventListener('pointermove', (e) => { if (down) paint(e); });
  cv.addEventListener('pointerup', () => { down = false; });
  cv.addEventListener('pointercancel', () => { down = false; });
  $('btnClear').addEventListener('click', () => {
    drawn.fill(0);
    drawGray(cv, drawn, S, PX);
    renderDrawPrediction();
  });
  drawGray(cv, drawn, S, PX);
}

function renderDrawPrediction() {
  let any = false;
  for (const v of drawn) if (v > 0.05) { any = true; break; }
  if (!any) {
    $('drawProbs').innerHTML = '';
    $('drawNote').textContent = net.step
      ? 'ここに描くと、その場で答えが出ます。同じ形を場所を変えて描いてみてください。'
      : '先に「学習させる」を押してから描くと、ちゃんと答えが返ります。';
    return;
  }
  const r = net.predict(drawn);
  bars($('drawProbs'), r.probs, r.pred);
  $('drawNote').textContent = net.step
    ? `いちばん強いのは「${CLASSES[r.pred].label}」です。位置を変えても答えが変わらないか試してください。`
    : 'まだ学習していないので、この答えはでたらめです。';
}

// ---------------------------------------------------------------------------
// 確認する
// ---------------------------------------------------------------------------
const CHECKS = [
  {
    id: 'share', kind: 'check',
    q: '確認：同じ型紙（カーネル）を画像じゅうで使い回すことの、いちばん大きな利点はどれでしょう？',
    options: [
      { t: '形がどこにあっても見つけられて、覚える重みも少なくて済む' },
      { t: '計算が速くなる', why: '畳み込みは場所の数だけ計算するので、計算量はむしろ増えます。得なのは重みの数と、位置が変わっても効くことです。' },
      { t: '色を扱えるようになる', why: '色は入力のチャンネルを増やせば扱えます。使い回しとは別の話です。' },
    ],
    answer: 0,
    explain: 'これをパラメータ共有と呼びます。全画素に別々の重みを持たせると、左上で覚えたことを右下では使えません。型紙を使い回せば、1つ覚えれば画像じゅうで効きます。',
  },
  {
    id: 'pool', kind: 'check',
    q: '確認：プーリング（2×2 の最大値だけ残す）は、何のためにあるでしょう？',
    options: [
      { t: 'すこしのズレを気にしないようにし、ついでに小さくするため' },
      { t: '画像を鮮明にするため', why: '情報は減ります。細かい位置はわざと捨てています。' },
      { t: '色を正規化するため', why: 'プーリングは位置の話で、値の正規化とは別です。' },
    ],
    answer: 0,
    explain: '「この 2×2 のどこかに強い反応があった」だけを残し、正確な位置は捨てます。1画素ずれても答えが変わらなくなり、次の段の計算も軽くなります。',
  },
  {
    id: 'same', kind: 'check',
    q: '確認：このページの CNN と、GAN や Transformer で共通しているのはどこでしょう？',
    options: [
      { t: '損失を決めて、勾配でパラメータを更新するところ' },
      { t: '次の1つを予測するところ', why: 'それは自己回帰モデル（Transformer）の話です。CNN は画像を一度に見て分類します。' },
      { t: '2つのネットワークが競い合うところ', why: 'それは GAN だけの特徴です。' },
    ],
    answer: 0,
    explain: '入力を数に直し、計算し、出力し、損失で測り、勾配で更新する —— 骨格は同じです。CNN が変えたのは「計算」の部分で、全結合の代わりに型紙の使い回しを持ち込みました。',
  },
];
$('checks').innerHTML = CHECKS.map((c) => `<div id="chk-${c.id}"></div>`).join('');
for (const c of CHECKS) renderQuiz($(`chk-${c.id}`), { ...c, id: `cnn:${c.id}` });

// ---------------------------------------------------------------------------
// 自分の言葉で
// ---------------------------------------------------------------------------
renderExplain($('explain'), {
  id: 'cnn',
  q: 'CNN が画像から形を見つけるしくみを、説明してみてください',
  lead: '「畳み込み」という言葉を使わずに説明できると、本当に分かっています。使ってもかまいませんが、そのときは必ず言い換えを添えてください。',
  words: [
    { t: 'カーネル（型紙）', any: ['カーネル', '型紙', 'フィルタ', 'フィルター'] },
    { t: 'ずらしながら当てる', any: ['ずらし', 'すべら', '滑ら', '動かしながら', '当てて'] },
    { t: '特徴マップ', any: ['特徴マップ', '反応', 'どこが光'] },
    { t: '使い回す（パラメータ共有）', any: ['使い回', '共有', '同じ型紙', '同じフィルタ'] },
    { t: 'プーリング', any: ['プーリング', '最大値', '2×2', 'ずれ'] },
  ],
  model: [
    '<b>3×3 の小さな型紙</b>を用意して、画像の左上から順に少しずつずらしながら重ね、「どれくらい合っているか」を計算します。その結果を並べたものが<b>特徴マップ</b>です。たての線に反応する型紙なら、たての線があるところだけが明るくなります。',
    '大事なのは、<b>同じ型紙を画像じゅうで使い回す</b>ことです。左上で覚えた「たての線の見つけ方」が、そのまま右下でも使えます。全部の画素に別々の重みを持たせるより、覚えることがずっと少なくて済みます。',
    '次に<b>プーリング</b>で、2×2 ごとに一番強い反応だけを残します。細かい位置は捨てるので、1〜2画素ずれても答えが変わりません。ついでに画像が小さくなり、次の計算が軽くなります。',
    '最後に、残った反応の並びを見て「たて棒か、よこ棒か…」を決めます。実際の CNN は、この「型紙 → 反応 → まとめる」を何段も積み、下の段では線や角、上の段では目や車輪のような大きな部品を見るようになります。',
  ],
  checks: [
    '型紙をずらしながら当てる、という動きを書いた',
    '同じ型紙を使い回すこと（位置が変わっても効く）に触れた',
    'プーリングが「ズレに強くする」ためだと書いた',
    '特徴マップが「どこが反応したか」の地図だと分かる書き方になっている',
    '実際の CNN が何段も積み重なっていることに触れた',
  ],
});

// ---------------------------------------------------------------------------
// コードで見る
// ---------------------------------------------------------------------------
{
  const box = document.createElement('section');
  box.className = 'panel';
  $('explain').closest('.panel').insertAdjacentElement('beforebegin', box);
  codeBox(box, {
    items: [
      {
        label: 'このページのネットワークを PyTorch で書くと',
        file: 'public/js/cnn.js',
        py: `import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=3, padding=1),  # 型紙 6 枚
    nn.ReLU(),
    nn.MaxPool2d(2),                            # 16×16 → 8×8
    nn.Flatten(),
    nn.Linear(6 * 8 * 8, 4),                    # 4 クラス
)
loss = F.cross_entropy(model(x), y)`,
        js: `// 型紙を 1 枚ずつ、画像の全部の場所に当てる
for (let f = 0; f < F; f++)
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      let s = this.b[f];
      for (let dy = 0; dy < K; dy++)
        for (let dx = 0; dx < K; dx++)
          s += this.w[(f * K + dy) * K + dx] * img[(y + dy - 1) * S + (x + dx - 1)];
      relu[o] = s > 0 ? s : 0;
    }`,
        note: 'PyTorch の Conv2d が内側でやっているのは、この二重ループです。パラメータは型紙 6×3×3 と全結合ぶんで 1,600 個だけ。',
      },
    ],
  });
}

// ---------------------------------------------------------------------------
renderSamples();
showOne(makeImage('v', mulberry32(11)));
refreshStats();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  renderSamples();
  refreshStats();
});
enableGlossary(['.hero', '.note', '.ex-model', '.lab-body']);

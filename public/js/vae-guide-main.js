// 「模写職人」ガイド：VAE をたとえ話で1ステップずつ見せる
import { VAE, VAE_DEFAULTS } from './vae.js';
import { mulberry32, gaussian } from './nn.js';
import { VAESpaceView, drawLatent } from './vae-viz.js';
import { mountHeader } from './common.js';

const $ = (id) => document.getElementById(id);

mountHeader({
  id: 'vae', mode: 'guide',
  sub: '「本物の絵を覚えて、メモだけを見て描き直す模写職人」として、VAE のしくみを順番に見ていく',
});

const DATASET = 'ring8';
const N = 400;
const N_ELLIPSE = 120;

const CONFIGS = {
  normal: { dataset: DATASET, seed: 1, beta: 1 },
  free: { dataset: DATASET, seed: 1, beta: 0 },        // メモの書式がバラバラ
  collapse: { dataset: DATASET, seed: 1, beta: 150 },  // 事後崩壊
};

const C = {
  real: '<span class="chip real">本物の絵</span>',
  drawn: '<span class="chip vae">描いた絵</span>',
  memo: '<span class="chip police">メモ</span>',
};

const state = {
  step: 0,
  config: null,
  running: false,
  queue: null,
  frame: 0,
  show: { density: false, real: true, samples: false, recon: false, grid: false },
};

let vae, realView, genZ;
const space = new VAESpaceView($('main'));

function resetVae(name) {
  stopAll();
  state.config = name;
  vae = new VAE({ ...VAE_DEFAULTS, ...CONFIGS[name] });
  const r = mulberry32(53 + 7);
  realView = vae.sampleReal(N);
  genZ = new Float64Array(N * vae.cfg.latentDim);
  for (let i = 0; i < genZ.length; i++) genZ[i] = gaussian(r);
}

function stopAll() {
  state.running = false;
  state.queue = null;
}

const STEPS = [
  {
    title: '模写職人の仕事',
    config: 'normal', fresh: true,
    show: { real: true },
    body: `<p>青い点は ${C.real} です。点の位置は絵の特徴（たとえば横が「明るさ」、縦が「線の太さ」）を表していて、8 種類の絵があります。</p>
      <p>ここに登場するのは <b>模写職人</b> です。職人は本物の絵を<b>見ることができます</b>（ここが GAN の偽札職人と違うところ）。</p>
      <p>ただし条件があります。職人は絵を持ち帰れず、<b>小さなメモ</b>だけを頼りに、あとで描き直さなければいけません。</p>`,
    term: `<span class="k">専門用語では</span>青い点 ＝ <b>学習データ</b>。メモ ＝ <b>潜在変数 z</b>。メモを作る人 ＝ <b>エンコーダ</b>、描き直す人 ＝ <b>デコーダ</b>。`,
  },
  {
    title: 'メモを取る',
    config: 'normal', fresh: true,
    show: { real: true },
    latent: true,
    body: `<p>右のメモ帳が、職人のメモの置き場所です。絵を1枚見るたびに、メモ帳のどこかに印をつけます。</p>
      <p>ここで大事なのは、印が<b>点ではなく楕円</b>だということです。「だいたいこのあたり、これくらいのぼんやりした範囲」という書き方をします。</p>
      <p>まだ何も学んでいないので、メモの置き方はでたらめです。</p>`,
    term: `<span class="k">専門用語では</span>楕円の中心 ＝ <b>μ</b>、楕円の大きさ ＝ <b>σ</b>。エンコーダは1点につきこの2つを出す。`,
  },
  {
    title: 'メモから描き直す',
    config: 'normal', needSteps: 0,
    show: { real: true, recon: true },
    latent: true,
    body: `<p>紫の線は「元の絵」と「メモから描き直した絵」を結んだものです。線が長いほど、うまく描き直せていません。</p>
      <p>ボタンを押して練習させると、線がみるみる短くなります。これが<b>再構成</b>の学習です。</p>`,
    actions: [{ label: '練習する（100回）', total: 100, perFrame: 5 }],
    term: `<span class="k">専門用語では</span>線の長さ ＝ <b>再構成誤差</b>。これだけを小さくする仕組みは<b>オートエンコーダ</b>と呼ばれる。`,
  },
  {
    title: 'メモの書式をそろえる',
    config: 'normal',
    show: { real: true, recon: true },
    latent: true, play: true,
    body: `<p>ところで、なぜ職人はメモを<b>ぼんやり</b>書き、しかも<b>決まった場所</b>に置くのでしょうか。</p>
      <p>それは、あとで「メモ帳の適当な場所」を指さして「ここのメモで1枚描いて」と言えるようにするためです。メモがバラバラの場所に散らばっていると、指さした先が空白で、何も描けません。</p>
      <p>そこで、すべての楕円を破線の円（標準的な書式）の中にそろえる力が働きます。自動で学習させて、メモ帳が整っていく様子を見てください。</p>`,
    term: `<span class="k">専門用語では</span>この「そろえる力」＝ <b>KL 項</b>。事前分布 N(0, I) に近づける。強さを決めるのが <b>β</b>。`,
  },
  {
    title: '新しい絵を描く',
    config: 'normal',
    show: { real: true, samples: true, density: true },
    latent: true, play: true,
    body: `<p>メモ帳が整ったので、今度は<b>本物を見ずに</b>新しい絵を描いてもらいます。メモ帳の適当な場所（標準正規分布からのくじ引き）を指さして、そこから描かせます。</p>
      <p>中抜きの丸が、そうして描かれた ${C.drawn} です。背景の濃淡は、職人が描きうる絵の広がりを表しています。</p>
      <p>本物の 8 か所をだいたい覆えていますが、少しぼやけて、山と山のあいだにも絵がこぼれます。これが「VAE はぼやける」と言われる現象です。</p>`,
    term: `<span class="k">専門用語では</span>メモ帳からのくじ引き ＝ <b>事前分布からのサンプリング</b>。ぼやけの正体は、再構成誤差の測り方（平均を取ると中間の絵が得をする）。`,
  },
  {
    title: '失敗その1：メモが自由すぎる',
    config: 'free', fresh: true,
    show: { real: true, samples: true, density: true },
    latent: true, play: true,
    body: `<p>「そろえる力」をゼロ（β = 0）にしてやり直します。職人は好きなようにメモを取れます。</p>
      <p>復元はとても上手になります。しかしメモ帳を見てください。印があちこちに散らばり、破線の円からはみ出しています。</p>
      <p>その状態でメモ帳の適当な場所を指さすと、そこは「何も書かれていない場所」なので、変な絵が出てきます。<b>復元はできるが、新しく作れない</b>という状態です。</p>`,
    term: `<span class="k">専門用語では</span>これはただの<b>オートエンコーダ</b>。生成モデルとしては使えない。`,
  },
  {
    title: '失敗その2：メモが空っぽ',
    config: 'collapse', fresh: true,
    show: { real: true, samples: true, density: true },
    latent: true, play: true,
    body: `<p>逆に「そろえる力」を極端に強く（β = 150）してやり直します。</p>
      <p>職人にとっては、何も書かずに<b>全部同じメモ</b>にしてしまうのが一番ラクです。メモ帳では、すべての楕円が円の中央に重なっていきます。</p>
      <p>こうなると、どこを指さしても同じような絵しか出てきません。下の「メモの情報量」がほぼ 0 になります。</p>`,
    term: `<span class="k">専門用語では</span><b>事後崩壊（posterior collapse）</b>。デコーダが z を無視してしまう状態。`,
  },
  {
    title: 'まとめ',
    config: 'normal', fresh: true,
    show: { real: true, samples: true, density: true },
    latent: true, play: true, autoplay: true,
    body: `<p>ちょうどよい β では、「復元できる」と「メモ帳が整っている」の両方が成り立ち、新しい絵を作れるようになります。</p>
      <p><b>GAN との違い：</b>GAN の職人は本物を見られず、警察の反応だけを頼りにしました。VAE の職人は本物を見られる代わりに、メモという細い管を通さなければいけません。だから GAN はくっきり、VAE はぼやける、という差が出ます。</p>
      <div class="links"><a href="vae.html">詳細タブで自由に動かす →</a><a href="gan-guide.html">GAN のガイドと見比べる →</a></div>`,
    term: `<table>
      <tr><td>模写職人（メモを取る人）</td><td>エンコーダ q(z|x)</td></tr>
      <tr><td>模写職人（描き直す人）</td><td>デコーダ p(x|z)</td></tr>
      <tr><td>メモ</td><td>潜在変数 z</td></tr>
      <tr><td>メモのぼんやりさ</td><td>σ（不確かさ）</td></tr>
      <tr><td>描き直しのズレ</td><td>再構成誤差</td></tr>
      <tr><td>メモの書式をそろえる力</td><td>KL 項（強さ = β）</td></tr>
      <tr><td>メモが自由すぎる</td><td>ただのオートエンコーダ</td></tr>
      <tr><td>メモが空っぽ</td><td>事後崩壊</td></tr>
    </table>`,
  },
];

function enterStep(i) {
  stopAll();
  state.step = i;
  const s = STEPS[i];
  if (s.fresh && (state.config !== s.config || vae.step > 0)) resetVae(s.config);
  else if (state.config !== s.config) resetVae(s.config);
  state.show = { density: false, real: false, samples: false, recon: false, grid: false, ...s.show };
  state.latent = !!s.latent;
  $('stepNo').textContent = `${i + 1} / ${STEPS.length}`;
  $('stepTitle').textContent = s.title;
  $('stepBody').innerHTML = s.body;
  $('stepTerm').innerHTML = s.term || '';
  $('prev').disabled = i === 0;
  $('next').disabled = i === STEPS.length - 1;
  [...$('dots').children].forEach((d, k) => d.classList.toggle('on', k === i));
  if (s.autoplay) state.running = true;
  renderStep();
  render(true);
}

function renderStep() {
  const s = STEPS[state.step];
  const box = $('stepActions');
  box.innerHTML = '';
  const busy = !!state.queue;
  for (const a of s.actions || []) {
    const b = document.createElement('button');
    b.className = 'act';
    b.textContent = busy ? `学習中…（残り ${state.queue.remaining}）` : a.label;
    b.disabled = busy;
    b.addEventListener('click', () => { state.queue = { remaining: a.total, perFrame: a.perFrame }; renderStep(); });
    box.appendChild(b);
  }
  if (s.play) {
    const b = document.createElement('button');
    b.className = 'act';
    b.textContent = state.running ? '❚❚ 止める' : '▶ 自動で学習';
    b.addEventListener('click', togglePlay);
    box.appendChild(b);
    const r = document.createElement('button');
    r.textContent = '最初から';
    r.addEventListener('click', () => { resetVae(s.config); renderStep(); render(true); });
    box.appendChild(r);
  }
}

function togglePlay() {
  state.queue = null;
  state.running = !state.running;
  renderStep();
}

function render(force) {
  const L = vae.cfg.latentDim;
  const samples = state.show.samples ? vae.decode(genZ, N) : null;
  const { mu, sd } = vae.encode(realView, N);
  const recon = state.show.recon ? vae.decode(mu.slice(0, N_ELLIPSE * L), N_ELLIPSE) : null;
  if (state.show.density && (force || state.frame % 8 === 0)) {
    space.updateDensity(vae.density(space.points, space.res * space.res, 120));
  }
  space.draw({ show: state.show, real: realView, samples, recon });
  drawLatent($('latent'), {
    latentDim: L,
    mu: state.latent ? mu.slice(0, N_ELLIPSE * L) : null,
    sd: state.latent ? sd.slice(0, N_ELLIPSE * L) : null,
  });
  $('latentNote').textContent = state.latent
    ? '紫の楕円が1枚の絵のメモ。破線の円が「標準的な書式」の目安。'
    : 'まだメモを取っていません。';

  $('mStep').textContent = vae.step.toLocaleString();
  if (vae.last) {
    const err = Math.sqrt(vae.last.mse);
    $('mErr').textContent = err.toFixed(3);
    $('mErrBar').style.width = `${Math.min(100, (err / 1.2) * 100)}%`;
    $('mKl').textContent = vae.last.kl.toFixed(2);
    $('mKlBar').style.width = `${Math.min(100, (vae.last.kl / 6) * 100)}%`;
  } else {
    $('mErr').textContent = $('mKl').textContent = '–';
    $('mErrBar').style.width = $('mKlBar').style.width = '0%';
  }
}

function loop() {
  state.frame++;
  let changed = false;
  if (state.queue) {
    const q = state.queue;
    const n = Math.min(q.perFrame, q.remaining);
    for (let i = 0; i < n; i++) vae.trainStep();
    q.remaining -= n;
    changed = true;
    if (q.remaining <= 0) {
      state.queue = null;
      renderStep();
      const done = document.createElement('span');
      done.className = 'done';
      done.textContent = '完了。もう一度押すと続けられます';
      $('stepActions').appendChild(done);
    } else if (state.frame % 5 === 0) renderStep();
  } else if (state.running) {
    const t0 = performance.now();
    for (let i = 0; i < 12 && performance.now() - t0 < 18; i++) vae.trainStep();
    changed = true;
  }
  if (changed) render(false);
  requestAnimationFrame(loop);
}

function init() {
  const dots = $('dots');
  STEPS.forEach((s, i) => {
    const b = document.createElement('button');
    b.title = `${i + 1}. ${s.title}`;
    b.setAttribute('aria-label', `${i + 1}. ${s.title}`);
    b.addEventListener('click', () => enterStep(i));
    dots.appendChild(b);
  });
  $('prev').addEventListener('click', () => state.step > 0 && enterStep(state.step - 1));
  $('next').addEventListener('click', () => state.step < STEPS.length - 1 && enterStep(state.step + 1));
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') $('next').click();
    else if (e.key === 'ArrowLeft') $('prev').click();
    else if (e.code === 'Space' && STEPS[state.step].play) { e.preventDefault(); togglePlay(); }
  });
  window.addEventListener('resize', () => render(true));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(true));

  resetVae('normal');
  enterStep(0);
  requestAnimationFrame(loop);
}

init();

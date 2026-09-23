// 「はじめて（偽札と警察）」ガイド：GAN をたとえ話で1ステップずつ見せる
import { GAN, DEFAULTS } from './gan.js';
import { DATASETS } from './datasets.js';
import { mulberry32, gaussian } from './nn.js';
import { SpaceView } from './viz.js';
import { modesCovered } from './metrics.js';
import { renderQuiz } from './quiz.js';
import { renderExplain } from './explain.js';

import { mountHeader, mountGoal, enableGlossary } from './common.js';

const $ = (id) => document.getElementById(id);

mountHeader({ id: 'gan', mode: 'guide', sub: '「偽札職人（Generator）」と「警察（Discriminator）」の勝負として、GAN のしくみを順に見ていきます' });
mountGoal('gan-guide');
const N = 500;
const DATASET = 'ring8';

// たとえ話で使う設定
const CONFIGS = {
  normal: { dataset: DATASET, seed: 1 },
  // せっかちな職人（学習率が大きい）と、のんびりした警察（学習率が小さい）
  collapse: { dataset: DATASET, seed: 1, lrG: 0.01, lrD: 0.0005 },
  // 厳しすぎる警察（何度も勉強する・学習率も大きい）と、昔のやり方（ミニマックス損失）
  vanish: { dataset: DATASET, seed: 1, loss: 'minimax', dSteps: 5, lrD: 0.005 },
};

const C = {
  real: '<span class="chip real">本物のお札</span>',
  fake: '<span class="chip fake">偽札</span>',
  maker: '<span class="chip fake">偽札職人</span>',
  police: '<span class="chip police">警察</span>',
};

const state = {
  step: 0,
  config: null,
  running: false,
  queue: null,
  dCount: 0,
  gCount: 0,
  frame: 0,
  show: { heat: false, real: true, fake: false, grad: false, grid: false, colorByZ: false },
};

let gan, viewZ, realView;
const space = new SpaceView($('main'));

function resetGan(name) {
  stopAll();
  state.config = name;
  gan = new GAN({ ...DEFAULTS, ...CONFIGS[name] });
  const r = mulberry32(31 + 5); // GAN タブと同じ乱数で、同じ 500 枚の偽札を追う
  viewZ = new Float64Array(N * gan.cfg.noiseDim);
  for (let i = 0; i < viewZ.length; i++) viewZ[i] = gaussian(r);
  realView = gan.sampleReal(N);
  state.dCount = 0;
  state.gCount = 0;
}

function ensureConfig(name) {
  if (state.config !== name) resetGan(name);
}

function stopAll() {
  state.running = false;
  state.queue = null;
}

// ---- 学習の小分け実行（アニメーションとして見せる） ----
const trainOnce = {
  D: () => { gan.trainD(); state.dCount++; },
  G: () => { gan.trainG(); state.gCount++; },
  both: () => { gan.trainStep(); state.dCount += gan.cfg.dSteps; state.gCount++; },
};

function enqueue(kind, total, perFrame, onDone) {
  state.running = false;
  state.queue = { kind, remaining: total, perFrame, onDone };
  renderStep();
}

// ---- ストーリー ----
const STEPS = [
  {
    title: '偽札職人と警察の勝負',
    config: 'normal', fresh: true,
    show: { real: true },
    body: `<p>この画面では、2つの AI が勝負します。お札を偽造する ${C.maker} と、それを見破る ${C.police} です。</p>
      <p>青い点は ${C.real} です。点の位置はお札の特徴（たとえば横が「色味」、縦が「大きさ」）を表しています。本物のお札には <b>8 種類</b>あるので、8 つのかたまりになっています。</p>
      <p>大事なルールがひとつあります。<b>偽札職人は本物のお札を一度も見たことがありません。</b></p>`,
    term: `<span class="k">専門用語では</span>青い点 ＝ <b>学習データ</b>（本物のデータの分布）。8 つのかたまり ＝ 8 つの<b>モード</b>。`,
  },
  {
    title: '職人の最初の偽札',
    config: 'normal', fresh: true,
    show: { real: true, fake: true },
    body: `<p>${C.maker} が、見よう見まねで ${C.fake} を 500 枚作りました（橙の点）。</p>
      <p>職人は毎回サイコロを振って、その目をもとに偽札を作ります。サイコロの目が違えば、違う偽札ができます。</p>
      <p>まだ何も学んでいないので、本物とはまったく違う場所にあります。</p>`,
    term: `<span class="k">専門用語では</span>偽札職人 ＝ <b>Generator（生成器、G）</b>。サイコロ ＝ <b>ノイズ z</b>（ランダムな数）。`,
  },
  {
    title: '警察が勉強する',
    config: 'normal',
    show: { heat: true, real: true, fake: true },
    body: `<p>${C.police} は本物のお札と偽札を見比べて、「このあたりのお札は本物っぽい（青）」「このあたりは怪しい（橙）」という<b>地図</b>を作ります。背景の色がその地図です。</p>
      <p>ボタンを押すと、警察だけが勉強します。職人はまだ何もしません。背景がどう変わるか見てください。</p>
      <p>地図にマウスを乗せると、その場所のお札を警察がどれくらい本物っぽいと思っているかが出ます。</p>`,
    actions: [{ label: '警察に勉強させる（20回）', kind: 'D', total: 20, perFrame: 1 }],
    term: `<span class="k">専門用語では</span>警察 ＝ <b>Discriminator（識別器、D）</b>。地図の色 ＝ <b>D(x)</b>（そのお札が本物である確率）。`,
  },
  {
    title: '職人は警察の反応から学ぶ',
    quiz: {
      q: '職人だけを修行させると、偽札（橙の点）はどちらへ動くでしょう？',
      options: [
        { t: '本物の点に向かってまっすぐ動く', why: '職人は本物を一度も見ていません。動く先は「警察が本物っぽいと思っている場所」であって、本物の点そのものではありません。両者はたいてい少しずれています。' },
        { t: '警察が「本物っぽい」と判定している場所（青い側）へ動く' },
        { t: '警察が「偽物」と判定している場所から、ただ遠ざかる', why: '遠ざかるだけでは方向が定まりません。勾配は「D(x) が上がる向き」を指しているので、逃げるのではなく青い方へ登っていきます。' },
      ],
      answer: 1,
      explain: '職人の目標は「警察に本物っぽいと思われる」ことだけです。矢印は D の出力が上がる方向、つまり青い方を指しています。',
    },
    config: 'normal', needD: 20,
    show: { heat: true, real: true, fake: true, grad: true },
    body: `<p>職人は本物を見られませんが、<b>警察の地図は見られます</b>。</p>
      <p>矢印は「偽札をこっちに直せば、警察にもっと本物っぽいと思われる」という方向です。スイカ割りで「右！もっと前！」と声をかけてもらうようなものです。</p>
      <p>ボタンを押すと、職人だけが修行します。押す前に、下の予想に答えてみてください。</p>`,
    reveal: '<p>矢印の方向、つまり青い場所へ偽札が動いていきます。職人は本物を見ないまま、警察の地図を登っているだけです。</p>',
    actions: [{ label: '職人に修行させる（10回）', kind: 'G', total: 10, perFrame: 1 }],
    term: `<span class="k">専門用語では</span>矢印 ＝ <b>勾配（こうばい）</b>。修行 ＝ <b>パラメータの更新</b>。職人は警察を通してしか本物を知らない、というのが GAN のいちばん不思議なところ。`,
  },
  {
    title: 'いたちごっこ',
    quiz: {
      q: '交互に修行させると、警察の地図（背景）はどうなるでしょう？',
      options: [
        { t: '一度できた地図はほぼ固定で、偽札だけが動いていく', why: '警察も毎回学習し直しています。相手が動くので、正解の地図も動き続けます。ここが「止まっている的を撃つ」普通の学習と決定的に違うところです。' },
        { t: '偽札を追いかけるように塗り替わり続ける' },
        { t: '偽札が本物に近づくほど、画面全体が青（本物）に寄っていく', why: '警察は最後まで本物と偽物を分けようとします。均衡で起きるのは「全体が青くなる」ではなく「色が薄くなる」（D ≈ 0.5）です。' },
      ],
      answer: 1,
      explain: '職人が新しい場所へ移ると、警察はそこを怪しいと塗り直します。互いに相手を追いかけ続けるのが敵対的学習です。',
    },
    config: 'normal', needD: 20,
    show: { heat: true, real: true, fake: true },
    body: `<p>職人がうまくなると、警察は新しい偽札を見破るために勉強し直します。すると職人もまた直します。</p>
      <p>これを <b>交互に</b> くり返すのが GAN の学習です。ボタンを押す前に、背景の地図がどうなるか予想してみてください。</p>`,
    reveal: '<p>警察の地図が、偽札を追いかけるように塗り替わり続けます。相手が動くので、目標そのものが動き続けます。</p>',
    actions: [{ label: '交互に修行（50回）', kind: 'both', total: 50, perFrame: 2 }],
    term: `<span class="k">専門用語では</span>交互に競わせて学ぶこと ＝ <b>敵対的学習</b>（GAN の「A」＝ Adversarial）。`,
  },
  {
    title: '決着はつくのか',
    config: 'normal', needD: 20,
    show: { heat: true, real: true, fake: true },
    play: true,
    body: `<p>自動で修行を続けてみましょう。</p>
      <p>ゴールは、<b>警察が本物と偽札を見分けられなくなる</b>ことです。そうなると警察の正解率はコイン投げと同じ <b>50%</b> に近づき、背景の色も薄くなります。</p>
      <p>下の「作れているお札の種類」が 8 つそろうかにも注目してください。本物は 8 種類あるので、全部作れてはじめて「本物そっくり」です。</p>`,
    term: `<span class="k">専門用語では</span>見分けがつかない状態 ＝ <b>均衡</b>（D(x) = 0.5）。実際にはぴったり止まらず、ゆらゆら揺れながら近づくのがふつう。`,
  },
  {
    title: '失敗その1：あせる職人',
    quiz: {
      q: '職人だけがせっかち（学習率が大きい）だと、何が起きるでしょう？',
      options: [
        { t: '上達が速くなり、8つの山を早く全部作れるようになる', why: '速く動くことと、まんべんなく作ることは別です。1回で大きく動くほど、その瞬間いちばん得な1か所へ集中しやすくなります。' },
        { t: '偽札が一部の山にかたまり、山から山へ飛び移る' },
        { t: '偽札が画面全体に均等にばらまかれる', why: 'それは学習がまだ進んでいない状態の見え方です。モード崩壊は逆で、狭い場所に集まってしまう失敗です。' },
      ],
      answer: 1,
      explain: 'これがモード崩壊です。うまく騙せた1か所に集中し、警察が気づくと、かたまりごと別の場所へ移動します。',
    },
    config: 'collapse', fresh: true,
    show: { heat: true, real: true, fake: true },
    play: true,
    body: `<p>最初からやり直します。今度は職人が<b>せっかち</b>（1回で大きく直しすぎる）で、警察は<b>のんびり</b>（なかなか学ばない）という組み合わせです。</p>
      <p>再生する前に、何が起きるか予想してみてください。下の「作れているお札の種類」に注目です。</p>`,
    reveal: '<p>職人は、うまく騙せた場所ばかり作るようになります。警察が気づくと、偽札が<b>ひとかたまりのまま</b>別の場所へ飛び移ります。一発屋の芸人が、持ちネタを1つずつ乗り換えていくようなものです。</p>',
    term: `<span class="k">専門用語では</span>一部の種類しか作れなくなる現象 ＝ <b>モード崩壊（mode collapse）</b>。1回に直す量 ＝ <b>学習率</b>。`,
  },
  {
    title: '失敗その2：厳しすぎる警察',
    quiz: {
      q: '警察が強すぎて偽札を 100% 見破るようになると、職人はどうなるでしょう？',
      options: [
        { t: '手がかりを失って迷走する' },
        { t: '差が大きいぶん強い信号が返るので、かえって速く上達する', why: '直感に反しますが逆です。D(G(z)) がほぼ 0 の領域では、元のミニマックス損失の勾配もほぼ 0 になります。「差が大きい」ことと「勾配が大きい」ことは別です。' },
        { t: 'G の損失が 0 になり、そこで学習が止まる', why: 'G の損失はむしろ大きいままです。値が大きいことと、直すべき向きが分かることは別問題で、消えているのは後者です。' },
      ],
      answer: 0,
      explain: '「全部ダメ」しか返ってこないと、どちらへ直せばよいかの情報（勾配）が消えます。これが勾配消失です。',
    },
    config: 'vanish', fresh: true,
    show: { heat: true, real: true, fake: true },
    play: true,
    body: `<p>もう一度やり直します。今度は警察が<b>とても優秀で厳しい</b>うえに、職人は昔ながらの学び方をします。</p>
      <p>警察はすぐに偽札を 100% 見破るようになります。そのとき職人がどうなるか、予想してから再生してみてください。</p>`,
    reveal: '<p>警察が「全部ダメ」としか言わなくなるので、職人は<b>どちらへ直せばいいかの手がかりを失い</b>、見当違いの方向へ進んで画面の外へ出ていきます。「全部ダメ」しか言わない先生からは何も学べない、というのと同じです。</p>',
    term: `<span class="k">専門用語では</span>手がかり（勾配）がほとんど消えてしまう現象 ＝ <b>勾配消失</b>。今は、職人が「どれくらい本物っぽいと思われたか」を直接上げにいく<b>非飽和損失</b>という学び方が標準で、この失敗を起こしにくい。`,
  },
  {
    title: 'まとめ',
    quiz: {
      kind: 'check',
      q: '確認：GAN の Generator は、本物のデータをどう使っているでしょう？',
      options: [
        { t: '学習の最初だけ本物を見て、あとは Discriminator の反応に従う', why: '最初も見ません。実装上も G の入力はノイズ z だけで、本物のデータは G の計算に一度も登場しません。' },
        { t: '本物は一度も見ず、Discriminator の判定だけを手がかりにしている' },
        { t: 'バッチの半分が本物、半分が生成で、両方を見比べながら学ぶ', why: 'その「見比べ」をしているのは Discriminator です。G の更新に使うのは、自分が作った偽物に対する D の反応だけです。' },
      ],
      answer: 1,
      explain: '本物を見るのは D だけ。G はその D の反応を通じて間接的に学びます。この非対称さが GAN の本質です。',
    },
    config: 'normal', fresh: true,
    show: { heat: true, real: true, fake: true },
    play: true, autoplay: true,
    body: `<p>最後に、うまくいく設定でもう一度動かしています。たとえ話と専門用語の対応を振り返っておきましょう。</p>
      <p><b>たとえ話の限界：</b>実際の警察（D）は本物のお札も見て学んでいます。本物を知らないのは職人（G）だけです。この「知っている側」と「知らない側」の非対称さが、GAN の本質です。</p>
      <div class="links"><a href="index.html?dataset=ring8">GAN タブで詳しく見る →</a><a href="ar.html?dataset=ring8">別の作り方（自己回帰）を見る →</a></div>`,
    term: `<table>
      <tr><td>偽札職人</td><td>Generator（G・生成器）</td></tr>
      <tr><td>警察</td><td>Discriminator（D・識別器）</td></tr>
      <tr><td>サイコロの目</td><td>ノイズ z（潜在変数）</td></tr>
      <tr><td>警察の地図</td><td>D(x)：本物である確率</td></tr>
      <tr><td>スイカ割りの声</td><td>勾配</td></tr>
      <tr><td>修行 1 回</td><td>パラメータの更新（1 ステップ）</td></tr>
      <tr><td>せっかち・のんびり</td><td>学習率の大小</td></tr>
      <tr><td>一発屋</td><td>モード崩壊</td></tr>
      <tr><td>「全部ダメ」の先生</td><td>勾配消失</td></tr>
    </table>`,
  },
];

function enterStep(i) {
  stopAll();
  state.step = i;
  const s = STEPS[i];
  if (s.fresh && (state.config !== s.config || state.dCount + state.gCount > 0)) resetGan(s.config);
  else ensureConfig(s.config);
  // 飛ばして来たときの帳尻合わせ（警察がまだ何も学んでいなければ、先に勉強させておく）
  if (s.needD && state.dCount < s.needD) {
    while (state.dCount < s.needD) trainOnce.D();
  }
  state.show = { heat: false, real: false, fake: false, grad: false, grid: false, colorByZ: false, ...s.show };
  $('heatLegend').style.visibility = state.show.heat ? 'visible' : 'hidden';
  $('stepNo').textContent = `${i + 1} / ${STEPS.length}`;
  $('stepTitle').textContent = s.title;
  $('stepBody').innerHTML = s.body;
  $('stepTerm').innerHTML = s.term || '';
  const qz = $('stepQuiz');
  qz.innerHTML = '';
  qz.className = '';
  delete qz.dataset.answered;
  if (s.reveal) {
    // 予想クイズがある回は、答えにあたる説明を回答後まで伏せる
    const rv = document.createElement('div');
    rv.className = 'reveal';
    rv.innerHTML = s.reveal;
    rv.hidden = !!s.quiz;
    $('stepBody').appendChild(rv);
    qz.dataset.reveal = '1';
  }
  if (s.quiz) renderQuiz(qz, { id: `gan-guide:${i}`, ...s.quiz }, () => {
    const rv = $('stepBody').querySelector('.reveal');
    if (rv) rv.hidden = false;
  });
  $('prev').disabled = i === 0;
  $('next').disabled = i === STEPS.length - 1;
  [...$('dots').children].forEach((d, k) => d.classList.toggle('on', k === i));
  if (s.autoplay) state.running = true;
  renderStep();
  render(true);
}

/** 操作ボタン（ストーリーごと）の描画 */
function renderStep() {
  const s = STEPS[state.step];
  const box = $('stepActions');
  box.innerHTML = '';
  const busy = !!state.queue;
  for (const a of s.actions || []) {
    const b = document.createElement('button');
    b.className = 'act';
    b.textContent = busy && state.queue.kind === a.kind ? `実行中…（残り ${state.queue.remaining}）` : a.label;
    b.disabled = busy;
    b.addEventListener('click', () => enqueue(a.kind, a.total, a.perFrame));
    box.appendChild(b);
  }
  if (s.play) {
    const b = document.createElement('button');
    b.className = 'act';
    b.textContent = state.running ? '❚❚ 止める' : '▶ 自動で修行';
    b.addEventListener('click', togglePlay);
    box.appendChild(b);
    const r = document.createElement('button');
    r.textContent = '最初から';
    r.addEventListener('click', () => {
      resetGan(s.config);
      if (s.needD) while (state.dCount < s.needD) trainOnce.D();
      renderStep();
      render(true);
    });
    box.appendChild(r);
  }
}

function togglePlay() {
  state.queue = null;
  state.running = !state.running;
  renderStep();
}

// ---- 描画 ----
function render(heat) {
  const fake = gan.generate(viewZ, N);
  if (state.show.heat && heat) space.updateHeat(gan.discriminate(space.heatPoints, space.heatRes ** 2));
  // 矢印は 150 本だけ（多すぎると見えなくなる）
  const grad = state.show.grad ? gan.dGradient(fake.slice(0, 300), 150) : null;
  // SpaceView は fake の先頭から順に矢印を描くので、先頭 150 点の偽札に矢印が付く
  space.draw({ show: state.show, real: realView, fake, grad, fakeLabel: '偽札' });

  // メーター
  $('mStep').textContent = `職人 ${state.gCount} ・ 警察 ${state.dCount}`;
  const dr = gan.discriminate(realView, N);
  const df = gan.discriminate(fake, N);
  let correct = 0;
  for (let i = 0; i < N; i++) { if (dr[i] >= 0.5) correct++; if (df[i] < 0.5) correct++; }
  const acc = correct / (2 * N);
  const trained = state.dCount > 0;
  $('mAcc').textContent = trained ? `${Math.round(acc * 100)}%` : 'まだ勉強していない';
  $('mAccBar').style.width = trained ? `${acc * 100}%` : '0%';
  const cov = modesCovered(fake, N, DATASETS[DATASET].modes);
  const kinds = $('mKinds');
  if (!kinds.children.length) for (let k = 0; k < cov.total; k++) kinds.appendChild(document.createElement('i'));
  const shown = state.show.fake;
  [...kinds.children].forEach((el, k) => el.classList.toggle('on', shown && cov.flags[k]));
  $('mKindsText').textContent = shown ? `${cov.covered} / ${cov.total}` : '–';
}

function loop() {
  state.frame++;
  let changed = false;
  if (state.queue) {
    const q = state.queue;
    const n = Math.min(q.perFrame, q.remaining);
    for (let i = 0; i < n; i++) trainOnce[q.kind]();
    q.remaining -= n;
    changed = true;
    if (q.remaining <= 0) {
      state.queue = null;
      renderStep();
      const done = document.createElement('span');
      done.className = 'done';
      done.textContent = '完了。もう一度押すと続けられます';
      $('stepActions').appendChild(done);
    } else if (state.frame % 4 === 0) {
      renderStep();
    }
  } else if (state.running) {
    const t0 = performance.now();
    for (let i = 0; i < 6 && performance.now() - t0 < 20; i++) trainOnce.both();
    changed = true;
  }
  if (changed) render(state.frame % 2 === 0 || !!state.queue);
  requestAnimationFrame(loop);
}

// ---- 初期化 ----
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

  const tip = $('mainTip');
  const canvas = $('main');
  canvas.addEventListener('pointermove', (e) => {
    if (!state.show.heat) { tip.hidden = true; return; }
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const [x, y] = space.toWorld(px, py);
    const p = gan.discriminate(Float64Array.from([x, y]), 1)[0];
    space.hover = [px, py];
    tip.innerHTML = `<div class="k">警察の判定</div><div>本物っぽさ <b>${Math.round(p * 100)}%</b>　${p >= 0.5 ? '（本物だと思う）' : '（偽札だと思う）'}</div>`;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${px + 14 + tw > rect.width ? px - tw - 14 : px + 14}px`;
    tip.style.top = `${Math.max(4, py - 44)}px`;
    if (!state.running && !state.queue) render(false);
  });
  canvas.addEventListener('pointerleave', () => { tip.hidden = true; space.hover = null; render(false); });
  window.addEventListener('resize', () => render(true));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(true));

  resetGan('normal');
  enableGlossary();
  enterStep(0);
  requestAnimationFrame(loop);
}

init();

// ---- 自分の言葉で説明する（読む → 予想する の次は、出力する） ----
renderExplain(document.getElementById('explain'), {
  id: 'gan',
  q: 'GAN はどうやって上手くなるのか、AI を知らない友だちに説明してみてください',
  lead: '正解の文章を思い出そうとしなくて大丈夫です。たとえ話のままでもかまいません。書いてから、下の一例と見くらべます。',
  words: [
    { t: 'Generator（偽札職人）', any: ['Generator', 'ジェネレータ', '生成器', '偽札職人', '職人'] },
    { t: 'Discriminator（警察）', any: ['Discriminator', 'ディスクリミネータ', '識別器', '警察'] },
    { t: 'ノイズ z', any: ['ノイズ', 'z', 'サイコロ', '乱数'] },
    { t: '本物を見ない', any: ['本物を見', '本物は見', '見ないまま', '見たことがない', '見られない'] },
    { t: '勾配', any: ['勾配', '手がかり', 'どちらに動か', '方向'] },
    { t: '失敗の例', any: ['モード崩壊', '一発屋', '勾配消失', '全部ダメ', '失敗'] },
  ],
  model: [
    '2つの AI が別々の目的で戦います。<b>Generator</b>（偽札職人）は、毎回ちがう<b>ノイズ z</b> を受け取って、そこから1枚の偽札を作ります。<b>Discriminator</b>（警察）は、本物と偽物を見せられて「どちらか」を当てる練習をします。',
    'ここが肝心なところで、<b>Generator は本物を一度も見ません</b>。見えるのは「警察がどれくらい本物だと思ったか」だけ。その判定がどちらに動けば上がるか（＝<b>勾配</b>）を聞いて、その方向に少しだけ動きます。目隠しをして、まわりの「もっと右」「もっと左」の声だけでスイカに近づくのと同じです。',
    '警察が強くなると職人も強くならざるを得ず、職人が強くなると警察も学び直す。このいたちごっこで両方が上がっていきます。ただし片方だけが速いと壊れます。職人が急ぎすぎると<b>モード崩壊</b>（いつも同じ偽札しか作らない）、警察が強すぎると「全部ダメ」しか返ってこなくなって<b>勾配消失</b>（手がかりが消える）が起きます。',
  ],
  checks: [
    'Generator が本物のデータを一度も見ないことに触れた',
    'Generator の手がかりが「Discriminator の判定」だけだと書いた',
    '2つが別々の目的（損失）を持って動いていることが伝わる',
    '失敗の例（モード崩壊 か 勾配消失）を少なくとも1つ挙げた',
    'ノイズ z が「毎回ちがう絵を作るための種」だと分かる書き方になっている',
  ],
});

// 言語モデルの心臓部：次の1文字を確率で予測し、1つ選んで、また予測する。
import { CharLM } from './charlm.js';
import { mountHeader, mountGoal, enableGlossary } from './common.js';
import { renderQuiz } from './quiz.js';
import { renderExplain } from './explain.js';
import { codeBox } from './codebox.js';

mountHeader({ id: 'llm', mode: 'detail', sub: '文字単位の小さな言語モデルで、次の1つを予測する仕組みを動かします' });
mountGoal('llm');

const $ = (id) => document.getElementById(id);
const TARGET = 2000;            // 学習ステップ数
const PRESETS = ['よろしくお', 'ありがと', 'ごかくにん', 'おせわに', 'あしたまでに', ''];

const lm = new CharLM();
let job = null;                 // 学習中かどうか
let trained = false;

$('vocab').textContent = `${lm.V} 種類`;
$('params').textContent = lm.net.paramCount().toLocaleString();
$('stH').textContent = lm.entropy.toFixed(3);

// ---------------------------------------------------------------------------
// 学習
// ---------------------------------------------------------------------------
function tick() {
  if (!job) return;
  const t0 = performance.now();
  while (performance.now() - t0 < 24 && job.done < TARGET) {
    lm.trainStep();
    job.done++;
  }
  const bar = $('prog').querySelector('i');
  bar.style.width = `${Math.round((job.done / TARGET) * 100)}%`;
  $('prog').querySelector('span').textContent = `${job.done.toLocaleString()} / ${TARGET.toLocaleString()} ステップ`;
  $('stStep').textContent = lm.step.toLocaleString();
  $('stNll').textContent = lm.last ? lm.last.nll.toFixed(3) : '–';
  render();
  if (job.done >= TARGET) {
    stopTraining();
    trained = true;
    $('btnTrain').textContent = 'もう一度 2,000 ステップ学習させる';
    $('trainPanel').classList.add('done');
  } else {
    requestAnimationFrame(tick);
  }
}

function startTraining() {
  job = { done: 0 };
  $('prog').hidden = false;
  $('btnTrain').disabled = true;
  $('btnStop').hidden = false;
  requestAnimationFrame(tick);
}

function stopTraining() {
  job = null;
  $('prog').hidden = true;
  $('btnTrain').disabled = false;
  $('btnStop').hidden = true;
}

$('btnTrain').addEventListener('click', startTraining);
$('btnStop').addEventListener('click', stopTraining);

// ---------------------------------------------------------------------------
// 次の1文字
// ---------------------------------------------------------------------------
const show = (ch) => (ch === '\n' ? '⏎ 文の終わり' : ch === ' ' ? '␣' : ch);

function render() {
  const prompt = $('prompt').value;
  const temp = +$('temp').value;
  const raw = lm.nextProbs(prompt);
  const hot = lm.withTemperature(raw, temp);
  // 温度をかけたあとの確率が高い順に並べる
  const idx = [...hot.keys()].sort((a, b) => hot[b] - hot[a]).slice(0, 8);
  const max = Math.max(...idx.map((i) => Math.max(raw[i], hot[i])), 0.01);
  $('bars').innerHTML = idx.map((i) => `
    <div class="lm-row">
      <span class="lm-ch">${show(lm.chars[i])}</span>
      <span class="lm-bar">
        <i class="raw" style="width:${(raw[i] / max) * 100}%"></i>
        <i class="hot" style="width:${(hot[i] / max) * 100}%"></i>
      </span>
      <span class="lm-p">${(hot[i] * 100).toFixed(1)}%</span>
    </div>`).join('');

  $('tempOut').textContent = temp.toFixed(1);
  $('tempNote').textContent =
    temp <= 0.3 ? 'ほぼ一番高い文字しか選ばれません。何度生成しても同じ答えになります。'
    : temp <= 1.05 ? 'モデルが出した確率どおりに近い選び方です。毎回すこし違う答えになります。'
    : '低い確率の文字も選ばれます。意外さは増えますが、途中で崩れやすくなります。';
}

for (const id of ['prompt', 'temp']) $(id).addEventListener('input', render);

$('presets').innerHTML = PRESETS
  .map((p) => `<button data-p="${p}">${p === '' ? '（文の最初から）' : p}</button>`).join('');
$('presets').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  $('prompt').value = b.dataset.p;
  render();
});

// ---------------------------------------------------------------------------
// 生成
// ---------------------------------------------------------------------------
function renderTrace(trace, text) {
  $('out').hidden = false;
  $('outText').textContent = text.replace(/\n/g, '');
  $('trace').innerHTML = trace.map((t) => `
    <span class="tr" title="選ばれた確率 ${(t.p * 100).toFixed(1)}%">
      <b>${show(t.ch)}</b><i style="opacity:${0.25 + 0.75 * t.p}"></i>
    </span>`).join('');
}

$('btnOne').addEventListener('click', () => {
  const temp = +$('temp').value;
  const raw = lm.nextProbs($('prompt').value);
  const p = lm.withTemperature(raw, temp);
  const id = lm.sampleFrom(p);
  if (lm.chars[id] !== '\n') $('prompt').value += lm.chars[id];
  render();
  renderTrace([{ ch: lm.chars[id], p: p[id] }], $('prompt').value);
});

$('btnGen').addEventListener('click', () => {
  const temp = +$('temp').value;
  const r = lm.generate($('prompt').value, 40, temp);
  renderTrace(r.trace, r.text);
});

$('btnClear').addEventListener('click', () => {
  $('prompt').value = '';
  $('out').hidden = true;
  render();
});

render();

// ---------------------------------------------------------------------------
// 確認する
// ---------------------------------------------------------------------------
const CHECKS = [
  {
    id: 'howgen',
    kind: 'check',
    q: '確認：言語モデルは、文章をどうやって作っているでしょう？',
    options: [
      { t: '文全体を思い浮かべてから、一度に書き出す', why: '人間の書き方に近い説明ですが、モデルはこうしていません。次の1つを決めるたびに、そこまでの文字列をもう一度全部読み直しています。' },
      { t: '次の1つを確率で予測し、1つ選んで、また次を予測する' },
      { t: '覚えている文の中から、いちばん近いものを探して返す', why: 'それは検索（RAG の前半）です。言語モデルは文を保管しておらず、確率を計算しています。' },
    ],
    answer: 1,
    explain: 'これを繰り返しているだけです。だから出力は1文字（実際には1トークン）ずつ出てきますし、長い文章ほど時間がかかります。',
  },
  {
    id: 'temp',
    kind: 'check',
    q: '確認：同じ質問に毎回まったく同じ答えを返してほしいとき、温度はどうしますか？',
    options: [
      { t: '下げる（0 に近づける）' },
      { t: '上げる（1.5 以上にする）', why: '上げると低い確率の文字も選ばれるようになり、毎回違う答えになります。' },
      { t: '温度は速さの設定なので、答えは変わらない', why: '温度は「どれくらい冒険するか」の設定です。選び方そのものが変わります。' },
    ],
    answer: 0,
    explain: '温度を 0 に近づけると、いつも一番確率の高い文字が選ばれます（貪欲法）。事実を答えさせる用途では低め、発想を広げたいときは高めにします。',
  },
  {
    id: 'notllm',
    kind: 'check',
    q: '確認：このページのモデルと本物の LLM で、いちばん大きく違うのはどこでしょう？',
    options: [
      { t: '仕組みそのもの（次の1つを予測していない）', why: '仕組みは同じです。次の1つを確率で予測して継ぎ足しています。' },
      { t: '規模と材料（パラメータ数、学習した文章の量、語彙の単位）' },
      { t: '温度の使い方', why: '温度の使い方も同じです。本物でも、softmax の前に log 確率を温度で割っています。' },
    ],
    answer: 1,
    explain: 'このページは2万パラメータ・44文・文字単位。本物は数十億〜のパラメータを膨大な文章で学び、語彙はサブワードです。規模が変わると、文法や知識のような「学ばせていないもの」まで現れます。ただし1文字（1トークン）ずつ作ることは変わりません。',
  },
];
$('checks').innerHTML = CHECKS.map((c) => `<div id="chk-${c.id}"></div>`).join('');
for (const c of CHECKS) renderQuiz($(`chk-${c.id}`), { ...c, id: `llm:${c.id}` });

// ---------------------------------------------------------------------------
// 自分の言葉で
// ---------------------------------------------------------------------------
renderExplain($('explain'), {
  id: 'llm',
  q: 'LLM が文章を作るしくみを、AI を知らない友だちに説明してみてください',
  lead: '「予測している」で止まらず、何を予測しているのか、選び方をどう変えられるのかまで書けると強いです。',
  words: [
    { t: '次の1つ', any: ['次の1', '次のトークン', '次の文字', '次に来る'] },
    { t: '確率', any: ['確率', '%', 'パーセント'] },
    { t: '繰り返す', any: ['繰り返', 'くりかえ', '何度も', 'そのたびに'] },
    { t: '温度', any: ['温度', 'Temperature', 'テンペラ'] },
    { t: '学習した材料', any: ['学習', '材料', 'データ', '学んだ'] },
  ],
  model: [
    'モデルは、そこまでの文字列を読んで、<b>次に来る1つ（1トークン）の確率</b>を語彙ぶん全部について出します。「よろしくお」まで来たら「ね」が 98%、という具合です。',
    'その確率から1つ選んで文字列の末尾に足し、<b>またそこまでを読み直して次を予測します</b>。これを終わりの合図が出るまで繰り返すだけです。文章全体を先に考えているわけではありません。',
    '選び方には幅があります。<b>温度</b>を下げると一番確率の高いものばかり選ぶので答えが安定し、上げると低い確率のものも選ばれて意外さが出る代わりに崩れやすくなります。',
    '確率は、学習した材料から決まります。材料にない言い回しは出てきませんし、材料に偏りがあれば出力も偏ります。',
  ],
  checks: [
    '「次の1つを予測して継ぎ足す」という繰り返しを書いた',
    '出しているのが1つの答えではなく「確率」だと書いた',
    '温度で選び方が変わることに触れた',
    '確率が学習した材料から決まることに触れた',
    '文章全体を先に考えているわけではない、と分かる書き方になっている',
  ],
});

// ---------------------------------------------------------------------------
// コードで見る
// ---------------------------------------------------------------------------
const box = document.createElement('section');
box.className = 'panel';
$('explain').closest('.panel').insertAdjacentElement('beforebegin', box);
codeBox(box, {
  items: [
    {
      label: '次の1つを予測して継ぎ足す（生成ループ）',
      file: 'public/js/charlm.js の generate()',
      py: `import torch, torch.nn.functional as F

ids = tokenizer.encode(prompt)
for _ in range(max_new_tokens):
    logits = model(ids[-block_size:])[:, -1, :]   # 最後の位置だけ使う
    logits = logits / temperature                  # 温度
    probs  = F.softmax(logits, dim=-1)
    nxt    = torch.multinomial(probs, num_samples=1)
    ids    = torch.cat([ids, nxt], dim=1)
    if nxt.item() == eos_id:
        break`,
      js: `let s = prefix;
for (let i = 0; i < n; i++) {
  const raw = this.nextProbs(s);        // 最後の位置の確率
  const p = this.withTemperature(raw, temp);
  const id = this.sampleFrom(p);
  s += this.chars[id];
  if (this.chars[id] === '\\n') break;  // 終わりの合図
}`,
      note: '温度はロジットを割るだけ（log 確率を T で割るのと同じ）。T → 0 で argmax、T が大きいほど一様に近づきます。',
    },
    {
      label: '学習（次の文字を当てる問題として解く）',
      file: 'public/js/charlm.js の trainStep()',
      py: `# 入力を1つずらしたものが正解になる
x = batch[:, :-1]
y = batch[:, 1:]

logits = model(x)
loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
loss.backward()
opt.step()`,
      js: `for (let t = 0; t < T; t++) {
  inp[b * T + t] = line[s + t];
  tgt[b * T + t] = line[s + t + 1];   // 1つ先が正解
}
const P = softmaxRows(net.forward(inp, B), B * T, V);
for (...) d[o + j] = (P[o + j] - (j === tgt[i] ? 1 : 0)) / (B * T);
net.backward(d);`,
      note: '正解ラベルを人が付ける必要がありません。文章そのものが「次は何か」の正解表になっています（自己教師あり学習）。',
    },
  ],
});

enableGlossary(['.hero', '.note', '.guide-grid', '.ex-model', '.lab-body']);

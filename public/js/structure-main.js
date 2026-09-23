// 共通のしくみ：5つのモデルを同じ枠に並べて見る。
import { MODELS, STAGES, FLOWS, CHECKS } from './structure.js';
import { mountHeader, mountGoal, enableGlossary } from './common.js';
import { renderQuiz } from './quiz.js';
import { renderExplain } from './explain.js';

mountHeader({ id: 'structure', mode: 'detail', sub: '5つのモデルを、入力から学習まで同じ枠に並べて見ます' });
mountGoal('structure');

const $ = (id) => document.getElementById(id);
let open = null;      // いま開いている段
let plain = false;    // 共通の言い方で見る

function renderTable() {
  $('table').innerHTML = `
    <thead>
      <tr>
        <th class="st-corner">段</th>
        ${MODELS.map((m) => `<th><a href="${m.page}" style="color:${m.color}">${m.label}</a></th>`).join('')}
      </tr>
    </thead>
    <tbody>
      ${STAGES.map((s, i) => `
        <tr data-i="${i}"${open === i ? ' class="on"' : ''}>
          <th class="st-stage">
            <button data-i="${i}"><b>${s.name}</b><span>${s.q}</span></button>
          </th>
          ${s.cells.map((c) => `<td>${plain ? `<span class="st-plain">${s.q}</span>` : `<b>${c.t}</b><span>${c.s}</span>`}</td>`).join('')}
        </tr>`).join('')}
    </tbody>`;
}

function renderNote() {
  if (open === null) {
    $('note').innerHTML = '<p class="note">段の名前をクリックすると、その段の解説が出ます。</p>';
    return;
  }
  const s = STAGES[open];
  $('note').innerHTML = `
    <div class="st-open">
      <h3>${s.name} — ${s.q}</h3>
      <p><span class="st-tag same">共通</span>${s.same}</p>
      <p><span class="st-tag diff">分かれる</span>${s.diff}</p>
    </div>`;
}

function renderFlows() {
  $('flows').innerHTML = MODELS.map((m) => `
    <div class="flow">
      <div class="flow-name" style="color:${m.color}">${m.label}</div>
      <ol>${FLOWS[m.id].map((t, i) => `<li${i === 0 ? ' class="flow-first"' : ''}>${t}</li>`).join('')}</ol>
    </div>`).join('');
}

$('table').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-i]');
  if (!b) return;
  const i = +b.dataset.i;
  open = open === i ? null : i;
  renderTable();
  renderNote();
});

$('plain').addEventListener('change', (e) => {
  plain = e.target.checked;
  renderTable();
});

renderTable();
renderNote();
renderFlows();

// ---- 確認する ----
$('checks').innerHTML = CHECKS.map((c) => `<div id="chk-${c.id}"></div>`).join('');
for (const c of CHECKS) renderQuiz($(`chk-${c.id}`), { ...c, id: `structure:${c.id}` });

// ---- 自分の言葉で ----
renderExplain($('explain'), {
  id: 'structure',
  q: '5つのモデルに共通する骨格を、説明してみてください',
  lead: '個々のモデルの説明ではなく、「どれも同じところ」と「分かれるところ」を書けると、次に新しいモデルが出てきたときに当てはめられます。',
  words: [
    { t: '入力 → 計算 → 出力', any: ['入力', '出力'] },
    { t: '損失', any: ['損失', 'loss', 'ロス', '良し悪し'] },
    { t: '更新（学習）', any: ['更新', '学習', '勾配', 'パラメータ'] },
    { t: '知識の置き場所', any: ['重み', '外', '資料', '知識', '置'] },
    { t: '推論', any: ['推論', '使うとき', '生成するとき'] },
  ],
  model: [
    'どれも <b>入力を数に直し → 計算し → 出力し → その良し悪しを1つの数（損失）で測り → パラメータを少し更新する</b>、という同じ骨格でできています。違うのは、その骨格のどこに工夫を入れたかです。',
    'GAN は<b>測り方</b>を変えました。良し悪しを人が決める代わりに、もう1つのネットワーク（Discriminator）に決めさせます。そのぶん損失が2つになり、学習が不安定になりました。',
    'VAE は<b>表現</b>を変えました。潜在変数を1点ではなく範囲（μ と σ）にして、全体を共通の書式にそろえます。だから知らない z からでも絵が描けます。',
    'Transformer は<b>作り方</b>を変えました。全体を一度に作らず、「次の1つ」だけを予測して継ぎ足します。そのぶん確率をきちんと出せます。',
    'RAG は<b>知識の置き場所</b>を変えました。重みの中ではなく外の資料に置き、使うたびに検索して渡します。学習はしません。',
  ],
  checks: [
    '「入力 → 計算 → 出力 → 損失 → 更新」という共通の骨格に触れた',
    'GAN の損失が2つあることと、その結果（不安定さ）を結びつけた',
    'VAE の潜在変数が「範囲」であることに触れた',
    'RAG が学習しないこと、知識が外にあることを書いた',
    '学習のときと推論のときで、使う部品が違うことに触れた',
  ],
});

enableGlossary(['.hero', '.st-open', '.guide-grid', '.note', '.ex-model', '.st-table td']);

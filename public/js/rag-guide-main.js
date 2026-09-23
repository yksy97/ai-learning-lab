// 「図書館の司書」ガイド：RAG のしくみを順番に見せる
import { RagIndex } from './rag.js';
import { RagMap } from './rag-viz.js';
import { mountHeader } from './common.js';

const $ = (id) => document.getElementById(id);

mountHeader({
  id: 'rag', mode: 'guide',
  sub: '「質問を受けた司書が資料を探し、その資料を読んで作家が答えを書く」として、RAG のしくみを順番に見ていく',
});

const C = {
  lib: '<span class="chip police">図書館（資料）</span>',
  writer: '<span class="chip fake">作家（LLM）</span>',
  librarian: '<span class="chip real">司書（検索）</span>',
};

const QUESTION = '出張の宿泊費はいくらまで出ますか？';
const MISSING = '育児休業は何日取れますか？';
const PARAPHRASE = 'ChatGPT を仕事で使ってもいいですか？';

const index = new RagIndex();
index.projection();
const map = new RagMap($('map'));

const state = { step: 0, question: null, k: 3, threshold: 0.2, expand: true, show: {} };

function searchNow() {
  if (!state.question) return null;
  index.opt.expand = state.expand;
  const res = index.search(state.question);
  const scores = new Array(index.chunks.length).fill(0);
  for (const r of res) scores[r.index] = r.score;
  const picked = res.filter((r) => r.score >= state.threshold).slice(0, state.k);
  return { res, scores, picked, pickedSet: new Set(picked.map((r) => r.index)), query: index.projectQuery(state.question) };
}

const STEPS = [
  {
    title: '新人AIの困りごと',
    show: { dim: true },
    body: `<p>ここに、とても賢い ${C.writer} がいます。日本語は上手ですが、<b>あなたの会社の社内規程は読んだことがありません</b>。</p>
      <p>「出張の宿泊費はいくらまで出ますか？」と聞いても、作家は知りません。それでも聞かれれば、それらしい文章を書いてしまいます。これが<b>もっともらしい嘘</b>（ハルシネーション）の正体です。</p>
      <p>そこで、答える前に<b>資料を探してくる担当</b>を付けます。これが RAG の考え方です。</p>`,
    term: `<span class="k">専門用語では</span>RAG ＝ <b>Retrieval-Augmented Generation</b>（検索で補強した生成）。作家 ＝ <b>LLM</b>。`,
  },
  {
    title: '資料を棚に並べる',
    show: { map: true },
    body: `<p>まず ${C.lib} を用意します。ここでは架空の会社の社内規程 ${index.chunks.length} 件です。</p>
      <p>長い規程はそのままでは扱いにくいので、条文くらいの大きさに切り分けておきます。この一つひとつが「棚に並んだ資料」です。</p>
      <p>右下の図で、点ひとつが資料1件です。</p>`,
    term: `<span class="k">専門用語では</span>切り分けること ＝ <b>チャンク化</b>。切り分けた資料を検索できる形にしておくこと ＝ <b>インデックス作成</b>。`,
  },
  {
    title: '言葉を数字にする',
    show: { map: true },
    body: `<p>司書が資料を探せるように、<b>文章を数字の列に変えて</b>おきます。似た内容の資料が近くに来るような数字の付け方をします。</p>
      <p>図は、その数字を2次元に押しつぶした「資料の地図」です。経費の話どうし、セキュリティの話どうしが、だいたい固まっているのが見えます。</p>
      <p>点にマウスを乗せると、その資料の中身が出ます。</p>`,
    term: `<span class="k">専門用語では</span>文章を数字の列にすること ＝ <b>埋め込み（ベクトル化）</b>。この画面では、文字2文字の組み合わせの出やすさ（TF-IDF）で数字にしている。本物の RAG は意味を捉えるニューラルネットを使う。`,
  },
  {
    title: '司書が資料を探す',
    show: { map: true, ranking: true },
    question: QUESTION,
    body: `<p>質問「${QUESTION}」を ${C.librarian} に渡しました。</p>
      <p>司書は質問も同じやり方で数字に変え、地図の上に置きます（ひし形）。あとは<b>近い順に資料を取る</b>だけです。線でつながれた資料が、司書が選んだものです。</p>
      <p>右下のリストが、近い順に並べた結果です。数字は近さ（類似度）で、1 に近いほど似ています。</p>`,
    term: `<span class="k">専門用語では</span>近さ ＝ <b>コサイン類似度</b>。上位 k 件を取ること ＝ <b>top-k 検索</b>。`,
  },
  {
    title: '資料を添えて作家に渡す',
    show: { map: true, prompt: true },
    question: QUESTION,
    body: `<p>司書は取ってきた資料を、質問といっしょに1枚の指示書にまとめて作家に渡します。これが<b>プロンプト</b>です。</p>
      <p>下の指示書を読んでみてください。「この資料だけを根拠に答えること」「書いていないことは<b>記載がありません</b>と答えること」と、はっきり指示しています。</p>
      <p>作家は物知りである必要がありません。<b>渡された資料を読んで書く</b>だけでよくなります。</p>`,
    term: `<span class="k">専門用語では</span>資料を詰めた指示書 ＝ <b>拡張プロンプト</b>。「資料だけを根拠に」＝ <b>グラウンディング</b>。`,
  },
  {
    title: '作家が答えを書く',
    show: { map: true, answer: true, prompt: true },
    question: QUESTION,
    body: `<p>作家が資料を読んで答えます。答えには <b>[1]</b> のように根拠の番号が付きます。これがあると、人間があとから確認できます。</p>
      <p>この画面には本物の LLM が入っていないので、答えは「取ってきた資料から関係の深い文を抜き出したもの」です。文章のなめらかさは本物に劣りますが、<b>根拠が資料の中にある</b>という RAG の肝は同じです。</p>`,
    term: `<span class="k">専門用語では</span>根拠の番号 ＝ <b>引用（citation）</b>。資料に基づいて答えること ＝ <b>根拠づけされた生成</b>。`,
  },
  {
    title: '失敗その1：棚にない質問',
    show: { map: true, ranking: true, answer: true },
    question: MISSING,
    body: `<p>今度は「${MISSING}」と聞いてみます。この会社の資料には育児休業の規程がありません。</p>
      <p>地図を見ると、質問のひし形がどの資料からも離れていて、どの類似度も低いままです。司書は「近い資料がない」と分かります。</p>
      <p>ここで大事なのは、<b>近い資料がないときは何も渡さない</b>という判断です。無理に「いちばんマシな資料」を渡すと、作家はそれらしい嘘を書いてしまいます。</p>`,
    term: `<span class="k">専門用語では</span>この下限 ＝ <b>類似度のしきい値</b>。渡す資料がないときに素直に「わからない」と答えさせるのが、RAG の品質管理の基本。`,
  },
  {
    title: '失敗その2：言い方が違う',
    show: { map: true, ranking: true },
    question: PARAPHRASE,
    expand: false,
    body: `<p>「${PARAPHRASE}」と聞いてみます。資料には「生成AI」と書かれていますが、「ChatGPT」という語は<b>1回も出てきません</b>。</p>
      <p>いまは語の一致だけで探しているので、司書はこの資料を見つけられません。類似度は 0 のままです。</p>
      <p>下のボタンで「言い換えを考慮する」に切り替えると、見つけられるようになります。本物の RAG では、ここを<b>意味のベクトル（埋め込み）</b>が引き受けます。</p>`,
    actions: [{ label: '言い換えを考慮する', toggle: 'expand' }],
    term: `<span class="k">専門用語では</span>語の一致で探す ＝ <b>キーワード検索（疎な検索）</b>、意味で探す ＝ <b>ベクトル検索（密な検索）</b>。両方を混ぜる <b>ハイブリッド検索</b>が実務ではよく使われる。`,
  },
  {
    title: 'まとめ',
    show: { map: true, ranking: true, answer: true },
    question: QUESTION,
    body: `<p>RAG は「賢い作家をもっと賢くする」技術ではなく、<b>作家に正しい資料を渡す</b>技術です。</p>
      <p>だから RAG がうまくいくかどうかは、後半の LLM より<b>前半の検索</b>で決まります。資料の切り分け方、探し方、渡す件数、しきい値。どれも地味ですが、ここが品質のほとんどを決めます。</p>
      <div class="links"><a href="rag.html">詳細タブで自由に試す →</a></div>`,
    term: `<table>
      <tr><td>図書館の資料</td><td>ナレッジベース（社内文書など）</td></tr>
      <tr><td>資料を切り分ける</td><td>チャンク化</td></tr>
      <tr><td>資料を数字にする</td><td>埋め込み（ベクトル化）</td></tr>
      <tr><td>棚（探せる状態）</td><td>ベクトルデータベース</td></tr>
      <tr><td>司書</td><td>検索（リトリーバ）</td></tr>
      <tr><td>近い順に数件取る</td><td>top-k 検索</td></tr>
      <tr><td>資料つきの指示書</td><td>拡張プロンプト</td></tr>
      <tr><td>作家</td><td>LLM（生成モデル）</td></tr>
      <tr><td>根拠の番号</td><td>引用</td></tr>
      <tr><td>それらしい嘘</td><td>ハルシネーション</td></tr>
    </table>`,
  },
];

function enterStep(i) {
  state.step = i;
  const s = STEPS[i];
  state.question = s.question || null;
  state.expand = s.expand === undefined ? true : s.expand;
  state.show = s.show || {};
  $('stepNo').textContent = `${i + 1} / ${STEPS.length}`;
  $('stepTitle').textContent = s.title;
  $('stepBody').innerHTML = s.body;
  $('stepTerm').innerHTML = s.term || '';
  $('prev').disabled = i === 0;
  $('next').disabled = i === STEPS.length - 1;
  [...$('dots').children].forEach((d, k) => d.classList.toggle('on', k === i));
  renderActions();
  render();
}

function renderActions() {
  const s = STEPS[state.step];
  const box = $('stepActions');
  box.innerHTML = '';
  for (const a of s.actions || []) {
    const b = document.createElement('button');
    b.className = 'act';
    b.textContent = state.expand ? '語の一致だけに戻す' : a.label;
    b.addEventListener('click', () => { state.expand = !state.expand; renderActions(); render(); });
    box.appendChild(b);
  }
}

function render() {
  const r = searchNow();
  $('stageTitle').textContent = state.question ? `図書館　質問：${state.question}` : '図書館';
  map.draw({
    coords: index.coords,
    chunks: index.chunks,
    query: r ? r.query : null,
    picked: r ? r.pickedSet : new Set(),
    scores: r ? r.scores : new Array(index.chunks.length).fill(state.show.dim ? 0 : 0.15),
  });

  const out = $('output');
  out.innerHTML = '';
  if (!r) return;
  if (state.show.ranking) {
    const rows = r.res.slice(0, 6).map((x, i) => {
      const on = r.pickedSet.has(x.index);
      return `<div class="out-row ${on ? 'on' : ''}">
        <span class="num">${i + 1}</span>
        <span><b class="t">${x.chunk.title}</b><div class="bar"><i style="width:${Math.min(100, x.score * 200)}%"></i></div></span>
        <span class="sc">${x.score.toFixed(3)}</span>
      </div>`;
    }).join('');
    out.insertAdjacentHTML('beforeend',
      `<div class="out-box"><h3>司書が並べた資料（近い順）　しきい値 ${state.threshold.toFixed(2)} 以上を ${state.k} 件まで採用</h3><div class="out-list">${rows}</div></div>`);
  }
  if (state.show.prompt) {
    out.insertAdjacentHTML('beforeend',
      `<div class="out-box"><h3>作家に渡す指示書（プロンプト）</h3><pre class="prompt">${RagIndex.buildPrompt(state.question, r.picked).replace(/</g, '&lt;')}</pre></div>`);
  }
  if (state.show.answer) {
    let body;
    if (!r.picked.length) {
      body = `<b>資料には記載がありません。</b>（いちばん近い資料でも類似度 ${r.res[0].score.toFixed(3)}）`;
    } else {
      const sents = [];
      const qVec = index.queryVector(state.question).vec;
      r.picked.forEach((p, i) => {
        for (const sent of p.chunk.text.split(/(?<=。)/).map((t) => t.trim()).filter(Boolean)) {
          let sc = 0;
          const seen = new Set();
          for (let j = 0; j + 1 < sent.length; j++) {
            const t = sent.slice(j, j + 2);
            if (seen.has(t)) continue;
            seen.add(t);
            sc += qVec.get(t) || 0;
          }
          sents.push({ sent, sc, cite: i + 1, rank: i });
        }
      });
      sents.sort((a, b) => b.sc - a.sc || a.rank - b.rank);
      body = sents.slice(0, 2).sort((a, b) => a.rank - b.rank).map((s) => `${s.sent} [${s.cite}]`).join(' ') +
        `<div class="cites">${r.picked.map((p, i) => `[${i + 1}] ${p.chunk.title}`).join('　')}</div>`;
    }
    out.insertAdjacentHTML('beforeend', `<div class="out-box"><h3>作家の答え</h3><div class="answer">${body}</div></div>`);
  }
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
  });
  const canvas = $('map');
  canvas.addEventListener('pointermove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const i = map.pick(e.clientX - rect.left, e.clientY - rect.top, index.coords);
    if (i !== map.hover) { map.hover = i >= 0 ? i : null; render(); }
    canvas.title = i >= 0 ? `${index.chunks[i].title}\n${index.chunks[i].text}` : '';
  });
  canvas.addEventListener('pointerleave', () => { map.hover = null; render(); });
  window.addEventListener('resize', render);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
  enterStep(0);
}

init();

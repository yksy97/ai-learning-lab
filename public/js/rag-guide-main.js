// 「図書館の司書」ガイド：RAG のしくみを順番に見せる
import { RagIndex } from './rag.js';
import { RagMap } from './rag-viz.js';
import { renderQuiz } from './quiz.js';
import { renderExplain } from './explain.js';
import { mountHeader, mountGoal, enableGlossary } from './common.js';

const $ = (id) => document.getElementById(id);

mountHeader({
  id: 'rag', mode: 'guide',
  sub: '「司書が資料を探し、その資料を読んで作家が答えを書く」として、RAG のしくみを順に見ていきます',
});
mountGoal('rag-guide');

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
    quiz: {
      q: '司書は、質問と資料の「近さ」をどうやって測っているでしょう？',
      options: [
        { t: '質問に含まれる語が、その資料に何個あるかを数える', why: '惜しいです。素朴に数えると、長い資料ほど有利になり、「の」のようなどこにでもある語も同じ重みになります。TF-IDF（珍しい語を重く）とコサイン類似度（長さで割る）は、まさにその2点を直したものです。' },
        { t: '数字のベクトルに変えて、向きの近さを測る' },
        { t: '資料を先頭から読んで、意味が合うかを判断する', why: '読んで判断しているのは後半の LLM です。検索の段階にあるのは数式だけで、意味の理解は入っていません。' },
      ],
      answer: 1,
      explain: '質問も資料も同じやり方でベクトルにし、コサイン類似度で比べています。地図はその近さを2次元に写したものです。',
    },
    show: { map: true, ranking: true },
    question: QUESTION,
    body: `<p>質問「${QUESTION}」を ${C.librarian} に渡しました。</p>
      <p>地図のひし形が質問の位置、線でつながれた資料が司書の選んだものです。司書がどうやって「近さ」を決めているか、下の問いに答えてみてください。</p>`,
    reveal: '<p>司書は質問も同じやり方で数字に変え、地図の上に置きます。あとは<b>近い順に取る</b>だけです。右下のリストの数字が近さ（類似度）で、1 に近いほど似ています。</p>',
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
    quiz: {
      q: '資料にない質問をされたとき、いちばん良い振る舞いは？',
      options: [
        { t: 'いちばん近い資料を1件だけ渡して、使うかどうかは LLM に判断させる', why: 'もっともらしいのですが危険です。LLM は渡された資料を「使うべきもの」と受け取るので、無関係でも無理に当てはめて書きます。渡す前に止めるほうが安全です。' },
        { t: '資料を渡さず、「記載がない」と答えさせる' },
        { t: 'k を増やして、候補をもっと広げる', why: '関係のない資料が増えるだけです。そもそも答えが資料の中にないので、広げても見つかりません。' },
      ],
      answer: 1,
      explain: '関係のない資料は、それらしい嘘の材料になります。しきい値を決めておき、届かなければ答えないのが基本です。',
    },
    show: { map: true, ranking: true, answer: true },
    question: MISSING,
    body: `<p>今度は「${MISSING}」と聞いてみます。この会社の資料には育児休業の規程がありません。</p>
      <p>地図を見ると、質問のひし形がどの資料からも離れていて、どの類似度も低いままです。このとき司書はどうするのがよいでしょうか。</p>`,
    reveal: '<p>大事なのは、<b>近い資料がないときは何も渡さない</b>という判断です。無理に「いちばんマシな資料」を渡すと、作家はそれらしい嘘を書いてしまいます。</p>',
    term: `<span class="k">専門用語では</span>この下限 ＝ <b>類似度のしきい値</b>。渡す資料がないときに素直に「わからない」と答えさせるのが、RAG の品質管理の基本。`,
  },
  {
    title: '失敗その2：言い方が違う',
    quiz: {
      q: '資料に「生成AI」としか書かれていないとき、「ChatGPT」で検索するとどうなる？',
      options: [
        { t: '文字の 2-gram で分けているので、「AI」の部分が一致して見つかる', why: '技術的にはもっともらしいのですが、この実装では英数字はひとかたまりの単語として扱います。「chatgpt」と「ai」は別のトークンなので、部分的にも一致しません。' },
        { t: '語が一致しないので見つからない（類似度 0）' },
        { t: '意味が近いので、順位は落ちるが上位には入る', why: 'この画面の検索に「意味」は入っていません。語が重ならなければ、内積は 0 のままです。意味で拾えるようにするのが埋め込みの役目です。' },
      ],
      answer: 1,
      explain: '語の一致だけで探すとこうなります。実際の RAG では、意味を捉える埋め込みがこの差を吸収します。',
    },
    show: { map: true, ranking: true },
    question: PARAPHRASE,
    expand: false,
    body: `<p>「${PARAPHRASE}」と聞いてみます。資料には「生成AI」と書かれていますが、「ChatGPT」という語は<b>1回も出てきません</b>。</p>
      <p>いまは語の一致だけで探しています。この質問で目的の資料が見つかるか、予想してから下のリストを見てください。</p>`,
    reveal: '<p>見つけられません（類似度 0）。下のボタンで「言い換えを考慮する」に切り替えると見つかります。本物の RAG では、ここを<b>意味のベクトル（埋め込み）</b>が引き受けます。</p>',
    actions: [{ label: '言い換えを考慮する', toggle: 'expand' }],
    term: `<span class="k">専門用語では</span>語の一致で探す ＝ <b>キーワード検索（疎な検索）</b>、意味で探す ＝ <b>ベクトル検索（密な検索）</b>。両方を混ぜる <b>ハイブリッド検索</b>が実務ではよく使われる。`,
  },
  {
    title: 'まとめ',
    quiz: {
      kind: 'check',
      q: '確認：RAG の答えの質は、主にどこで決まるでしょう？',
      options: [
        { t: 'LLM の賢さ', why: '渡した資料に答えが入っていなければ、賢さでは埋まりません。埋めようとした結果がハルシネーションです。' },
        { t: '前半の検索（何を渡せたか）' },
        { t: 'プロンプトの指示の丁寧さ', why: '「資料だけを根拠に」という指示は歯止めとして有効ですが、良い資料がなければ良い答えにはなりません。効くのは資料があるときです。' },
      ],
      answer: 1,
      explain: '渡した資料に答えが含まれていなければ、どんな LLM でも正しく答えられません。RAG の改善は検索の改善から始めます。',
    },
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
  if (s.quiz) renderQuiz(qz, { id: `rag-guide:${i}`, ...s.quiz }, () => {
    const rv = $('stepBody').querySelector('.reveal');
    if (rv) rv.hidden = false;
  });
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
  enableGlossary();
  enterStep(0);
}

init();

// ---- 自分の言葉で説明する ----
renderExplain(document.getElementById('explain'), {
  id: 'rag',
  q: 'RAG が「学習していないこと」を答えられる理由を、説明してみてください',
  lead: 'モデルを賢くしているわけではない、という点が伝わるかどうかが分かれ目です。',
  words: [
    { t: 'チャンク（資料を切る）', any: ['チャンク', '切り分け', '分割', '条文ごと', '文ごと'] },
    { t: 'ベクトルにする', any: ['ベクトル', '数字に', '数値に', '埋め込み', 'TF-IDF'] },
    { t: 'コサイン類似度', any: ['コサイン', 'cos', '向き', '類似度', '近さ'] },
    { t: '上位 k 件', any: ['上位', 'k 件', 'k件', '何件', 'しきい値', '上から'] },
    { t: 'プロンプトに貼る', any: ['プロンプト', '一緒に渡', '添え', '貼り付け', '資料を渡'] },
    { t: '出典', any: ['出典', 'どの資料', '根拠', '引用'] },
  ],
  model: [
    'モデルの中身（重み）は一切さわりません。やっていることは<b>その場で資料を渡す</b>だけです。まず資料を扱いやすい大きさに<b>切り分け</b>（チャンク）、それぞれを<b>ベクトル</b>にして棚に並べておきます。',
    '質問が来たら、質問も同じ方法でベクトルにして、<b>コサイン類似度</b>（向きの近さ）が大きい順に並べ、<b>上位 k 件</b>だけを取り出します。ここが司書の仕事です。',
    '取り出した資料を、質問といっしょに<b>プロンプトに貼って</b>言語モデルに渡します。モデルは「目の前にある資料を読んで答える」だけなので、学習していない社内規程でも答えられます。ついでに<b>どの資料から答えたか（出典）</b>を示せるのも、この形の利点です。',
    '弱点も同じところから来ます。検索で取り逃がした資料のことは答えられませんし（言い方が違うだけで順位が落ちる）、資料に書いていないことを聞かれたときに、黙って想像で埋めないようにする工夫が別に必要です。',
  ],
  checks: [
    'モデルを再学習していない（知識は外にある）ことを書いた',
    '検索 → プロンプトに貼る → 答える、の順番が書けている',
    '「近さ」をコサイン類似度で測っていることに触れた',
    '出典が示せる理由が分かる書き方になっている',
    '弱点（検索で取り逃がす／資料にないことは答えられない）を1つ挙げた',
  ],
});

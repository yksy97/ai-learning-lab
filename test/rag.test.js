// RAG の検索テスト：想定した資料が上位に来るか、関係ない質問で類似度が下がるか
import assert from 'node:assert/strict';
import { RagIndex, tokenize, chunkCorpus } from '../public/js/rag.js';

{
  const t = tokenize('宿泊費の上限は？ PC も');
  assert.ok(t.includes('宿泊'), '2-gram が作れていない');
  assert.ok(t.includes('pc'), '英数字が拾えていない');
}

const idx = new RagIndex();
console.log(`チャンク数 ${idx.chunks.length} / 語彙数 ${idx.vocab.length}`);

const cases = [
  ['出張の宿泊費はいくらまで出ますか？', 'keihi-03'],
  ['領収書をなくしてしまった場合はどうすればいい？', 'keihi-05'],
  ['在宅勤務のときに通信費はもらえますか', 'zaitaku-03'],
  ['パスワードのルールを教えて', 'sec-01'],
  ['ChatGPT を仕事で使ってもいいですか？', 'sec-04'],
  ['健康診断はいつ？', 'anzen-01'],
  ['荷物が壊れていたときの対応', 'unyu-02'],
];
let hit1 = 0;
for (const [q, want] of cases) {
  const res = idx.search(q);
  const rank = res.findIndex((r) => r.chunk.id === want) + 1;
  if (rank === 1) hit1++;
  console.log(`${rank === 1 ? '○' : '△'} 「${q}」→ 1位 ${res[0].chunk.id} (${res[0].score.toFixed(3)}) / 正解 ${want} は ${rank} 位`);
  assert.ok(rank <= 3, `${want} が上位3件に入っていない`);
}
assert.ok(hit1 >= 6, '1位で当てられた質問が少なすぎる');

// 言い換えを切ると「ChatGPT」は資料の「生成AI」に結びつかない（語の一致の限界）
{
  const plain = new RagIndex({ expand: false });
  const rank = plain.search('ChatGPT を仕事で使ってもいいですか？').findIndex((r) => r.chunk.id === 'sec-04') + 1;
  console.log(`言い換えなし：「ChatGPT」→ sec-04 は ${rank} 位`);
  assert.ok(rank > 3, '言い換えなしでも当たってしまう（デモにならない）');
}

// 資料にない質問では、最大の類似度が低くなる
for (const q of ['育児休業は何日取れますか？', '会社の株価はいくらですか？']) {
  const top = idx.search(q)[0];
  console.log(`× 「${q}」→ 最大類似度 ${top.score.toFixed(3)}（${top.chunk.id}）`);
  assert.ok(top.score < 0.25, `資料にない質問なのに類似度が高い: ${top.score}`);
}

// 文単位に分割しても動く
const idx2 = new RagIndex({ chunkMode: 'sentence' });
assert.ok(idx2.chunks.length > idx.chunks.length);
const r2 = idx2.search('宿泊費の上限');
console.log(`文単位: ${idx2.chunks.length} チャンク / 1位 ${r2[0].chunk.id} (${r2[0].score.toFixed(3)})`);
assert.ok(r2[0].chunk.source === 'keihi-03');

// 2次元の地図：同じ規程どうしが近くに来ているか（ざっくり確認）
const co = idx.projection();
assert.equal(co.length, idx.chunks.length);
const group = (id) => id.split('-')[0];
let same = 0;
for (let i = 0; i < co.length; i++) {
  let bj = -1, bd = Infinity;
  for (let j = 0; j < co.length; j++) {
    if (i === j) continue;
    const d = Math.hypot(co[i][0] - co[j][0], co[i][1] - co[j][1]);
    if (d < bd) { bd = d; bj = j; }
  }
  if (group(idx.chunks[i].id) === group(idx.chunks[bj].id)) same++;
}
console.log(`地図：いちばん近い文書が同じ規程グループだったのは ${same} / ${co.length} 件`);
assert.ok(same / co.length >= 0.55, '地図の配置が内容を反映していない');

const q = idx.projectQuery('宿泊費の上限は？');
assert.ok(Number.isFinite(q[0]) && Number.isFinite(q[1]));
console.log('OK');

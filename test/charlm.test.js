// 文字単位の言語モデルのテスト：語彙・学習が進むこと・温度の性質
import { CharLM } from '../public/js/charlm.js';
import { mulberry32 } from '../public/js/nn.js';

const ok = (cond, msg) => { if (!cond) { console.error('NG:', msg); process.exit(1); } console.log('  ○', msg); };

const lm = new CharLM();
console.log(`語彙 ${lm.V} 種類 / 文 ${lm.lines.length} / 2-gram の下限 ${lm.entropy.toFixed(3)}`);
ok(lm.V > 30 && lm.V < 120, `語彙の大きさが妥当（${lm.V}）`);
ok(lm.chars.includes('\n'), '文の終わりの合図が語彙にある');

// 最後の「。→ 改行」が学習対象に入っているか（入っていないと生成が止まらない）
let sawEnd = false;
for (let i = 0; i < 400 && !sawEnd; i++) {
  const T = lm.cfg.seqLen;
  const { line, s } = lm.sampleWindow(T);
  for (let t = 0; t < T; t++) if (s + t + 1 === line.length - 1) sawEnd = true;
}
ok(sawEnd, '文末（。→ 改行）が学習の対象に入る');

const before = lm.trainStep().nll;
for (let i = 0; i < 1500; i++) lm.trainStep();
const after = lm.last.nll;
console.log(`NLL ${before.toFixed(3)} → ${after.toFixed(3)}`);
ok(after < before * 0.4, '学習で損失が大きく下がる');
ok(after < 1.0, `損失が 1.0 を下回る（${after.toFixed(3)}）`);

const p = lm.nextProbs('よろしくお');
const top = lm.top(p, 1)[0];
console.log(`「よろしくお」→ ${top.ch} ${(top.p * 100).toFixed(1)}%`);
ok(top.ch === 'ね' && top.p > 0.8, '定型文の続きを当てられる');

const end = lm.top(lm.nextProbs('ありがとうございます。'), 1)[0];
ok(end.ch === '\n' && end.p > 0.5, '文末では「終わり」を高い確率で選ぶ');

// 温度の性質：下げると尖り、上げると平らになる
const raw = lm.nextProbs('よろしくお');
const cold = lm.withTemperature(raw, 0.2);
const hot = lm.withTemperature(raw, 2.0);
const maxOf = (a) => Math.max(...a);
console.log(`最大確率 温度0.2 ${maxOf(cold).toFixed(3)} / そのまま ${maxOf(raw).toFixed(3)} / 温度2.0 ${maxOf(hot).toFixed(3)}`);
ok(maxOf(cold) >= maxOf(raw) && maxOf(raw) >= maxOf(hot), '温度を上げるほど分布が平らになる');
const sum = (a) => [...a].reduce((x, y) => x + y, 0);
ok(Math.abs(sum(hot) - 1) < 1e-9, '温度をかけても確率の合計は 1');

// 生成：低温では材料にある文がそのまま出る
const r = mulberry32(42);
const gen = lm.generate('', 40, 0.2, r).text.replace(/\n/g, '');
console.log(`温度0.2 の生成: ${gen}`);
ok(gen.length > 4 && gen.endsWith('。'), '低い温度では文として完成する');

console.log('OK');

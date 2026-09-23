// 学習対象となる「本物」の2次元分布
import { gaussian } from './nn.js';

export const DATASETS = {
  gaussian: {
    label: '1つのガウス分布',
    note: '最も簡単な例。位置と広がりが合っていく様子が見やすい。',
    sample(r) {
      const x = gaussian(r), y = gaussian(r);
      return [0.6 + 0.35 * x + 0.15 * y, -0.4 + 0.25 * y];
    },
  },
  ring8: {
    label: '円周上の8つのガウス',
    modes: { centers: Array.from({ length: 8 }, (_, k) => [1.4 * Math.cos((k / 8) * Math.PI * 2), 1.4 * Math.sin((k / 8) * Math.PI * 2)]), radius: 0.25 },
    note: '8 つの山。取りこぼしが起きやすく、モデルの差が出やすい定番の例。',
    sample(r) {
      const k = Math.floor(r() * 8);
      const t = (k / 8) * Math.PI * 2;
      return [1.4 * Math.cos(t) + 0.06 * gaussian(r), 1.4 * Math.sin(t) + 0.06 * gaussian(r)];
    },
  },
  grid25: {
    label: '5×5 格子のガウス',
    modes: { centers: Array.from({ length: 25 }, (_, k) => [(Math.floor(k / 5) - 2) * 0.7, ((k % 5) - 2) * 0.7]), radius: 0.15 },
    note: '25 個の山。すべての山を覆えるかが試される難しめの課題。',
    sample(r) {
      const i = Math.floor(r() * 5) - 2, j = Math.floor(r() * 5) - 2;
      return [i * 0.7 + 0.04 * gaussian(r), j * 0.7 + 0.04 * gaussian(r)];
    },
  },
  circle: {
    label: '円（リング）',
    note: '連続した1次元の構造。潜在空間がどう折り畳まれるかに注目。',
    sample(r) {
      const t = r() * Math.PI * 2;
      const rad = 1.2 + 0.05 * gaussian(r);
      return [rad * Math.cos(t), rad * Math.sin(t)];
    },
  },
  moons: {
    label: '2つの三日月',
    note: '分離した2つの曲がった形。',
    sample(r) {
      const t = r() * Math.PI;
      const n = [0.06 * gaussian(r), 0.06 * gaussian(r)];
      if (r() < 0.5) return [Math.cos(t) - 0.5 + n[0], Math.sin(t) - 0.25 + n[1]];
      return [1 - Math.cos(t) - 0.5 + n[0], 0.25 - Math.sin(t) + n[1]];
    },
  },
  spiral: {
    label: '渦巻き',
    note: '長く曲がりくねった構造。大きめのネットワークと長い学習が必要。',
    sample(r) {
      const t = Math.sqrt(r()) * 3 * Math.PI;
      const rad = (t / (3 * Math.PI)) * 1.7;
      return [rad * Math.cos(t) + 0.04 * gaussian(r), rad * Math.sin(t) + 0.04 * gaussian(r)];
    },
  },
};

// 学習パス：0（はじめて触る）から実用までの順路。
// 各ステップは「なぜやるか → どこを触るか → どう確かめるか」の3点セット。
export const PATH = [
  // ---- 第0層 ----
  {
    id: 'l0-feel', layer: 0, title: 'まず1つ、学習が起きる様子を見る',
    why: '数式の前に「勝手にうまくなっていく」現象を目で見ておくと、あとの説明が全部それの説明として聞ける。',
    links: [{ t: 'GAN のガイドを最後まで', h: 'gan-guide.html' }],
    check: '「モード崩壊」と「勾配消失」が、どう見えたかを一言で言える。',
    terms: ['generator', 'discriminator', 'grad'],
  },
  {
    id: 'l0-vector', layer: 0, title: 'ベクトルと内積、コサイン類似度',
    why: '「似ている」を数で表す方法。RAG も埋め込みも、ここが分かれば半分は分かる。',
    links: [{ t: 'ドリル：コサイン類似度', h: 'drills.html' }, { t: 'RAG の地図を触る', h: 'rag.html' }],
    check: '長さの違う2つのベクトルでも、向きが同じなら cos = 1 になる理由を説明できる。',
    terms: ['vector', 'dot', 'cosine'],
  },
  {
    id: 'l0-prob', layer: 0, title: '確率分布・期待値・対数',
    why: '生成モデルの式は、ほぼ全部この3つの組み合わせでできている。',
    links: [{ t: '用語辞典の第0層', h: 'glossary.html' }, { t: 'ドリル：交差エントロピー', h: 'drills.html' }],
    check: '「確率の掛け算が log で足し算になる」ことを式で書ける。',
    terms: ['prob', 'expect', 'log', 'gaussian'],
  },
  {
    id: 'l0-symbol', layer: 0, title: '記号を声に出して読めるようにする',
    why: '式が読めない原因のほとんどは記号。読み方さえ分かれば、意味は推測できる。',
    links: [{ t: '記号の読み方', h: 'glossary.html#symbols' }],
    check: '𝔼_{z～N(0,I)}[log D(G(z))] を日本語で読み下せる。',
    terms: ['expect', 'prob'],
  },
  // ---- 第1層 ----
  {
    id: 'l1-loss', layer: 1, title: '損失・勾配・学習率の三角関係',
    why: 'どのモデルでも学習の骨格は同じ。「損失を決めて、勾配で下る」だけ。',
    links: [{ t: 'GAN の詳細で学習率を変える', h: 'gan.html' }, { t: 'ドリル：GAN の損失', h: 'drills.html' }],
    check: '学習率を上げると何が起きるかを、自分の言葉で説明できる。',
    terms: ['loss', 'sgd', 'lr', 'grad'],
  },
  {
    id: 'l1-net', layer: 1, title: 'ニューラルネットと順伝播・逆伝播',
    why: '「掛けて足して曲げる」を重ねただけ、と分かると怖くなくなる。',
    links: [{ t: 'GAN の詳細でネットワークを小さくする', h: 'gan.html' }, { t: '用語辞典の第1層', h: 'glossary.html' }],
    check: '隠れ層を 1 層 16 ユニットにすると渦巻きが真似できない理由を説明できる。',
    terms: ['mlp', 'forward', 'backward', 'param'],
  },
  {
    id: 'l1-softmax', layer: 1, title: 'softmax と交差エントロピー',
    why: '分類も言語モデルも、出力はこの2つ。実務で最初に書くコードもここ。',
    links: [{ t: 'ドリル：softmax', h: 'drills.html' }, { t: 'Transformer の生成パネル', h: 'ar.html' }],
    check: 'softmax の出力を足すと必ず 1 になる理由を説明できる。',
    terms: ['softmax', 'ce', 'nll'],
  },
  // ---- 第2層 ----
  {
    id: 'l2-gan', layer: 2, title: 'GAN：2人の目的関数',
    why: 'D と G が「別々の損失」を持つ点が、他のモデルと決定的に違う。',
    links: [{ t: 'GAN の詳細で式を見る', h: 'gan.html' }],
    check: '非飽和損失とミニマックス損失の違いと、なぜ前者が標準かを言える。',
    terms: ['adversarial', 'nonsat', 'modecollapse', 'vanish'],
  },
  {
    id: 'l2-vae', layer: 2, title: 'VAE：ELBO と再パラメータ化',
    why: '「復元」と「潜在空間の整理」の綱引きという構図が、拡散モデルにもつながる。',
    links: [{ t: 'VAE のガイド', h: 'vae-guide.html' }, { t: 'ドリル：KL', h: 'drills.html' }],
    check: 'β を 0 にしたときと大きくしたとき、それぞれ何が壊れるかを説明できる。',
    terms: ['latent', 'encoder', 'decoder', 'reparam', 'elbo', 'kl', 'collapse', 'beta'],
  },
  {
    id: 'l2-tf', layer: 2, title: 'Transformer：トークンとアテンション',
    why: '今の LLM の土台。「次を当てる」だけで文章が書ける、という感覚をつかむ。',
    links: [{ t: 'Transformer の詳細', h: 'ar.html' }],
    check: '因果マスクがないと何が困るかを説明できる。',
    terms: ['token', 'embedding', 'attention', 'causal', 'autoregressive', 'layernorm', 'residual'],
  },
  {
    id: 'l2-rag', layer: 2, title: 'RAG：検索とプロンプト',
    why: '実務でいちばん出番が多い。しかも失敗の原因は前半の検索にあることが多い。',
    links: [{ t: 'RAG のガイド', h: 'rag-guide.html' }, { t: 'RAG の詳細でしきい値を動かす', h: 'rag.html' }],
    check: 'k・しきい値・チャンクの大きさを変えると何が変わるかを説明できる。',
    terms: ['rag', 'chunk', 'tfidf', 'topk', 'grounding', 'halluc', 'embedding'],
  },
  {
    id: 'l2-compare', layer: 2, title: '3つのモデルを比べる',
    why: 'それぞれの得意・不得意を1枚で見ると、選ぶ基準ができる。',
    links: [{ t: '比較ページ', h: 'compare.html' }],
    check: '品質と網羅性の違いと、「ばらまくと網羅性が高く出る」落とし穴を説明できる。',
    terms: ['precision', 'recall'],
  },
  // ---- 第3層 ----
  {
    id: 'l3-code', layer: 3, title: 'コードで書けるようにする',
    why: '画面で見た式が、PyTorch の数行と同じものだと分かると、実装の壁が下がる。',
    links: [{ t: '各ページの「この画面を動かしているコード」', h: 'gan.html' }],
    check: 'GAN の学習ループを、D の更新と G の更新に分けて擬似コードで書ける。',
    terms: ['adam', 'batch'],
  },
  {
    id: 'l3-debug', layer: 3, title: '壊れ方を知る',
    why: '実務では「うまくいかないとき、どこを疑うか」が仕事のほとんど。',
    links: [{ t: 'GAN の失敗例', h: 'gan-guide.html' }, { t: 'VAE の失敗例', h: 'vae-guide.html' }],
    check: '損失が下がらないとき、学習率・初期化・データ・実装のどれから疑うか順番を言える。',
    terms: ['modecollapse', 'vanish', 'collapse', 'overfit'],
  },
  {
    id: 'l3-eval', layer: 3, title: '評価の考え方を持つ',
    why: '「良くなった」を数字で言えないと、改善も説明もできない。',
    links: [{ t: '比較ページの指標', h: 'compare.html' }],
    check: '1つの指標だけを見てはいけない理由を、具体例で説明できる。',
    terms: ['precision', 'recall', 'nll'],
  },
  {
    id: 'l3-next', layer: 3, title: '実データへ出る',
    why: 'ここから先はこのアプリの外。PyTorch のチュートリアルや、小さな自作データセットで同じことをやる。',
    links: [{ t: 'ホームに戻る', h: 'index.html' }],
    check: 'MNIST か手元のデータで、VAE か小さな Transformer を1本動かせた。',
    terms: ['vectordb'],
  },
];

export const PATH_LAYERS = [
  { id: 0, label: '第0層：道具としての数学', note: '式を読むための最低限。ここを飛ばすと、あとで必ず戻ってくることになる。' },
  { id: 1, label: '第1層：共通の骨格', note: 'どのモデルにも共通する部分。ここが9割。' },
  { id: 2, label: '第2層：モデル固有', note: '各モデルの「その1点だけが違う」ところ。' },
  { id: 3, label: '第3層：評価と実用', note: '動かす・直す・測る。実務はここから。' },
];

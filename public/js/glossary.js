// 用語カードの辞書。各用語は「意味 → たとえ → 記号 → 式」の4点セットで覚える。
//   ja:     日本語の見出し（本文中でこの文字列を見つけてカードにする）
//   alias:  同じ意味で本文に出てくる別の書き方
//   en:     英語表記（論文やライブラリで出会う形）
//   layer:  0=道具としての数学 / 1=共通の骨格 / 2=モデル固有 / 3=実用
//   tags:   関係するモデル
//   short:  一言の意味
//   story:  このアプリでのたとえ
//   symbol: 記号と読み方
//   formula:式（あれば）
//   where:  このアプリのどこで見られるか（{text, href}）

export const TERMS = [
  // ---- 第0層：道具としての数学 ----
  {
    id: 'vector', ja: 'ベクトル', en: 'vector', layer: 0, tags: ['共通'],
    short: '数を順番に並べたもの。位置や特徴を表す。',
    story: 'このアプリの点はすべて2つの数の組（x₁, x₂）＝ 2次元ベクトル。',
    symbol: '太字の x、または x ∈ ℝ²（「エックスは2次元の実数ベクトル」）',
    where: { text: 'GAN のデータ空間', href: 'gan.html' },
  },
  {
    id: 'dot', ja: '内積', alias: ['ドット積'], en: 'dot product / inner product', layer: 0, tags: ['共通', 'RAG'],
    short: '2つのベクトルの「向きの似ている度合い」を測る掛け算。',
    story: 'RAG の類似度は、質問と資料のベクトルの内積そのもの。',
    symbol: 'a · b、⟨a, b⟩（「エーとビーの内積」）',
    formula: 'a · b = a₁b₁ + a₂b₂ + …',
    where: { text: 'RAG の類似度', href: 'rag.html' },
  },
  {
    id: 'cosine', ja: 'コサイン類似度', en: 'cosine similarity', layer: 0, tags: ['RAG'],
    short: '長さの影響を除いた内積。向きだけを比べる。1 に近いほど似ている。',
    story: '長い文書ほど有利にならないように、長さで割ってから比べている。',
    symbol: 'cos(q, d)',
    formula: 'cos(q, d) = (q · d) / (‖q‖ ‖d‖)',
    where: { text: 'RAG の検索結果', href: 'rag.html' },
  },
  {
    id: 'prob', ja: '確率分布', alias: ['分布'], en: 'probability distribution', layer: 0, tags: ['共通'],
    short: '「どこがどれくらい出やすいか」を表す地図。全部足すと 1 になる。',
    story: '青い点のばらつき方そのもの。生成モデルはこの地図を真似しようとしている。',
    symbol: 'p(x)（「ピー・オブ・エックス」）、x ～ p（「エックスはピーに従う」）',
    where: { text: 'Transformer の p(x)', href: 'ar.html' },
  },
  {
    id: 'expect', ja: '期待値', en: 'expectation', layer: 0, tags: ['共通'],
    short: '確率で重みをつけた平均。「たくさん試したときの平均」。',
    story: 'バッチ 64 個の平均を取っているところが、式の期待値にあたる。',
    symbol: '𝔼[·]（「イー、期待値」）、𝔼_{x～p}[f(x)]',
    formula: '𝔼[f(x)] = Σ p(x) f(x)',
  },
  {
    id: 'log', ja: '対数', en: 'logarithm', layer: 0, tags: ['共通'],
    short: '掛け算を足し算に変える道具。確率の計算で必ず出てくる。',
    story: '4つのトークンの確率の掛け算が、log を取ると足し算になる。桁が小さくなりすぎるのも防げる。',
    symbol: 'log p（自然対数。機械学習で log と書いたらたいてい ln）',
    formula: 'log(ab) = log a + log b',
    where: { text: 'Transformer の NLL', href: 'ar.html' },
  },
  {
    id: 'grad', ja: '勾配', alias: ['こうばい'], en: 'gradient', layer: 0, tags: ['共通', 'GAN'],
    short: '「どちらへ動かせば損失が増えるか」を指す矢印。逆向きに進めば減る。',
    story: 'GAN の白い矢印そのもの。スイカ割りの「右！もっと前！」の声。',
    symbol: '∇（ナブラ）、∂L/∂θ（「ラージエルをシータで偏微分」）',
    where: { text: 'GAN の矢印', href: 'gan-guide.html' },
  },
  {
    id: 'gaussian', ja: 'ガウス分布', alias: ['正規分布'], en: 'Gaussian / normal distribution', layer: 0, tags: ['共通', 'VAE'],
    short: '真ん中が高く、左右に裾が広がる、いちばんよく使う分布。',
    story: 'VAE のメモ帳の破線の円が、標準正規分布 N(0, I) の目安。',
    symbol: 'N(μ, σ²)（「エヌ、平均ミュー、分散シグマ二乗」）',
    where: { text: 'VAE の潜在空間', href: 'vae.html' },
  },

  // ---- 第1層：共通の骨格 ----
  {
    id: 'param', ja: 'パラメータ', alias: ['重み'], en: 'parameter / weight', layer: 1, tags: ['共通'],
    short: 'モデルが学習で調整する数。これが変わることを「学習した」という。',
    story: '画面の「パラメータ数」がその個数。GAN の職人は約 2,000 個の数を調整している。',
    symbol: 'θ（シータ）、W（重み行列）、b（バイアス）',
  },
  {
    id: 'loss', ja: '損失', alias: ['損失関数'], en: 'loss / objective', layer: 1, tags: ['共通'],
    short: '「どれくらい下手か」を1つの数にしたもの。これを小さくするのが学習。',
    story: '各ページの損失グラフ。GAN だけは2人ぶんあり、下がり続けないのが特徴。',
    symbol: 'L、J、ℓ',
  },
  {
    id: 'sgd', ja: '勾配降下法', alias: ['勾配降下'], en: 'gradient descent', layer: 1, tags: ['共通'],
    short: '勾配の逆向きに少しずつパラメータを動かして、損失を減らす方法。',
    story: '「修行1回」がこの1ステップ。目隠しで坂を下るのに似ている。',
    symbol: 'θ ← θ − η ∇L',
    formula: 'θ_{t+1} = θ_t − η ∇L(θ_t)',
  },
  {
    id: 'lr', ja: '学習率', en: 'learning rate', layer: 1, tags: ['共通'],
    short: '1回で動かす量。大きすぎると行き過ぎ、小さすぎると進まない。',
    story: 'GAN の「せっかちな職人」＝学習率が大きい状態。モード崩壊の引き金になる。',
    symbol: 'η（イータ）、α（アルファ）、コードでは lr',
    where: { text: 'GAN で学習率を上げる', href: 'gan.html' },
  },
  {
    id: 'batch', ja: 'バッチ', alias: ['バッチサイズ', 'ミニバッチ'], en: 'batch / mini-batch', layer: 1, tags: ['共通'],
    short: '1回の更新に使うデータの数。多いほど滑らかで、遅い。',
    story: 'このアプリの既定は 64。毎ステップ 64 点を引き直している。',
    symbol: 'N、B',
  },
  {
    id: 'forward', ja: '順伝播', en: 'forward pass', layer: 1, tags: ['共通'],
    short: '入力を層に通して出力を出す計算。',
    story: '点 → 隠れ層 → 出力、と左から右に進む流れ。',
    symbol: 'h = f(Wx + b)',
  },
  {
    id: 'backward', ja: '逆伝播', alias: ['誤差逆伝播'], en: 'backpropagation', layer: 1, tags: ['共通'],
    short: '出力側の誤差を入力側へ戻しながら、各パラメータの勾配を求める計算。',
    story: 'このアプリは全部手書きで実装していて、数値微分と一致することをテストしている。',
    symbol: '連鎖律 ∂L/∂x = ∂L/∂y · ∂y/∂x',
  },
  {
    id: 'adam', ja: 'Adam', en: 'Adam optimizer', layer: 1, tags: ['共通'],
    short: '勾配の平均と大きさを覚えておいて、進む量を自動で調整する最適化の方法。',
    story: 'このアプリの全モデルが Adam を使っている。GAN は β1 = 0.5 が定番。',
    symbol: 'β₁, β₂（モーメントの減衰率）',
  },
  {
    id: 'mlp', ja: 'ニューラルネット', alias: ['MLP', '多層パーセプトロン'], en: 'neural network / MLP', layer: 1, tags: ['共通'],
    short: '「掛けて足して曲げる」を何段も重ねた関数。曲げる部分が活性化関数。',
    story: 'GAN も VAE もこのアプリでは 2〜3 層の小さな MLP。',
    symbol: 'f(x) = W₂ σ(W₁x + b₁) + b₂',
  },
  {
    id: 'softmax', ja: 'ソフトマックス', alias: ['softmax'], en: 'softmax', layer: 1, tags: ['Transformer'],
    short: '好き勝手な数の列を、合計 1 の確率に変える関数。',
    story: 'Transformer が次のトークンを選ぶときの棒グラフは、softmax の出力。',
    symbol: 'softmax(z)ᵢ',
    formula: 'softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)',
    where: { text: 'ドリルで手計算する', href: 'drills.html' },
  },
  {
    id: 'ce', ja: '交差エントロピー', en: 'cross entropy', layer: 1, tags: ['Transformer'],
    short: '「正解にどれだけ確率を割り当てたか」の罰点。正解の確率が高いほど小さい。',
    story: 'Transformer の NLL グラフがこれ。点線の下限がデータのエントロピー。',
    symbol: 'H(p, q)',
    formula: 'L = −log q(正解)',
    where: { text: 'ドリルで手計算する', href: 'drills.html' },
  },
  {
    id: 'kl', ja: 'KL ダイバージェンス', alias: ['KL', 'KL 項'], en: 'KL divergence', layer: 1, tags: ['VAE'],
    short: '2つの分布のズレの大きさ。0 なら完全に一致。距離ではない（向きで値が変わる）。',
    story: 'VAE でメモの楕円を標準の円へそろえる力。大きくしすぎると事後崩壊。',
    symbol: 'KL(q ‖ p)（「キューからピーへのKL」）',
    formula: 'KL(q‖p) = Σ q log(q/p)',
    where: { text: 'ドリルで手計算する', href: 'drills.html' },
  },
  {
    id: 'overfit', ja: '過学習', en: 'overfitting', layer: 1, tags: ['共通'],
    short: '訓練データを覚えすぎて、新しいデータで外すこと。',
    story: 'このアプリは分布から毎回サンプルし直すので過学習が起きにくい。実務では必ず向き合う。',
  },

  // ---- 第2層：GAN ----
  {
    id: 'generator', ja: 'Generator', alias: ['生成器', '偽札職人'], en: 'generator (G)', layer: 2, tags: ['GAN'],
    short: 'ノイズから偽物を作るネットワーク。本物のデータは直接見ない。',
    story: '偽札職人。警察の反応だけを頼りに腕を上げる。',
    symbol: 'G(z)',
    where: { text: 'GAN のガイド', href: 'gan-guide.html' },
  },
  {
    id: 'discriminator', ja: 'Discriminator', alias: ['識別器', '警察'], en: 'discriminator (D)', layer: 2, tags: ['GAN'],
    short: '本物か偽物かを当てるネットワーク。出力はその確率。',
    story: '警察。背景の色が、この警察の判定地図。',
    symbol: 'D(x) ∈ [0, 1]',
    where: { text: 'GAN の背景', href: 'gan.html' },
  },
  {
    id: 'adversarial', ja: '敵対的学習', en: 'adversarial training', layer: 2, tags: ['GAN'],
    short: '2つのネットワークを競わせて、片方を上達させる学習の枠組み。',
    story: 'いたちごっこ。GAN の A（Adversarial）はここ。',
    symbol: 'min_G max_D V(D, G)',
    formula: 'V(D,G) = 𝔼_{x～data}[log D(x)] + 𝔼_{z}[log(1 − D(G(z)))]',
  },
  {
    id: 'nonsat', ja: '非飽和損失', en: 'non-saturating loss', layer: 2, tags: ['GAN'],
    short: 'G が「本物らしさ」を直接上げにいく損失。原論文の式より勾配が消えにくい。',
    story: 'ミニマックスに切り替えると、生成点が画面外へ飛ぶ様子が見られる。',
    formula: 'L_G = −log D(G(z))',
    where: { text: 'GAN で損失を切り替える', href: 'gan.html' },
  },
  {
    id: 'modecollapse', ja: 'モード崩壊', en: 'mode collapse', layer: 2, tags: ['GAN'],
    short: 'G が一部の種類しか作らなくなる失敗。',
    story: '一発屋の芸人。持ちネタを1つずつ乗り換えていく。',
    where: { text: 'ガイドの失敗その1', href: 'gan-guide.html' },
  },
  {
    id: 'vanish', ja: '勾配消失', en: 'vanishing gradient', layer: 2, tags: ['GAN'],
    short: '勾配がほぼ 0 になり、学習が進まなくなること。',
    story: '「全部ダメ」としか言わない先生。何を直せばいいか分からない。',
    where: { text: 'ガイドの失敗その2', href: 'gan-guide.html' },
  },

  // ---- 第2層：VAE ----
  {
    id: 'latent', ja: '潜在変数', alias: ['潜在空間', 'メモ'], en: 'latent variable / latent space', layer: 2, tags: ['VAE', 'GAN'],
    short: 'データの裏にある、短い数の組。ここから元のデータを作る。',
    story: 'VAE の「メモ」、GAN の「サイコロの目」。',
    symbol: 'z',
    where: { text: 'VAE の潜在空間', href: 'vae.html' },
  },
  {
    id: 'encoder', ja: 'エンコーダ', en: 'encoder', layer: 2, tags: ['VAE'],
    short: 'データを潜在変数の分布（中心 μ と広がり σ）に変えるネットワーク。',
    story: '絵を見てメモを書く人。点ではなく楕円を書くのがポイント。',
    symbol: 'q(z|x)（「エックスが与えられたときのゼットの分布」）',
  },
  {
    id: 'decoder', ja: 'デコーダ', en: 'decoder', layer: 2, tags: ['VAE', 'Transformer'],
    short: '潜在変数からデータを作り直すネットワーク。GAN の Generator と同じ役。',
    story: 'メモを見て描き直す人。',
    symbol: 'p(x|z)',
  },
  {
    id: 'reparam', ja: '再パラメータ化', en: 'reparameterization trick', layer: 2, tags: ['VAE'],
    short: 'ランダムに引く操作を「決まった計算 ＋ 外から来た乱数」に書き換えて、勾配を通せるようにする工夫。',
    story: 'くじ引きの部分を外に出すと、残りは普通の計算なので微分できる。',
    formula: 'z = μ + σ · ε,  ε ～ N(0, I)',
  },
  {
    id: 'elbo', ja: 'ELBO', alias: ['変分下界'], en: 'evidence lower bound', layer: 2, tags: ['VAE'],
    short: 'VAE が最大化する目標。「復元の良さ」から「KL」を引いたもの。',
    story: '画面の「2つの損失の綱引き」がこれ。符号を反転して損失として最小化している。',
    formula: 'ELBO = 𝔼_q[log p(x|z)] − KL(q(z|x) ‖ p(z))',
  },
  {
    id: 'collapse', ja: '事後崩壊', en: 'posterior collapse', layer: 2, tags: ['VAE'],
    short: 'エンコーダが情報を捨て、デコーダが z を無視してしまう失敗。',
    story: 'メモが空っぽ。楕円が全部、標準の円に重なる。',
    where: { text: 'VAE ガイドの失敗その2', href: 'vae-guide.html' },
  },
  {
    id: 'beta', ja: 'β', alias: ['ベータ'], en: 'beta (β-VAE)', layer: 2, tags: ['VAE'],
    short: 'KL 項の重み。復元を優先するか、潜在空間の整理を優先するかのつまみ。',
    story: '0 にするとただのオートエンコーダ、大きくすると事後崩壊。',
    where: { text: 'VAE で β を動かす', href: 'vae.html' },
  },

  // ---- 第2層：Transformer ----
  {
    id: 'token', ja: 'トークン', en: 'token', layer: 2, tags: ['Transformer'],
    short: 'モデルが扱う最小の単位。文章なら語や文字の断片。',
    story: 'このアプリでは、1点を「x₁粗・x₂粗・x₁細・x₂細」の4トークンにしている。',
    where: { text: 'Transformer の生成', href: 'ar.html' },
  },
  {
    id: 'embedding', ja: '埋め込み', en: 'embedding', layer: 2, tags: ['Transformer', 'RAG'],
    short: 'トークンや文章を、意味が近いほど近くなるベクトルに変えること。',
    story: 'RAG の「資料の地図」は、この埋め込みを2次元に押しつぶしたもの。',
    symbol: 'E ∈ ℝ^{語彙数 × d}',
  },
  {
    id: 'attention', ja: 'アテンション', alias: ['注意機構', '自己注意'], en: 'attention / self-attention', layer: 2, tags: ['Transformer'],
    short: '次を予測するために、今までのどの位置をどれだけ見るかを決める仕組み。',
    story: '蛍光ペン。どの入力に線を引いて読むかの割合が、アテンションの表の数字。',
    symbol: 'Q, K, V（クエリ・キー・バリュー）',
    formula: 'Attention(Q,K,V) = softmax(QKᵀ/√d) V',
    where: { text: 'Transformer のアテンション', href: 'ar.html' },
  },
  {
    id: 'causal', ja: '因果マスク', en: 'causal mask', layer: 2, tags: ['Transformer'],
    short: '未来のトークンを見られないようにする覆い。自己回帰の生成に必須。',
    story: 'アテンションの表の右上が空白なのはこのため。',
  },
  {
    id: 'autoregressive', ja: '自己回帰', en: 'autoregressive', layer: 2, tags: ['Transformer'],
    short: '前に決めたものを見ながら、次を1つずつ決めていく作り方。',
    story: '20の質問。「右半分？」「上のほう？」と絞り込む。',
    formula: 'p(x) = p(t₁) p(t₂|t₁) p(t₃|t₁,t₂) …',
  },
  {
    id: 'nll', ja: '負の対数尤度', alias: ['NLL'], en: 'negative log-likelihood', layer: 2, tags: ['Transformer'],
    short: 'モデルが正解にどれだけ低い確率しか与えられなかったかの指標。小さいほど良い。',
    story: '天気予報の成績。外すほど大きな罰点。',
    formula: 'NLL = −log p(データ)',
  },
  {
    id: 'layernorm', ja: 'LayerNorm', alias: ['層正規化'], en: 'layer normalization', layer: 2, tags: ['Transformer'],
    short: '各層の出力の大きさをそろえて、学習を安定させる処理。',
    symbol: 'LN(x) = γ (x − μ)/σ + β',
  },
  {
    id: 'residual', ja: '残差接続', en: 'residual connection', layer: 2, tags: ['Transformer'],
    short: '層の出力に入力を足し戻す配線。深くしても勾配が届く。',
    symbol: 'x + F(x)',
  },

  // ---- 第2層：RAG ----
  {
    id: 'rag', ja: 'RAG', en: 'retrieval-augmented generation', layer: 2, tags: ['RAG'],
    short: '答える前に資料を検索し、それを読ませて生成する仕組み。',
    story: '図書館の司書（検索）と作家（LLM）の分業。',
    where: { text: 'RAG のガイド', href: 'rag-guide.html' },
  },
  {
    id: 'chunk', ja: 'チャンク', alias: ['チャンク化'], en: 'chunk / chunking', layer: 2, tags: ['RAG'],
    short: '長い文書を検索しやすい大きさに切り分けること、またその断片。',
    story: '大きすぎると余計な話が混ざり、小さすぎると前後の条件が切れる。',
    where: { text: 'RAG で分け方を変える', href: 'rag.html' },
  },
  {
    id: 'tfidf', ja: 'TF-IDF', en: 'TF-IDF', layer: 2, tags: ['RAG'],
    short: 'よく出る語は軽く、珍しい語は重く数える重み付け。',
    story: '「の」「こと」より「宿泊費」を重く見るということ。',
    formula: 'w = tf × log(N / df)',
  },
  {
    id: 'topk', ja: 'top-k 検索', alias: ['top-k'], en: 'top-k retrieval', layer: 2, tags: ['RAG'],
    short: '似ている順に上から k 件だけ取ること。',
    story: 'k を増やすと拾いこぼしは減るが、関係ない資料も混ざる。',
  },
  {
    id: 'grounding', ja: 'グラウンディング', en: 'grounding', layer: 2, tags: ['RAG'],
    short: '「渡した資料だけを根拠に答えよ」と縛ること。',
    story: 'プロンプトの1行目がその指示になっている。',
  },
  {
    id: 'halluc', ja: 'ハルシネーション', en: 'hallucination', layer: 2, tags: ['RAG'],
    short: 'モデルが、もっともらしいが根拠のない内容を作ってしまうこと。',
    story: '資料にない質問に無理に答えさせると起きる。しきい値で防ぐ。',
    where: { text: 'RAG ガイドの失敗その1', href: 'rag-guide.html' },
  },
  {
    id: 'vectordb', ja: 'ベクトル検索', alias: ['ベクトルデータベース'], en: 'vector search / vector database', layer: 3, tags: ['RAG'],
    short: '埋め込みベクトルを大量に保存し、近いものを高速に探す仕組み。',
    story: 'このアプリは 24 件なので総当たり。実務では近似最近傍探索を使う。',
  },

  // ---- 第1〜3層：評価 ----
  {
    id: 'precision', ja: '品質', alias: ['precision', 'プレシジョン'], en: 'precision', layer: 3, tags: ['共通'],
    short: '生成したもののうち、本物らしいものの割合。',
    story: '漁の網でいうと「獲れたもののうち狙った魚の割合」。',
    where: { text: '比較ページ', href: 'compare.html' },
  },
  {
    id: 'recall', ja: '網羅性', alias: ['recall', 'リコール'], en: 'recall', layer: 3, tags: ['共通'],
    short: '本物のうち、モデルが作れている範囲の割合。',
    story: '「海にいる魚のうち獲れた割合」。広くばらまくと高く出てしまうので注意。',
    where: { text: '比較ページ', href: 'compare.html' },
  },
];

/** 記号の読み方 */
export const SYMBOLS = [
  { sym: 'θ', read: 'シータ', mean: 'モデルのパラメータ全体', ex: '∇L(θ)＝パラメータについての勾配' },
  { sym: 'η, α', read: 'イータ / アルファ', mean: '学習率', ex: 'θ ← θ − η∇L' },
  { sym: '∇', read: 'ナブラ', mean: '勾配（全部の偏微分を並べたもの）', ex: '∇L＝損失の勾配' },
  { sym: '∂', read: 'ラウンドディー / パーシャル', mean: '偏微分', ex: '∂L/∂θ＝「エルをシータで偏微分」' },
  { sym: 'Σ', read: 'シグマ', mean: '合計', ex: 'Σᵢ xᵢ＝全部の x を足す' },
  { sym: 'Π', read: 'パイ', mean: '掛け算の合計版', ex: 'Π p(tᵢ)＝確率を全部掛ける' },
  { sym: '𝔼[·]', read: 'イー / 期待値', mean: '平均（確率で重みづけ）', ex: '𝔼_{x～p}[f(x)]＝「p から引いた x での f の平均」' },
  { sym: '～', read: 'チルダ / 従う', mean: 'その分布から引く', ex: 'z ～ N(0, I)' },
  { sym: 'p(x|z)', read: 'ピー・エックス・ギブン・ゼット', mean: 'z が分かっているときの x の確率', ex: 'VAE のデコーダ' },
  { sym: '‖x‖', read: 'ノルム', mean: 'ベクトルの長さ', ex: 'cos の分母' },
  { sym: 'KL(q‖p)', read: 'ケーエル、キュー・パラレル・ピー', mean: 'q から見た p とのズレ', ex: 'VAE の KL 項' },
  { sym: 'argmax', read: 'アーグマックス', mean: '最大にする「引数」を返す', ex: 'argmax_x f(x)＝f を最大にする x' },
  { sym: 'ε', read: 'イプシロン', mean: '小さな乱数、または小さな数', ex: 'z = μ + σ·ε' },
  { sym: '∈ ℝᵈ', read: 'エレメント・オブ・アールディー', mean: 'd 次元の実数ベクトルである', ex: 'x ∈ ℝ²' },
  { sym: 'ᵀ', read: 'てんち / トランスポーズ', mean: '転置（行と列の入れ替え）', ex: 'QKᵀ' },
  { sym: '∝', read: '比例する', mean: '定数倍を無視して等しい', ex: 'p(z|x) ∝ p(x|z)p(z)' },
];

export const LAYERS = [
  { id: 0, label: '第0層：道具としての数学' },
  { id: 1, label: '第1層：共通の骨格' },
  { id: 2, label: '第2層：モデル固有' },
  { id: 3, label: '第3層：評価と実用' },
];

export const TERM_BY_ID = new Map(TERMS.map((t) => [t.id, t]));

/** 本文に出てくる見出し語（長いものから順に探す） */
const LOOKUP = (() => {
  const pairs = [];
  for (const t of TERMS) {
    pairs.push([t.ja, t]);
    for (const a of t.alias || []) pairs.push([a, t]);
  }
  pairs.sort((a, b) => b[0].length - a[0].length);
  return pairs;
})();

let card;
function ensureCard() {
  if (card) return card;
  card = document.createElement('div');
  card.className = 'term-card';
  card.hidden = true;
  document.body.appendChild(card);
  document.addEventListener('click', (e) => {
    if (!card.contains(e.target) && !e.target.classList.contains('term-link')) card.hidden = true;
  });
  window.addEventListener('scroll', () => { card.hidden = true; }, { passive: true });
  return card;
}

export function showTermCard(id, anchor) {
  const t = TERM_BY_ID.get(id);
  if (!t) return;
  const c = ensureCard();
  c.innerHTML = `
    <div class="tc-head"><b>${t.ja}</b><span>${t.en}</span></div>
    <div class="tc-row"><span class="tc-k">意味</span><span>${t.short}</span></div>
    ${t.story ? `<div class="tc-row"><span class="tc-k">たとえ</span><span>${t.story}</span></div>` : ''}
    ${t.symbol ? `<div class="tc-row"><span class="tc-k">記号</span><span class="mono">${t.symbol}</span></div>` : ''}
    ${t.formula ? `<div class="tc-row"><span class="tc-k">式</span><span class="mono">${t.formula}</span></div>` : ''}
    ${t.where ? `<div class="tc-row"><span class="tc-k">見る</span><a href="${t.where.href}">${t.where.text} →</a></div>` : ''}
    <div class="tc-foot"><a href="glossary.html#${t.id}">用語辞典で見る →</a></div>`;
  c.hidden = false;
  const r = anchor.getBoundingClientRect();
  const w = c.offsetWidth;
  c.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left))}px`;
  c.style.top = `${r.bottom + 6}px`;
}

const SKIP = new Set(['SCRIPT', 'STYLE', 'PRE', 'CODE', 'TEXTAREA', 'INPUT', 'SELECT', 'OPTION', 'CANVAS', 'BUTTON', 'A', 'H1']);

/**
 * 指定した範囲の本文から用語を見つけてカードにリンクする。
 * 同じ用語は1つの箱につき最初の1回だけリンクにする（読みにくくならないように）。
 */
export function attachGlossary(root = document.body) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      for (let p = node.parentElement; p; p = p.parentElement) {
        if (SKIP.has(p.tagName) || p.classList.contains('no-glossary') || p.classList.contains('term-card') || p.classList.contains('term-link')) {
          return NodeFilter.FILTER_REJECT;
        }
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  const used = new Set();
  for (const node of nodes) {
    let text = node.nodeValue;
    let hit = null, at = -1, label = '';
    for (const [word, term] of LOOKUP) {
      if (used.has(term.id)) continue;
      const i = text.indexOf(word);
      if (i >= 0 && (at < 0 || i < at)) { hit = term; at = i; label = word; }
    }
    if (!hit) continue;
    used.add(hit.id);
    const after = node.splitText(at);
    after.splitText(label.length);
    const link = document.createElement('span');
    link.className = 'term-link';
    link.textContent = label;
    link.tabIndex = 0;
    link.setAttribute('role', 'button');
    link.title = `${hit.ja}：${hit.short}`;
    link.addEventListener('click', (e) => { e.stopPropagation(); showTermCard(hit.id, link); });
    link.addEventListener('keydown', (e) => { if (e.key === 'Enter') showTermCard(hit.id, link); });
    after.parentNode.replaceChild(link, after);
  }
}

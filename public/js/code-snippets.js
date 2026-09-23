// 画面で見ていたものが、実務のコードのどの行にあたるか。
// py＝実務で書く Python（PyTorch）、js＝この画面の中で動いているコード（抜粋・参考）。

export const GAN_CODE = [
  {
    label: 'D の更新（警察の勉強）',
    file: 'public/js/gan.js の trainD()',
    js: `// 本物 → 1 に近づける
const lr_ = this.D.forward(xr, n);
for (let i = 0; i < n; i++) {
  const p = sigmoid(lr_[i]);
  gr[i] = (p - 1) / n;      // ∂L/∂logit
}
this.D.backward(gr);

// 偽物 → 0 に近づける
const xf = this.generate(this.sampleNoise(n), n);
const lf = this.D.forward(xf, n);
for (let i = 0; i < n; i++) {
  gf[i] = sigmoid(lf[i]) / n;
}
this.D.backward(gf);
this.D.adamStep(lrD);`,
    py: `opt_d.zero_grad()
d_real = D(x_real)
d_fake = D(G(z).detach())          # G には勾配を流さない

loss_d = bce(d_real, torch.ones_like(d_real)) \\
       + bce(d_fake, torch.zeros_like(d_fake))
loss_d.backward()
opt_d.step()

# bce = nn.BCEWithLogitsLoss()`,
    note: '手書きの (p − 1)/n は、BCEWithLogitsLoss の勾配そのもの。ライブラリは同じ計算をしている。',
  },
  {
    label: 'G の更新（職人の修行）',
    file: 'public/js/gan.js の trainG()',
    js: `const xg = this.G.forward(z, n);
const lg = this.D.forward(xg, n);
for (let i = 0; i < n; i++) {
  const p = sigmoid(lg[i]);
  dl[i] = (p - 1) / n;      // 非飽和損失
}
// D は通すが更新しない（第2引数 false）
const dx = this.D.backward(dl, false);
this.G.backward(dx);
this.G.adamStep(lrG);`,
    py: `opt_g.zero_grad()
d_fake = D(G(z))                   # detach しない
loss_g = bce(d_fake, torch.ones_like(d_fake))
loss_g.backward()
opt_g.step()                       # G のパラメータだけ更新`,
    note: 'D を通して勾配を G へ流すが、更新するのは G だけ。PyTorch では opt_g が G のパラメータしか持たないことで同じことをしている。',
  },
  {
    label: '勾配の矢印（D の入力に対する勾配）',
    file: 'public/js/gan.js の dGradient()',
    js: `this.D.forward(X, n);
const d = new Float64Array(n).fill(1);
return this.D.backward(d, false);  // 入力についての勾配`,
    py: `x = x.requires_grad_(True)
score = D(x).sum()
grad, = torch.autograd.grad(score, x)`,
    note: '「入力をどちらへ動かすとスコアが上がるか」。画面の白い矢印はこれ。',
  },
];

export const VAE_CODE = [
  {
    label: '再パラメータ化と損失',
    file: 'public/js/vae.js の trainStep()',
    js: `// エンコーダの出力を μ と logσ² に分ける
const mu = encOut[i * 2 * L + k];
const lv = encOut[i * 2 * L + L + k];
const s  = Math.exp(0.5 * lv);
Z[i * L + k] = mu + s * E[i * L + k];   // z = μ + σ·ε

// 損失
recon += 0.5 * inv * d * d / n;          // 再構成
kl    += 0.5 * (mu*mu + s*s - 1 - lv) / n;  // KL`,
    py: `mu, logvar = encoder(x).chunk(2, dim=-1)
std = torch.exp(0.5 * logvar)
z = mu + std * torch.randn_like(std)     # 再パラメータ化

x_hat = decoder(z)
recon = F.mse_loss(x_hat, x, reduction='sum') / (2 * sigma**2)
kl = 0.5 * (mu.pow(2) + std.pow(2) - 1 - logvar).sum()
loss = (recon + beta * kl) / x.size(0)`,
    note: 'randn_like が「外から来た乱数 ε」。これがあるおかげで、くじ引きを含む計算でも勾配が通る。',
  },
  {
    label: '新しい点を作る（事前分布からのサンプリング）',
    file: 'public/js/vae.js の sample()',
    js: `for (let i = 0; i < Z.length; i++) Z[i] = gaussian(rand);
return { Z, X: this.decode(Z, n) };`,
    py: `z = torch.randn(n, latent_dim)
x = decoder(z)`,
    note: '学習後はエンコーダを使わない。メモ帳の適当な場所を指さして描かせている部分。',
  },
];

export const AR_CODE = [
  {
    label: '因果的自己注意',
    file: 'public/js/transformer.js の forward()',
    js: `// 位置 i は 0..i だけを見る
for (let j = 0; j <= i; j++) {
  let s = 0;
  for (let e = 0; e < dh; e++) s += q[qi+e] * k[kj+e];
  att[ao + j] = s * scale;           // QKᵀ / √d
}
// softmax して V を重み付き平均
for (let j = 0; j <= i; j++) {
  ctx[qi+e] += att[ao+j] * v[vj+e];
}`,
    py: `import torch.nn.functional as F
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# または nn.MultiheadAttention(..., batch_first=True) に
# attn_mask=torch.triu(torch.ones(T, T), 1).bool() を渡す`,
    note: 'is_causal=True が、このアプリの「j <= i だけ見る」に対応する。',
  },
  {
    label: '学習（次のトークンを当てる）',
    file: 'public/js/ar.js の trainStep()',
    js: `inp[b*T + t] = t === 0 ? BOS : tk[t-1];
tgt[b*T + t] = tk[t];
...
const P = softmaxRows(logits, B*T, V);
perPos[i % T] -= Math.log(P[o + tgt[i]]) / B;   // NLL
d[o + j] = (P[o+j] - (j === tgt[i] ? 1 : 0)) / B;`,
    py: `logits = model(inp)                 # (B, T, V)
loss = F.cross_entropy(
    logits.view(-1, V), tgt.view(-1))
loss.backward()`,
    note: '(p − onehot) が交差エントロピーの勾配。cross_entropy はこれを内部でやっている。',
  },
];

export const RAG_CODE = [
  {
    label: '検索（ベクトル化と類似度）',
    file: 'public/js/rag.js の search()',
    js: `// 質問をベクトルに
const { vec } = this.queryVector(q);
// 各資料との内積（正規化済みなのでコサイン類似度）
for (const [t, w] of vec) {
  const w2 = dv.get(t);
  if (w2 !== undefined) score += w * w2;
}`,
    py: `from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-small')

doc_vecs = model.encode(docs, normalize_embeddings=True)
q_vec    = model.encode([question], normalize_embeddings=True)
scores   = q_vec @ doc_vecs.T          # コサイン類似度
top = scores[0].argsort()[::-1][:k]`,
    note: '違いは「ベクトルの作り方」だけ。このアプリは語の一致（TF-IDF）、実務は意味を捉える埋め込みモデル。',
  },
  {
    label: 'プロンプトの組み立て',
    file: 'public/js/rag.js の buildPrompt()',
    js: `return \`以下の資料だけを根拠に答えてください。
資料に書かれていないことは「記載がありません」と。

# 資料
\${docs}

# 質問
\${question}\`;`,
    py: `context = "\\n\\n".join(
    f"[{i+1}] {d['title']}\\n{d['text']}" for i, d in enumerate(picked))

messages = [
  {"role": "system", "content": "資料だけを根拠に答えてください。"},
  {"role": "user", "content": f"# 資料\\n{context}\\n\\n# 質問\\n{question}"},
]
resp = client.messages.create(model=..., messages=messages)`,
    note: 'RAG の「G」はここ1回の API 呼び出し。仕事のほとんどは、その前の検索で決まる。',
  },
];

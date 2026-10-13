<h1 align="center">Strata-GLM-AMD</h1>

[English](README.md) · **日本語**

<p align="center"><b>GLM-5.3-Flash を 256K コンテキストで、AMD のワークステーションカード 2 枚で動かす</b><br>
<a href="https://github.com/Niko1221/Strata">Strata</a> のフォーク · Linux + ROCm · 無料のオープンソース</p>

Strata-GLM-AMD は [Strata](https://github.com/Niko1221/Strata) に 2 つ目のエンジン **`strata-glm5`** を追加したものです。
**[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**（`glm5-next` アーキテクチャ：45 層、288 エキスパート、
マルチトークン予測つき）を、48 GB の AMD カード 2 枚で **256K トークンのコンテキスト** で動かします（1 枚でも動きますが遅くなります）。
チャット、コード生成、必要なときの思考、tool call に対応し、Strata と同じ OpenAI / Anthropic / Responses API と
ブラウザアプリを提供します。このモデルに関係しない Strata の部分はそのままです。元の README は
[upstream](https://github.com/Niko1221/Strata/blob/main/README.ja.md) にあります。

> **このリポジトリは AMD RDNA 3 に特化しています**（gfx1100：Radeon Pro W7900、Radeon RX 7900 XTX）。それ以外ではテストしていません。
> ほかのアーキテクチャ（RDNA 4、CDNA、NVIDIA）に対応したい場合は、お手持ちの Codex（などのコーディングエージェント）に
> このリポジトリと [docs/GLM5_NEXT.md](docs/GLM5_NEXT.md) を渡して聞いてください。

エンジンは Strata の設計（よく使うエキスパートは GPU、残りはピン留めした RAM、ミスは CPU で計算、小さなドラフトモデルで
「予想してから確認」）の上に、このモデル向けの工夫を重ねています：

- **長文をコンテキスト長に依存しないコストで読む。** このモデルの DSA 層は、学習済みの indexer でコンテキスト全体から attention の
  セルを選びます。Strata-GLM-AMD は indexer の選択結果は使いつつ、評価を窓単位で行います。窓ごとに選択を共有し、選ばれたセルを
  1 つのコンパクトな int8 オペランドに集め、decode は latent キャッシュ上の split-K 融合カーネルで読む。これで窓あたりの
  attention コストがコンテキスト長に依存しなくなりました。128K のプロンプトは素の経路の 3 倍速く読め、needle-in-a-haystack は
  236K まで通ります。
- **キャッシュはすべて Q8_0。** latent KV、indexer のキーとゲートのキャッシュを 8-bit 行で持ちます。1 スロットの 256K コンテキストは
  f32 なら 5 GiB のところ、カードあたり約 1.2 GiB。prompt の attention はこれをそのまま int8 オペランドとして読みます。
- **Q2_K エキスパートをテンソルコアで。** 2-bit のエキスパート重み向けの WMMA カーネルを、prompt 用（汎用カーネルの 10〜12 に対し
  22〜28 TMAC/s）と decode 用（115〜160 GB/s に対し 550 GB/s）に用意。量子化はダウンロードしたままです。
- **decode 中のエキスパートキャッシュ。** カードに載らないエキスパートはピン留め RAM から CPU で計算しますが、これはメモリバス帯域の
  床です。decode 中は prompt 用バッファを「PCIe で運んだエキスパートのキャッシュ」として貸し出し、新しい会話でも VRAM に当たる率を
  上げます（新規プロンプトで +7〜19%）。同梱のルーティング profile（`data/glm53/expert-profile.txt`）が静的な順位の初期値になり、
  使うほど蓄積されます。
- **2 枚のカードで prompt をパイプライン処理。** 層をカードに分け、カード 1 が窓 k を処理している間にカード 0 が窓 k+1 を始めます。
  128K のプロンプト：592 → 1,060 tokens/s、出力は完全に同じ。
- **tool call と思考** は GLM-5 自身のチャットテンプレート形式（`<tool_call>name<arg_key>...`）。思考はリクエストごとに on/off。

## どのくらい速い？

開発機で測りました：Radeon Pro W7900 ×2（各 48 GB、gfx1100）、Ryzen 7 5700X3D（8 コア、AVX2 のみ）、RAM 125 GB、Linux、ROCm 7。
モデルは 3-bit GSQ-RCO の GGUF（117 GB）と Q4_K の MTP ドラフト層です。

| | 2 枚 | 1 枚 |
| --- | ---: | ---: |
| **回答の書き出し**（チャット 1 本、ドラフトモデルあり） | 30〜39 tokens/s | 15〜16 tokens/s |
| **チャット 2 本同時**（`"parallel": 2`、合計） | 約 30 tokens/s | - |
| **4K プロンプトの読み込み** | 750 tokens/s | - |
| **12K プロンプトの読み込み** | 1,000〜1,100 tokens/s | 490 tokens/s |
| **128K プロンプトの読み込み** | 1,060 tokens/s | - |
| **Needle in a haystack**（3 深度） | 236K まで 3/3 | - |

書き出しの速さはこの機体のホスト RAM 帯域で決まります（カードに載っていないエキスパートを毎トークン、ピン留め RAM から読むため）。
メモリチャネルの速い機体ならもっと速くなります。プロンプトの読み込みはカード上のエキスパート積で決まります。
数値の全部、その裏にある実験、うまくいかなかった試みは [docs/GLM5_NEXT.ja.md](docs/GLM5_NEXT.ja.md)（英語版: [GLM5_NEXT.md](docs/GLM5_NEXT.md)） にあります。

## 必要なもの

| | |
| --- | --- |
| **グラフィックカード** | テスト済み：**AMD Radeon Pro W7900 ×2**（48 GB、gfx1100）。W7900 1 枚でも、常駐エキスパートを減らせば動きます（[1 枚の場合](#1-枚の場合)）。48 GB のほかの RDNA 3 / RDNA 4 カードは動くはずですが未テスト。NVIDIA は未テストです。 |
| **RAM** | **128 GB。** カードに載らないエキスパートはピン留めした RAM に置きます。2 枚なら 43 GiB、1 枚なら 86 GiB。 |
| **ディスク** | モデルファイルに約 125 GB。SSD に置いてください。 |
| **OS** | Linux と ROCm 7（HIP エンジンは手元の PC でコンパイルします）、Python 3.11 以上、C++ コンパイラ、CMake 3.24 以上、Ninja。 |

## インストール

### インストーラーで

```bash
git clone https://github.com/OpenMOSE/Strata-GLM-AMD
cd Strata-GLM-AMD
./setup.sh --family glm53
```

カードと RAM を調べ、3 つの質問（カード 2 枚か 1 枚か、コンテキスト、同時リクエスト数）をして、モデルファイル 2 つ（122 GB）を
ダウンロードし、エンジンをコンパイルし（初回 10〜20 分）、トークナイザを取り出し、カードに合わせた expert budget で設定ファイルを
書いて、モデルを起動します。Enter を押せばおすすめの答えになります。引数で渡すこともできます：
`--gpus 0,1 --context 262144 --parallel 1 --yes`。モデルファイルをすでに持っているなら `--gguf-dir <フォルダ>`。
次回からは `./setup.sh` だけで起動します（`run-glm53.sh` でも可）。`./setup.sh --family glm53` をもう一度実行すると設定を
変えられます。`git pull` のあとは `./update.sh` を実行してください（起動だけではエンジンは再ビルドされません）。

### 手動で

同じ手順を 1 つずつ。

#### 1. エンジンをビルドする

```bash
git clone https://github.com/OpenMOSE/Strata-GLM-AMD
cd Strata-GLM-AMD
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_PREFILL_MMQ=ON
cmake --build build-hip --target strata-glm5 -j
```

`STRATA_PREFILL_MMQ=ON` は必須です（prompt の経路は ggml の int8 MMQ カーネルで動きます）。`STRATA_GGML_DIR` を指定しなければ
CMake がピン留めされた ggml のコミットを取得します。自分の llama.cpp チェックアウトを使うならそのパスを指定してください。
初回のビルドは 10〜20 分かかります。

#### 2. モデルをダウンロードする

Hugging Face から GGUF を 2 つ：

- **モデル本体：** [`GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf`](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
  （117.5 GB）。3-bit の GSQ-RCO 量子化、エキスパートは Q2_K。
- **ドラフト層：** [`GLM-5.3-Flash-MTP-Q4_K.gguf`](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF)
  （4.6 GB）。モデル自身のマルチトークン予測層で、次のトークンの予想に使います。

#### 3. トークナイザを取り出す

トークナイザとチャットテンプレートをモデルファイルから小さなディレクトリ（約 20 MB）に取り出します：

```bash
.venv/bin/python tools/strata_tokenizer.py --gguf /path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf \
      --out /path/to/strata-data/glm53/pack
```

これで `.../glm53/pack/tokenizer/` が書かれます。このときチャットテンプレートにパッチが当たり、思考オフのリクエストで本当に思考しなくなります（GLM のテンプレートは
そのままだと常に think ブロックを開きます）。

#### 4. サーバの設定を書く

Strata-GLM-AMD フォルダに `strata-glm53.json`（2 枚、同時 1 リクエスト、256K コンテキスト）：

```json
{
  "model_name": "GLM-5.3-Flash",
  "exe": "/path/to/Strata-GLM-AMD/build-hip/strata-glm5",
  "cwd": "/path/to/Strata-GLM-AMD",
  "tokenizer": "/path/to/strata-data/glm53/pack/tokenizer",
  "log": "/path/to/Strata-GLM-AMD/strata-glm53.log",
  "backend": "hip",
  "gpu": [0, 1],
  "env": {"HIP_VISIBLE_DEVICES": "0,1", "STRATA_GLM5_PF_BUFS": "2",
          "STRATA_GLM5_LOD_ROUTE": "pool", "STRATA_GLM5_LOD_TOP_POOLS": "1024", "STRATA_GLM5_LOD_NOSUM": "1"},
  "engine_silence_s": 600,
  "parallel": 1,
  "args": ["--serve",
           "--gguf", "/path/to/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf",
           "--mtp", "/path/to/GLM-5.3-Flash-MTP-Q4_K.gguf",
           "--devices", "0,1",
           "--expert-budget-gib", "31.0,29.5",
           "--profile", "/path/to/strata-data/glm53/profile_chat.txt",
           "--profile-save", "/path/to/strata-data/glm53/profile_chat.txt",
           "--window", "8", "--prefill-window", "4096", "--spec", "2",
           "--ctx", "262144",
           "--attn", "lod", "--lod-top", "128", "--lod-top-decode", "64"]
}
```

`--expert-budget-gib` はカードごとにエキスパートが使ってよい VRAM です。それ以外（密な重み、キャッシュ、prompt のバッファ、
prefetch ring）が先に確保されます。上の値だと 256K で W7900 1 枚あたり 1.2〜1.5 GB 空きます。起動時に `out of memory` と
出たら下げてください。起動ログは各段階のあとの空き VRAM を表示します。profile ファイルはエキスパートの使用頻度の順位で、
最初は無くてかまいません（その場合は id の小さい順）。リクエストごとに保存されます。インストーラーは実際のチャットで測った
順位（`data/glm53/expert-profile.txt`）を初期値として置きます。

#### 5. 起動する

```bash
.venv/bin/python serve/server.py --engine strata --config strata-glm53.json --port 8096
```

ロードには 2〜3 分かかります（密な重み、60 GB のエキスパートをカードへ、43 GB を RAM にピン留め）。そのあとブラウザアプリが
`http://127.0.0.1:8096`（Chat、Monitor、About）で、API が `/v1/chat/completions`、`/v1/messages`、`/v1/responses` で使えます。
Strata とまったく同じです（[使い方](https://github.com/Niko1221/Strata/blob/main/README.ja.md#使い方)）。
ほかのマシンから使うには `--host 0.0.0.0 --api-key <secret>` を付けてください。

## 構成

3 つの構成を測定しました。パスを置き換えた設定ファイルが [`configs/glm53/`](configs/glm53/) にあります。

| ファイル | カード | コンテキスト | 同時リクエスト | `--expert-budget-gib` | 備考 |
| --- | --- | --- | --- | --- | --- |
| `strata-glm53-np1.json` | 2 | 256K | 1 | 31.0,29.5 | いちばん速い：60 GB のエキスパートが常駐 |
| `strata-glm53.json` | 2 | 256K | 2（`"parallel": 2`） | 28.5,25.8 | 2 本のチャットが同時に decode。スロットごとに 256K のキャッシュ |
| `strata-glm53-1gpu.json` | 1 | 256K | 1 | 17.9 + `--pin-gib 96` | 下を参照 |

**コンテキストと VRAM。** バッチスロットそれぞれと solo スロットが自分のキャッシュを持ちます。256K ではスロットあたり
カード 0 で 1.2 GiB、カード 1 で 1.8 GiB なので、`"parallel": 2` は同時 1 本よりカードあたり約 3 GB 多く VRAM を使い、
そのぶんエキスパートの予算、つまり decode の速さが減ります。

### 1 枚の場合

W7900 1 枚には、密な重み、キャッシュ、2×3.8 GB の prefetch ring、18 GB のエキスパートが載ります。残り 86 GB のエキスパートは
**ホスト RAM にピン留め** しなければなりません（`--pin-gib 96`。既定の上限 64 GB だとピン留めされず、prompt の経路がその数を
staging できません）。RAM は 128 GB が最低ラインです。decode は 15〜16 tokens/s、12K のプロンプトは約 490 tokens/s。
`"gpu": [0]`、`HIP_VISIBLE_DEVICES=0`、`--devices 0`。

### エンジンのオプション

| オプション | 意味 |
| --- | --- |
| `--ctx N` | スロットあたりのコンテキスト（262144 までテスト済み） |
| `--devices 0,1` / `--layer-split` | 使うカードと、カード間の層の分割（auto：VRAM に応じて） |
| `--expert-budget-gib G[,G2]` | 常駐エキスパートに使うカードごとの VRAM |
| `--pin-gib G` | 非常駐エキスパートのピン留め RAM 上限（既定 64）。超えるとファイルから読む |
| `--prefill-window T` | prompt の窓のトークン数（4096 推奨。窓が 2 枚のパイプラインに流れる） |
| `--window T`、`--spec K` | decode の窓とドラフトトークン数（8 と 2） |
| `--attn lod`、`--lod-top`、`--lod-top-decode`、`--lod-page`、`--lod-local`、`--lod-sink` | 窓単位のスパース attention モードと選択のサイズ |
| `--slots N`（設定の `"parallel": N`） | バッチスロット。推奨最大は 2 |
| `--profile FILE`、`--profile-save FILE` | 起動時に読む / リクエストごとに保存するエキスパート使用頻度の順位 |

環境変数（設定の `"env"`）：`STRATA_GLM5_LOD_ROUTE=pool`（indexer のプールでページを選択。`mean`、`tok`、`idx` は実験的な
ルーティング）、`STRATA_GLM5_LOD_NOSUM=1`（ページ要約なし＝窓つき DSA。サーバの既定）、`STRATA_GLM5_LOD_TOP_POOLS`
（prefill のプール数、1024）、`STRATA_GLM5_PF_BUFS`（prefetch ring の深さ、2）、`STRATA_GLM5_NOPIPE=1`（2 枚のパイプラインを使わず
窓を順に処理）、`STRATA_GLM5_NOQ2K=1`（WMMA の Q2_K カーネルの代わりに汎用 MMQ）。

## 既知の制限

- テストは Linux + ROCm 7 + gfx1100 のみ。NVIDIA 向けのビルドは同じ CUDA 風のソースを通りますが、動かしていません。
- 画像は未対応：GLM-5.3-Flash の vision projector はこのエンジンにつないでいません。
- `"parallel"` は 2 までを推奨。decode はメモリ帯域律速で、スロット間で共有されます。
- チャットの最初のメッセージは全部読みます（2 枚で 1,000 tokens/s 以上）。続きのメッセージは会話キャッシュから再開します。
  長い共通プレフィックスを持つ別の会話は読み直しになります。
- 1 枚では prompt の窓ごとにほぼ全エキスパートを PCIe で prefetch するため、プロンプトの読み込みは 2 枚の半分の速さです。

## もっと詳しいしくみ

[docs/GLM5_NEXT.ja.md](docs/GLM5_NEXT.ja.md)（英語版: [GLM5_NEXT.md](docs/GLM5_NEXT.md)） が開発記録です。GGUF と llama.cpp から読んだモデルの構造、decode 経路（mHC、KDA、DSA、
MoE と doorbell）、エキスパート階層と各方式の計測、MTP ドラフト層、prompt 経路（int8 MMQ、prefetch ring、Q2_K WMMA カーネル、
4096 トークン窓、2 枚のパイプライン）、窓単位の attention モードとサイズ・ルーティングごとの NIAH 結果、サーバの修正
（stop トークン、思考オフ、tool call）。照合に使った参照実装は llama.cpp の `glm5-next` で、`tools/glm_oracle/` がその中間テンソルを
吐き、`ref/glm5.py` が numpy での再実装です。

## 更新内容

- **2026-10-13 (後半)** - 小さな改善の第 3 回、decode ラウンド: KDA 層の射影を積み、scan をゲートと norm と融合、mHC 末尾を 1 起動に、
  常駐 expert の down 積と共有 expert を GPU が CPU 供給 expert を待つ間に計算、生成側カーネルが q8 行も書く、decode 時の expert
  キャッシュを prompt スクラッチへ拡張 (622 → 1,514 スロット)。ラウンドあたりの起動 3,430 → 約 2,100、decode ラウンド 79 → 67 ms
  (40 トークンのテストで 28.8 → 約 34 tok/s)、プロンプトは変わらず ([docs/GLM5_NEXT.md](docs/GLM5_NEXT.md))。
- **2026-10-13** - 小さな改善の第 1 回・第 2 回: 単一トークンの GEMV が 8 列まで重み行を 1 回で読む、mHC と DSA の absorb/expand
  カーネルを窓の大きさに合わせる、起動時の BLAS ウォームアップ、中サイズ窓の CPU/PCIe 分担を 640 トークンまでの単一窓プロンプトに拡張。
  decode 27.0 → 28.8 tok/s、40 トークンのプロンプト 1.81 → 0.97 s、147 トークン 3.02 → 2.03 s、430 トークン 3.35 → 2.58 s
  ([docs/GLM5_NEXT.md](docs/GLM5_NEXT.md))。
- **2026-10-10** - ドラフト本数はラウンドあたり 2 本固定のまま。ラウンドごとに本数を変える方式 (実測レートで 2〜4 本) を試したが、
  利得が測定のばらつきの範囲だったため撤回 ([docs/GLM5_NEXT.md](docs/GLM5_NEXT.md))。
- **2026-10-09** - decode: 小さな窓向けに組んだ WMMA カーネルで Q2_K エキスパート積 (115〜160 GB/s に対し 550 GB/s)。decode 中の
  エキスパートキャッシュを prompt 用バッファに (新規プロンプトで +7〜19%)。それが長い回答で起こしたクラッシュの整合性修正。
  ルーティング profile を同梱し、保存をセッションをまたいで累積に。`./update.sh` が質問なしでエンジンを再ビルド。リクエストごとの
  ログ行にキャッシュの件数。
- **2026-10-08** - 初回リリース: GLM-5.3-Flash を 48 GB の AMD カード 2 枚で動かす `strata-glm5` エンジン、256K コンテキスト、
  2 枚のカードでの prompt パイプライン、GLM-5 形式の tool call、思考の on/off、インストーラー (`./setup.sh --family glm53`)。

## クレジットとライセンス

[Strata](https://github.com/Niko1221/Strata)（Niko1221）がここにあるすべての土台です（サーバ、アプリ、エキスパート階層、投機、
HIP バックエンド）。[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) は Z.ai によるものです。3-bit GSQ-RCO の GGUF は
[pfeifferj](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)、MTP の GGUF は
[neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF) によるものです。エンジンは
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) のカーネル（MMQ）を使い、llama.cpp の `glm5-next` 実装に従っています。
Strata-GLM-AMD は [MIT License](LICENSE) のオープンソースです。モデルにはそれぞれ独自のライセンスがあります。

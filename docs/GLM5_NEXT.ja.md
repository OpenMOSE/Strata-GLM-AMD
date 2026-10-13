[English](GLM5_NEXT.md) · **日本語**

# GLM-5-Next (GLM-5.3-Flash) を Strata で動かす: 差分と計画

対象ファイル: `/home/client/Projects/llm/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` (117.5 GB, `general.architecture = glm5-next`)
参照実装: `llama.cpp` (`src/models/glm5-next.cpp`, コミット 0faee5004 / 2026-10-04)
開発機: Radeon Pro W7900 ×2 (gfx1100, 48 GB ずつ), Ryzen 7 5700X3D (8 コア, AVX2 のみ), 125 GB RAM

この文書は「何が違うか」(第 1〜4 節) と「どう作るか」(第 5〜8 節) の二部構成。数字はすべて GGUF のメタデータ、
llama.cpp のソース、Strata のソースから読んだもの。推定値には「推定」と書く。

---

## 進捗 (2026-10-04)

| フェーズ | 状態 | 成果物 |
| --- | --- | --- |
| P0 オラクル | **完了** | `tools/glm_oracle/dump_tensors.cpp` (+`build.sh`): llama.cpp に名前付き中間テンソルを .npy で吐かせる。`ref/glm5.py`: numpy 参照実装。CPU 実行の llama.cpp と greedy 4 ステップ一致、層ごとのコサイン 0.9999 以上 |
| P1 ローダ | 骨格 | `src/glm5/model.cpp`: GGUF のメタデータから `HParams`、密テンソルを GGUF の型のまま VRAM へ (9.6 GiB、3 秒)。パック形式・トークナイザは未着手 |
| P2 カーネル | 単トークン版 完了 | `src/glm5/kernels.cu`: mHC (Sinkhorn)、KDA (3 conv、ベクトル減衰の再帰)、DSA (k-pool、スコア、latent 上の MQA、wk_b 吸収 / wv_b 展開)、sigmoid ルータ、clamp SwiGLU。`native_mmvq` に **Q2_K** を追加 (`native_q2k_parity`: ggml と相対 5e-3、多列ビット一致) |
| P3 decode 貫通 | **完了 (bring-up)** | `strata-glm5 --gguf … --tokens … --n-predict K --dump DIR`。llama.cpp と greedy トークン一致 (3 ステップ、"The capital of France is" → " Paris"), 最終 logits のコサイン 0.9991 |
| P4 expert 常駐 | **動作 (静的)** | `src/glm5/experts.cpp` `ExpertStore`: 層ごとのデバイス割当、プロファイル順 (なければ id 順) に VRAM 予算まで常駐、ミスはその場で mmap から H2D。`--profile-save` でルーティング回数を保存。2 GPU の層分割 (`--devices 0,1 --layer-split auto`): 各カードが自分の層の密な重み・状態・expert を持ち、境界で R (64 KB) をピアコピー |
| P4b doorbell | **動作** | MoE の ids を GPU がマップ済みピン留めメモリに publish し、ホストはミスだけを DMA して flag を立て、GPU 側は `moe_wait` カーネルで待つ (Strata 本体の doorbell と同じ形)。DSA の top-k も GPU 上 (radix select)。ホスト同期は 1 トークンあたり 52 回 → 19 回 (残りは argmax とデバイス境界) |
| P6 検証窓 + MTP | **動作** | `forward_window(T<=8)`: 密な重みを窓ごとに 1 回読み、expert は (層, expert) ごとに 1 回 (`native_mmvq_grouped`: ポインタ表 + 列数、単列呼び出しとビット一致)。KDA はトークンごとの状態スナップショットで `commit(n)` の巻き戻し。MTP (NextN) 層を `attach_mtp` で頭デバイスに載せ、expert 4.4 GB は常駐。投機: ドラフト k 個 + 1 の窓で検証、greedy と同一のトークン列 |
| P7 サーバ接続 | **動作 (2026-10-05)** | `strata-glm5 --serve`: `serve/server.py` の行プロトコル (GEN/STOP/QUIT ↔ INFO/READY/PP/RESUME/T/DONE)。トークナイザは `tools/strata_tokenizer.py` に `glm4` 前処理 (CHATGLM4 正規表現 + `ignore_merges`) を追加し `--gguf` から抽出 (llama-tokenize と 6 種の文で一致)。設定 `strata-glm53.json`、Chat ページ / OpenAI 互換 API で応答。会話キャッシュ: 前回の続きなら継続、前回のプロンプトを延長するだけ (思考が履歴から落ちた場合) なら「マーク」(プロンプト末尾の KDA 状態 + 隠れ状態) へ巻き戻し |
| P5 prefill | **第 1 段 (2026-10-05)** | プロンプト窓を 8 → 最大 2048 トークンに (`--prefill-window`, 既定 256、サーバ設定は 1024)。1,064 トークンのプロンプト: 窓 8 で 54.6 s (19 tok/s) → 窓 256 で 19.9 s (53) → **窓 1024 で 13.9 s (77 tok/s)**、生成トークン列は窓 8 と完全一致。実装: `k_moe_group` を counting sort に書き換え (任意 T、expert 昇順にグループ化、ホストもミスを昇順に番号付け)、グループ化 MMVQ の 8 列超は列チャンクでループ、密 GEMV は 8 トークンずつ、KDA スナップショットは先頭 8 トークンのみ (プロンプト窓は巻き戻し不要)、logits は最終トークンだけ、MTP の追い付きも窓 (`mtp_window`)、staging は層あたりの非常駐数 (≤288 スロット)、プロンプト窓では swap しない (全 expert が要るためスラッシュするだけ)。**doorbell をホストの専用スレッドに** (エンキュー側のスレッドがキュー満杯で launch にブロックしても、GPU が待つ flag を別スレッドが書けるので、デッドロックしない。窓 256 以上では必須だった) |
| P8 | 未着手 | グラフ捕捉、ミスの CPU 計算、prefill 第 2 段 (下) |

**expert 階層の計測と改修 (2026-10-04 深夜)**

ラウンドごとの内訳 (`STRATA_GLM5_ROUNDS=1`、`STRATA_GLM5_DMA=1`) で、1 ラウンド (窓 4) の時間はほぼ全部が「ミスした expert の PCIe 転送待ち」と分かった:
ミスは窓あたり 50〜200 個 (distinct)、1 個 8.26 MB、しかも層ごとに直列 (層 L+1 のルーティングは層 L が終わるまで分からない)。

| 改修 | 結果 |
| --- | --- |
| 静的常駐のみ (profile) | 103〜200 ms/ラウンド。生成文が同じ句の繰り返しでも毎ラウンド同じ expert をミス |
| 適応スワップ v1 (窓の終わりに staging→slot の D2D) | **誤り**: staging は層で共有、最後の層以外のデータは上書き済み → NaN。撤回 |
| swap-in on miss (ミスを犠牲スロットへ直接 DMA、デバイスは wait 後に `k_moe_place` で配置を適用) | 正しい。追い出された expert の再ミスが pread (ページキャッシュが荒れて NVMe 速度) になり遅い |
| 退避プール (追い出し時に D2H でピン留めプールへ、再ミスは DMA 1 回) | pread 消滅。ただし D2H が同じコピーエンジンに乗り実効 7〜8 GB/s |
| 先読み (次層のルータを今の層の FFN 入力に掛ける: top-8 中 6.1 個的中 = 76%) | 正しく動くが**遅くなる**: 外れ分 + 追い出しで転送量が増え、PCIe が律速のため逆効果。既定オフ (`--lookahead N`) |
| 犠牲 = ピン留めコピー持ちを優先 | D2H 消滅 (14 GB/s) だが常駐集合が profile で凍結し作業集合がスラッシュ (ミス増加)。撤回 |
| **真の LRU + 同率ならピン留め持ち + プール 20 GiB/枚** | **37.9 ms/token (26.4 tps)**、採択 2.8/3、3.84 トークン/ラウンド、トークン列は greedy と一致 |

教訓: この機体では **PCIe 帯域 (実測 14 GB/s) × ミス数** が decode の床。先読みは帯域が余っているときにしか効かない。
残る手段は (a) ミスを CPU で計算する (Strata 本体の設計: 8 MB を運ばず 16 KB を返す。AVX2 8 コアの Q2_K で 1 expert 1 トークン
約 0.3〜0.5 ms、GPU と並列)、(b) 常駐率を上げる (今 63%: 密 10 GB + MTP 4.4 GB + KV を除くと 2 枚で 67 GB)、(c) 窓を大きくして
密な重み読み出しを薄める (窓 8 のプロンプトは 42 tok/s)。ハング対策: 例外時は Engine/Model/ExpertStore がデバイスを触らない
(`faulted_`、`std::uncaught_exceptions`)、main は意図的にリークする。rocprofv3 の下では doorbell 版がデッドロックする (理由不明)。

**WebUI 経由の実トラフィック (2026-10-05)**: ベンチマーク (英語ドキュメント文、同プロンプトのプロファイル) では 26.4 tps / 採択 93% /
ヒット 99.4% だったが、実際のチャット (日本語・英語・コード・思考あり、16 プロンプト × 350 トークン) では **7〜23 tps**
(中央値 15)、採択 56〜85%、ヒット 85〜96%。差の正体は要求ごとの統計行 (`glm5: … experts: …`) で判明:
(1) LRU の swap-in on miss が常駐集合 (62%) より大きい作業集合で **ほぼ全てスラッシュ** (swap 9,000〜14,000 / 350 トークン、
thrash 97%)、追い出しの D2H + 再取得の H2D で PCIe が倍食われる。(2) 退避プール (20 GiB/枚) が 1 要求で 2〜3 周し、
溢れた expert は GGUF からの pread に戻る (1 要求で 5〜9 GB、3〜5 秒)。(3) MTP の採択率は内容依存 (日本語・コードで低い)。
対策として **admission filter** (`--admit N`、既定 32 窓: 直近 N 窓以内に 2 回ミスした expert だけ常駐を奪う。一度きりの
expert は staging 止まり)、実トラフィックから取ったプロファイル (`--profile-save` を serve 中も要求ごとに保存)、そして
**交換スワップ**: 全 expert の居場所を「VRAM スロット か ピン留めホスト blob のどちらか 1 つ」に固定し、スワップは
入ってくる expert を staging へ H2D → 犠牲を入ってきた expert のホスト blob へ D2H → staging をスロットへ D2D (同じコピー
ストリームで順序保証)。退避プールは不要 (`--evict-pool-gib` 既定 0)、ロード後のファイル読みはゼロ、ピン留めホストは
補集合ぶん (36.9 GiB) だけ。同じ 16 プロンプトの再生:

| 構成 | 全体 tok/s | 最遅の要求 | ラウンド中央値 | swap/要求 | pread/要求 | ホスト RAM |
| --- | --- | --- | --- | --- | --- | --- |
| LRU 常時 admit、ドキュメント profile、プール 20 GiB/枚 | 10.9 | 58.4 s | 186 ms | 8,993 | 1,304 (11 GB, 6.6 s) | 87 GB |
| admit 32、チャット profile、プール 20 GiB/枚 | 15.4 | 41.5 s | 140 ms | 4,279 | 1,066 (9 GB, 2.7 s) | 87 GB |
| **+ 交換スワップ、プールなし** | **17.9** | **21.4 s** | ~140 ms | ~4,500 | **0** | **47 GB** |

**prefill の内訳と次の一手 (2026-10-05)**: 窓 1024 の 1 窓 ≈ 12 s のうち PCIe (非常駐 expert ≈ 42 GB/窓、窓の大きさによらずほぼ一定) は ≈ 3 s。残り ≈ 9 s が計算で、
内訳は (a) 密な Q8_0 重み 10 GB を 8 トークンごとに読み直す MMVQ (1024 トークンで 1.3 TB の VRAM 読み ≈ 1.5 s) → hipBLASLt の BF16 GEMM
(`src/prefill/gemm.cu` が既にある) に置き換える、(b) DSA の per-token ループ (1024 × 3 launch × 11 層 = 34k launch) → T をまとめたカーネルに、
(c) KDA の逐次 scan (1024 ステップ × 34 層) → チャンク並列化、(d) expert の列チャンク再読み。さらに、プロンプト窓では次層の非常駐 expert を
**全部先読み** (大きな窓ではほぼ全部使うので安全な上位集合) すれば PCIe 3 s を計算と重ねられる (staging を 2 面に)。デバッグ中に見つけた
`d.probs`/`d.wts` 未確保 (行末コメントで確保文を潰していた) は修正済み。

**prefill 第 1 段の到達点 (2026-10-05 夕)**: 1,064 トークン 8.2 s (130 tok/s)、トークン列は窓 8 と一致。1 窓 (1024) の内訳: MMQ expert 2.5 s
(gate|up 1.06 + down 1.46; 期待より 5〜10 倍遅い、要 rocprof)、router+group 0.92 (行数取得の同期を含む)、KDA q/k/v GEMM+conv 0.77、DSA attend 0.59、
DSA 出力 0.42、DSA 射影 0.36、KDA 出力 0.23、共有 expert 0.18、KDA scan 0.18 (レジスタ化で 1.4 s → 0.18 s)、gather 0.16。PCIe は先読みで 0。
上流 (`original/Strata` 6f32ec0、#465 batch slots ほか) を `upstream` remote としてマージ済み (衝突は CMakeLists.txt のみ)。

**decode: ミスを CPU で計算 (2026-10-05 夜)**: 本家の設計を GLM に適用。decode 窓 (T ≤ 8) では、ルーティングと一緒に MoE 入力 (T × 4096 floats)
をマップ済みホストメモリへ publish し、doorbell スレッドがミスした expert をピン留め blob 上で ggml-cpu の vec_dot (Q2_K/Q3_K/Q4_K × Q8_K) で
計算 (8 スレッド、gate/up 128 行チャンク → clamp swiglu → down 512 行チャンク)、結果 (16 KB/ペア) をデバイスのペア行バッファへ H2D して flag。
GPU 側は hits の gate/up/down を計算し、combine の直前でだけ待つ。ペアの並びはホストとデバイスで同じ (hits 昇順 → misses 昇順、グループ内はトークン順:
`k_moe_group` を n ≤ 64 で決定的に)。結果: 1 ラウンド 170〜280 ms → **83〜88 ms**、moe_wait 90〜200 ms → 0、decode **14.6 → 28.4 tok/s**
(96 トークン、採択 74%、2.46 トークン/ラウンド)。トークン列は GPU 計算と同じ。残り: moe_down 24 ms (ペア行カーネルがプロンプト窓の 8,192 行で
起動していた → 窓サイズに修正)、moe_hits 18 ms、小カーネル約 40 ms (グラフ捕捉で削る)。

**decode の続き (2026-10-05 深夜)**: (1) doorbell の flag をホスト→ホストの `hipMemcpyAsync` で書いていたが、これは呼び出しスレッドを
コピーストリームが空くまでブロックする (CPU サービスの時間の 95% がこれだった)。コピーストリーム上の 1 スレッドのカーネルで書くように変更
(別デバイスのストリームへの launch はそのデバイスを current にしてから)。prefill も 128 → 138 tok/s。(2) ミスの分担: 各層のミスを
コスト見積もり (CPU 1 ペア 0.25 ms、PCIe 1 expert 0.65 ms、GPU が各 wait の前に持つ仕事 0.4 / 1.0 ms) で CPU と PCIe に分け、PCIe 側は
既存の swap-in (admission filter 付き) を通して常駐集合を会話に適応させる。flag を 2 本に (DMA 到着 → ミスのカーネル、CPU 結果到着 →
combine)。200 トークン decode: **25.2 tok/s** (94 ms/ラウンド)、全部 CPU で swap なしだと 19.9 tok/s (ヒット率 57% のまま)。
残る床: 1 層あたり GPU 自身の仕事 ≈ 2 ms (attention + ルータ + hits) に対して、ミスのサービス (CPU 1.5 ms / PCIe 1.8 ms) が
publish → 計算 → combine の鎖で直列に乗る。常駐率が上がるほど消える。次: 窓のグラフ捕捉 (小カーネル約 40 ms/ラウンド)、
グループ化 MMVQ の効率 (moe_hits 12〜18 ms = 帯域の 1/4)、採択率。

**prefill 第 2 段 (2026-10-06)**: 1,064 トークン 7.7 s → **3.46 s (308 tok/s)**、1 窓 (1064) の GPU 内訳 3.4 s: moe_wait (PCIe) 1.38 s、
expert MMQ 1.08 s、KDA scan 0.19、KDA ゲート 0.16、KDA qkv 0.14、DSA attention 0.07、その他 0.4。やったこと (計測はすべて標準の
ハーネスで、`scratchpad/mmq_bench` 等):
1. **hipBLASLt / rocBLAS の FP16 GEMM はこの GPU で 13 TFLOPS が上限** (ROCm 7.0、1024×4096×4096; 付属のチューニング表は hipBLASLt
   100100 用で 100401 では不一致)。llama.cpp の int8 MMQ は Q8_0 密重みで **36 TMAC/s (72 TOPS)** → 幅広い窓の密な積を全部 MMQ に
   (`gemv()` の分岐、`STRATA_GLM5_NODENSEMMQ=1` で戻せる)。注意: MMQ の活性化は k ブロック主導のレイアウトで行数がストライド →
   quantize の行数と Product.total_rows は一致させる。
2. **MMQ の列タイルは型ごとに固定**: Q2_K は 8 (32/96 タイルは 15〜100 倍遅い、64/128 は半速)、Q3_K/Q4_K は 24。自動選択 (最大
   expert の行数) は 64〜128 を選んでいた。`Product.ncols_opt` を追加。
3. 結合カーネルを per-token 表 (`tok_pairs`) に (全ペア走査で 1 s/窓かかっていた)。
4. k_b/v_b (BF16) の **Q8_0 コピー** を作り、absorb/expand を「head = expert」の MMQ に (0.6 s → 0.07 s)。
5. **先読みリング** (デバイスごと 4 本、非常駐 expert だけを rank 順に圧縮格納; 常駐 expert はスロットを直接読む 2 積構成) で
   gather (0.16 s) を廃止。中規模窓 (9..255) はルーティングされた非常駐だけを doorbell スレッドが DMA。
6. **MTP 層は gate/up が Q4_K、down が Q5_K** で blob がブロックサイズの倍数にならず、MMQ (expert をブロック単位でアドレス) が
   2 番目以降の expert を読み違えていた → スロット/リングのピッチを lcm(ブロック) の倍数に切り上げ (`LayerExperts::stride`)。
7. **密な DSA attention** (8K セルまで): latent cache を Q8_0 行 (と転置) に再量子化し、スコアと P·V を 64 トークン/バッチの MMQ 2 積、
   選択は bit マスク付き softmax。669 → 66 ms。`STRATA_GLM5_NODENSEATTN=1` で戻せる。
8. プロンプトは **等分の窓** (≤ 2048): 非常駐 expert の転送 (42 GB) は窓の大きさによらないので、1024+40 の尾の窓は 1024 と同じコストだった。
9. バグ: bounds 表をピン留めホストから async copy していて、ホストが先行して書き換える競合 → デバイス側カーネルで書く (`set_pair`)。
   doorbell の flag 書きも同じ理由でカーネル化済み。リングのイベントは層ごと (バッファごとだと自分の record を待つ)。
今の床: PCIe。1 窓あたり非常駐 expert 42 GB (各 GPU 21 GB、14 GB/s = 1.5 s) に対して計算 2.0 s。リングが 4 層ぶんしかないので
相手 GPU の計算中に先行できるのは 0.4 s → 約 1.4 s が露出。窓を 2048 にすると計算 4 s > 転送 → 隠れる見込み (下の測定)。
**測定 (2026-10-06, `--prefill-window 2048`)**

| プロンプト | 窓 | 所要 | tok/s | 窓内 GPU | うち PCIe 待ち |
| --- | --- | --- | --- | --- | --- |
| 1,064 | 1 × 1064 | 3.46 s | 308 | 3.40 s | 1.38 s |
| 2,047 | 1 × 2047 | 3.80 s | **539** | 3.69 s | 0.13 s |
| 4,095 | 2 × 2048 | 7.66 s | 534 | 3.70 + 3.83 s | 0.12 s |

2048 トークンの窓では転送が計算 (3.7 s) の裏に隠れ、expert MMQ (1.84 s: Q2_K が密で 14、分散ルーティングで 9 TMAC/s) が最大項。
残り: KDA scan 0.37 (逐次)、KDA qkv 0.27、mHC 0.17、DSA attention 0.17、KDA ゲート 0.16 (小さな GEMM と l2 norm)。
デバッグ: `STRATA_GLM5_SYNC=1` は MTP 窓・DSA 内部にも同期点、NaN 検査あり。

**並列 (バッチスロット, 2026-10-06)**: 上流 #465 と同じ 1 プロセス内のスロット方式。`--slots N` (設定 `"parallel": N`) で N 本の会話状態
(KDA 状態 + DSA キャッシュ + 位置 + マーク) を solo 用とは別に確保し、`forward_rows()` が各行 = 別スロットの 1 トークンを 1 窓で decode
(行ごとのカーネル: conv/scan は行のスロット状態に 1 歩、キャッシュ行はスロットの位置へ scatter、pool/score/attention は行ごとのベース)。
`copy_slot()` で会話をスロット間で移す (solo で始めた会話を STOP → BGEN で引き継ぐ、別スロットが持つ会話の次ターン)。serve は
`BGEN/BADM/BT/BDONE/BSTOP` を実装、`INFO batch_slots=N slot_cache=1`。一致性: `--slots-test K` で K 会話を単独/一括で decode して比較、
GPU のみのミス計算 (`STRATA_GLM5_CPU=0`) で 4 × 48 トークン **0 不一致** (CPU 計算のミスは ggml の丸めで同点が変わる: 上流も同じ注記)。
その過程で直したもの: ペア結合の順序を router の top-k 順に (hits 先の並びだと常駐状況で結果が変わる)、solo の pool 範囲が 1 プール
ずれていた (indexer が実際に選別する長さまで見えない)、リングのイベントを窓ごとの 2 面に (中規模窓で自分の record を待つ競合)、
`serve_ring` の非常駐 rank を毎回再計算 (swap 後に古い rank で詰まっていた)、doorbell スレッドの例外を本体に伝播。
DSA キャッシュを f16 に (スロット 1 本 1.46 → 0.89 GiB @32K; llama.cpp の既定 KV 型と同じ)。VRAM 内訳の表示 `glm5 vram:`。
性能: バッチ窓はミスのサービス (CPU+PCIe) が支配的で、4 行の窓 ≈ 125〜170 ms。合計スループットは solo (MTP あり) の 1.0〜1.2 倍、
効果は主に待ち時間 (全員同時に進む)。次: スロットでも MTP ドラフト (行数は 4 × 2 ≤ 8)、常駐率。

**スロットの MTP ドラフト (2026-10-06)**: `forward_rows()` が各スロットの連続する行の「走り (run)」を取る (行 = cur + ドラフト)。
KDA は走りごとに状態を逐次更新するグループ版カーネル (`k_kda_conv_grp`/`k_kda_step_grp`: 畳み込みのタップは走りの前の行か状態から、
行ごとのスナップショット) で、`commit_rows(n_keep[])` が採択された長さまで状態を巻き戻す (DSA キャッシュは位置で上書きされるので不要)。
MTP 層のキャッシュも全スロット分 (`mtp_window_slot/mtp_step_slot`、`copy_slot` は MTP 層もコピー)。decode 窓は 8 行固定 (スナップ
ショット・staging・行表) なので、同時 S スロットで 1 スロット `min(spec, (8-S)/S)` 本 (4 並列 → 1 本、2 並列 → 2 本)。
`--slots-test K` はドラフトなし/ありの両方を単独 greedy と比較: 4 × 48 で **0 不一致** (CPU=0)。

性能 (4 会話、合成プロンプト、ヒット率 73〜80%): 4 行窓 125 ms → 8 行窓 (1+1 ドラフト) 188〜200 ms、採択 79〜85%、合計 31.8 → 33〜36.6 tok/s
(+10%)。窓時間の内訳 (`STRATA_GLM5_PROF=1`): `moe_wait` 43 ms + `moe_down` 内の CPU 行待ち ≈ 15 ms = **約 60 ms がミスのサービス待ち**、
GPU の計算 (hits 21 + down 22 + その他 ≈ 65 ms) は大半がその陰。

**なぜサービスが遅いか = ホスト RAM の帯域**: expert ブロブ 9.2 MB、4 行窓あたり非常駐 335 (= 3.1 GB)。CPU 計算は 0.30 ms/ペア
= 30 GB/s で、スレッド 8/12/16 で不変 (`STRATA_GLM5_CPU_THREADS`)、CPU→DMA に寄せても (`GPU_BEFORE_CPU_MS=0.5`) 逆に遅い。
ホスト RAM の読み出しは実測 **42 GB/s** (4 スレッドで飽和、DDR4 デュアルチャネル) で、CPU 30 + PCIe DMA ≈ 10 でそこに張り付く。
solo も同じ: 1 層 4 ミス × 9.2 MB = 37 MB → 0.9 ms + GPU 0.7 ms ≈ 1.7 ms/層 × 45 = 78 ms/round (実測どおり)。
先読み (`--lookahead`) は帯域律速では外れ分が純損 (しかもスロット経路で予測未公開のまま busy-wait してハング: 使わない)。
→ 残る手は「トークンあたりのホスト RAM 転送量」: 常駐率。プリフェッチリングを 4 → 2 本 (`STRATA_GLM5_PF_BUFS=2`、2.8 GiB/台) にして
予算 30.8/28.8 (+669 expert) にすると 4 行窓 109 ms、8 行窓 174 ms = **38 tok/s 合計**、solo 56 → 53 ms/token。

**DSA キャッシュを Q8_0 で保持 (2026-10-06)**: 潜在 KV (512)、indexer key (128)、key-gate (128) の各行を ggml の Q8_0 ブロック
(32 値 = f16 スケール + int8 × 32 = 34 B) で格納 (`q8_block`/`q8_at`、`store_q8_rows`/`scatter_q8_rows`)。1 トークン 1 層あたり
1536 B (f16) → 816 B。decode のカーネル (score / ctx / pool、行モード含む) はブロックのスケール × int8 をその場で戻して計算。
prompt 窓の密な attention は以前からキャッシュを Q8_0 に変換して MMQ に掛けていたので、**キャッシュ自体を MMQ のオペランドに**
(変換 1 本が消える; 転置側 `q8_0_rows_T` は Q8_0 を読んで再量子化)。MMQ のタイル読み越し分としてキャッシュは max_ctx + 512 行。
スロット 1 本の状態 @32K: 0.89 → 0.47 GiB、5 本で 2.1 GiB 空く → expert 予算へ。精度: llama.cpp の `-ctk q8_0` と同じ丸め。

## 窓単位のスパース attention — `--attn lod` (2026-10-06)

GLM-5.3-Flash の DSA 層向けの実験的な attention モード。absorb 後の MLA (64 クエリヘッドが 512 次元の潜在 c を共有) に対して、
各クエリが固定量の exact セル (sink ページ、選んだページ、局所窓) だけを読むので、窓あたりの attention コストが文脈長に依存しない。
**基本モードでは k-pool / indexer (idx_q_b, idx_k, idx_gate, idx_proj, pool, score, select) を使わない。**
訓練時の DSA (indexer の top-2048) とは別の注意なので、精度はモデルごとに要評価。

- **ページ** (64 トークン) ごとの統計量 `pmean_` (Q8_0 行) を、ページが完成した時点で取る (`k_lod_page_mean[_rows]`)。
  キャッシュ行の純関数なので巻き戻し (mark / commit_rows) で状態を持たず、位置が戻れば再計算される。
- **選択** (`k_lod_select`、トークンごと): 候補 = 局所窓 (512) より前の完成ページ。sink (ページ 0) + 候補の上位 K + 局所窓の全セル +
  端数ページを exact セルとして `sel` に入れる。
- **prompt 窓** (T ≥ 9): 窓で選択を共有 (窓内のクエリからページごとに 1 つのスコア、MMQ と `colmax`)、[選択 | 窓自身のトークン] を
  圧縮 Q8_0 オペランドに gather (`k_lod_gather`) → 既存の MMQ 密 attention (因果マスク `k_lod_mask`、列バイアス付き `masked_softmax`)。
  列数 ≈ K·64 + 576 + T なので文脈長にほぼ依存せず、8192 のフォールバックも無い。
- パラメータ: `--attn lod`, `--lod-page 64`, `--lod-top 32` (prefill), `--lod-top-decode 32`, `--lod-local 512`, `--lod-sink 1`
  (環境変数 `STRATA_GLM5_LOD*`)。デバッグ: `STRATA_GLM5_LOD_SYNC=1` (窓単位の経路だけ段階同期; `STRATA_GLM5_SYNC=1` は doorbell と噛む)。
- 正しさの検査: 文脈 ≤ 2048 では DSA も密注意なので両モードのトークンが一致する: 1064 トークン + 24 生成で **24/24 一致**、
  `--slots-test 4` 0 不一致。
- 結果 (stage12、ring 2、予算 30.8/28.8):

  | プロンプト | DSA prefill | 窓単位 prefill | 生成 (32 トークン greedy) |
  |---|---|---|---|
  | 4095 | 7.71 s (531 tok/s) | 7.57 s (541 tok/s) | 31/32 一致 (「グラフィックメモリ」vs「グラフィックスメモリ」) |
  | 12285 | 35.7 s (344 tok/s) | **23.4 s (525 tok/s)** | DSA は英語で think 開始、窓単位モードは 4K と同じ日本語要約を出力 (どちらも妥当、優劣は未評価) |

  12K の窓時間は 3.65〜3.98 s で平坦 (DSA は 8K 超で 9.1 s)。attention は 0.43 s/窓で飽和、ページスコアの MMQ+colmax が 0.3 s/窓。
  12K ではドラフト採択 18/26 (DSA と同じ)。品質は NIAH / 長文 QA で要評価 (長文での選択精度が未知数)。
- top の掃引 (12285 トークン、attention バッファ上限を 16384 セルに: `STRATA_GLM5_ATTN_MAXKV`、既定 16384、+200 MB/台):

  | 設定 (top / top-decode) | prefill | attention/窓 | decode | 生成 32 トークン vs DSA |
  |---|---|---|---|---|
  | 32 / 32 | 23.4 s (525 tok/s) | 0.43 s | 22.7 tok/s | 別の応答 (think を飛ばして日本語要約) |
  | 128 / 32 | 25.3 s (486 tok/s) | 1.0 s | 24.3 tok/s | 先頭 13 トークン一致、以後は近い言い換え |
  | 128 / 128 | 25.4 s (484 tok/s) | 1.1 s | 17.9 tok/s | **32/32 一致** |

  decode は exact 行数 (128 ページ = 8192 + 局所 ≈ 9K 行/トークン) に比例して落ちていた (トークン単位カーネル)。
- **融合 decode カーネル** (`k_lod_attn_split` / `k_lod_attn_merge`、選択リスト上の split-K + online softmax カーネル、Strata の行構造の上に;
  MLA なので K と V は同じ Q8_0 行で 1 回読む、2 ヘッド/wave タイル、8 タイル/ブロック、`STRATA_GLM5_LOD_SPLIT` 既定 16、T ≤ 8 の窓で使用):
  12K top 128/128 の decode **17.9 → 25.8 tok/s** (DSA 24.8)、生成は DSA と 32/32 一致のまま; `--slots-test 4` 0 不一致、1064+24 も DSA と一致。
- 落とし穴: `cudaMemcpyAsync` を 2 台目のデバイスのストリームに積むときはそのデバイスを current に (既存の注意と同じ);
  行カーネルは MTP 層のスロット表も要る (DSA モードは MTP でキャッシュを直接使うので表が無かった)。

会話の継続 (4 ターン): 2 ターン目以降は「マーク」へ巻き戻して前回プロンプトぶんを再利用 (cached_tokens 24/37/66)、
24〜29 tok/s。残る床はラウンドあたり 110〜150 ms = 1 ラウンドのミス 40〜70 個 × 0.6 ms (PCIe) + 密な重み。

**MTP の採択率 (2026-10-04、283 トークンのプロンプト + 24 トークン)**: 6 ラウンド、ドラフト 18 個中 17 個採択 (94.4%)、
**4.00 トークン/ラウンド**。トークン列は greedy と完全一致 (投機は正確)。vLLM から転記した NextN の式は正しい。
最初の実装では decode 41.8 ms/token、MTP 1 ステップ 25 ms だった。グループ化カーネルの列数テンプレート化、全常駐層の同期省略、
MoE グループ化の並列化で MTP は約 2 ms/ステップに。窓あたりの時間の正体は上の表 (expert のミス転送) だった。

**測定 (2026-10-04、トークン列はすべて llama.cpp と一致)**

| 構成 | 常駐 expert | ヒット率 | decode ms/token | 読み込み |
| --- | --- | --- | --- | --- |
| 1 枚、常駐なし (毎トークン mmap から) | 0 | 0% | 378 | |
| 1 枚、33 GiB、id 順 | 3,996 (33%) | 33.7% | 220 | 9 s |
| 1 枚、33 GiB、同プロンプトのプロファイル (上限) | 3,866 | 82.6% | 182 | 9 s |
| 1 枚、33 GiB、別プロンプトのプロファイル | 3,866 | 36.8% | 211 | 9 s |
| **2 枚**、36+36 GiB、同プロンプトのプロファイル (上限) | 8,526 (70%) | 99.4% | **103〜135** | 20 s |
| **2 枚**、36+36 GiB、別プロンプトのプロファイル | 8,526 (70%) | 74.8% | **106** | 20 s |
| 2 枚、同上、**Sinkhorn 並列化 + expert 8 個を 1 起動** (後述) | 8,526 (70%) | 99.2% | **47〜57** | 20 s |
| 2 枚、同上、ミス元を層ごとにピン留め (28 GiB) した RAM に | 8,526 (70%) | 99.0% | **35〜40** | 107 s |
| 2 枚、同上、doorbell (ホスト同期なし) | 8,526 (70%) | 99.0% | **35〜43** | 94 s |
| 2 枚、34 プロンプトから集めたプロファイル (`--prompts`) | 8,566 (71%) | 89.8% / 76.5% | 67〜92 / 53〜77 | 94 s |

**doorbell 版のプロファイル (rocprofv3, 2 枚, 99% ヒット)**: 1 トークンのカーネル実行は GPU0 17.6 ms + GPU1 15.4 ms = 33 ms で、
壁時間 42 ms のうち GPU1 が GPU0 を待つ 21.8 ms は層分割の直列性そのもの。内訳の上位は Q8_0 の密 GEMV 11.3 ms
(9.6 GiB を毎トークン読む = W7900 の帯域の下限)、`moe_wait` 7.4 ms (ミス 1 個 8.26 MB の PCIe DMA 0.6 ms × 約 7 個/token)、
expert の一括カーネル 6.3 ms (帯域下限 3.3 ms)、起動の隙間 4〜5 ms/GPU (2,486 起動/token)。
このハードでの下限はおよそ 11 + 3.3 + 数 ms ≒ 16〜18 ms/token (55〜60 tps)。残る手段は (1) グラフ捕捉で起動の隙間を消す、
(2) 適応スワップでヒット率を 99% に保つ (プロファイル外のプロンプトでは 76〜90%、ミス 1 個 0.6 ms が効く)、
(3) 検証窓 (投機) で密な重みの読み出しを複数トークンで共有する。PCIe は 2 枚とも Gen4 x16、実測 14 GB/s。

**最初のプロファイル (2 枚、137 ms/token の時点、rocprofv3)**: カーネル実行の合計は 44 ms/token だけで、残りはホスト側
(カーネル起動 4,197 回/token = 起動 API だけで 20 ms、ホスト同期 52 回/token のたびに GPU のキューが空になる)。
カーネル側の上位は Q8_0 密 GEMV 11.3 ms、1 スレッド直列の Sinkhorn 9.8 ms、Q2_K expert 6.3 ms。
対策の第一弾: Sinkhorn を 1 warp (16 レーン + shuffle) に、expert 8 個の gate/up を 1 起動・down を 1 起動にまとめる
`native_mmvq_batched` (ポインタ表、単体呼び出しとビット一致) → 103〜135 ms が 47〜57 ms に。
注意: HIP では GPU 側で不正アクセスが起きた後の `hipStreamSynchronize` が永遠に戻らない (CPU 100% の「無限ループ」に見える) ので、
例外時は同期せず `_Exit` する。

283 トークン 1 本から作ったプロファイルは別のプロンプトに転移しない (36.8%)。34 本 (英文書、日本語、コード、チャット) から
集めたプロファイルで 76〜90%。ホストのピン留めは層ごとの `hipHostMalloc` (最大 4 GiB、約 0.7 s/GiB) なら問題なく、
1 回の 28 GiB は ROCm 7 で 10 分以上戻らなかった。ピン留めは `pread` でページキャッシュを汚さずに埋める。

**測定 (W7900 1 枚, bring-up 構成)**: 0.69〜0.75 s/トークン。expert は毎トークン mmap からページャブルコピーで VRAM に運び (8 × 8.26 MB × 42 層 = 2.8 GB/トークン)、DSA の top-k はホストで取っている。速度はこれから。

**一致の読み方**: 層ごとの差は活性量子化 (llama.cpp CPU は Q8_K/Q8_0、本エンジンと llama.cpp CUDA は Q8_1) の差で、
深い層では近接した expert のスコアが入れ替わる (層 21、40 で top-8 のうち 2 個が違う) ため `ffn_moe_out` のコサインが 0.97 まで
落ちるが、どちらの選択も正しい。この帯域は Unsloth UD-Q4_K_XL の検証 (docs/UNSLOTH_Q4.md) と同じ。

再現手順:

```sh
# llama.cpp (HIP) と dump_tensors
cd ../llama.cpp && cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF && cmake --build build-hip --target llama-cli -j14 && cd - && tools/glm_oracle/build.sh
../llama.cpp/build-hip/bin/dump_tensors -m GLM.gguf -o oracle -p "The capital of France is" -ngl 0 --ubatch 1 \
  --n-predict 2 --filter '^l_out-' --filter '^result_output'
# この repo の HIP ビルドと bring-up エンジン
cmake -S . -B build-hip -G Ninja -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_GGML_DIR=<llama.cpp checkout> && cmake --build build-hip --target strata-glm5 native_q2k_parity -j14
build-hip/strata-glm5 --gguf GLM.gguf --tokens $(paste -sd, oracle/tokens.txt) --n-predict 2 --dump engine
# 2 GPU, 36 GiB of experts each, routing counts saved for the next start's profile
build-hip/strata-glm5 --gguf GLM.gguf --tokens ... --devices 0,1 --layer-split auto --expert-budget-gib 36 --profile-save prof.txt
python3 tools/glm_oracle/compare_dumps.py oracle engine --step 0 --token 4        # 層出力
python3 tools/glm_oracle/compare_dumps.py oracle engine --step 0 --token 4 --layer 3 --all
PYTHONPATH=../llama.cpp/gguf-py python ref/glm5.py --gguf GLM.gguf --compare oracle --greedy 2   # numpy 参照
```

---

## 1. モデルの比較

| | Qwen3.8-Flash-Next (今の Strata) | GLM-5.3-Flash (`glm5-next`) |
| --- | --- | --- |
| 総パラメータ / 活性 | 125B / 6B | 320B / 18B (`LLM_TYPE_320B_A18B`) |
| 層数 | 48 | **45** |
| 隠れ次元 `n_embd` | 2,560 | **4,096** |
| 語彙 | 248,320 | **154,880** (`output`, `token_embd` ともに Q8_0, 0.67 GB ずつ) |
| 線形層 (再帰) | Gated DeltaNet ×36 (3 層に 1 回は QSA) | **Kimi Delta Attention (KDA) ×34** |
| 全アテンション層 | QSA ×12 (層 3,7,…,47) | **DSA (nope MLA + k-pool indexer) ×11** (層 3,7,…,43) |
| 位置符号化 | M-RoPE 64 次元, base 1e7 | **なし** (`rope.dimension_count = 0`, nope-only MLA)。位置情報は KDA の conv と再帰のみ |
| 残差 | 低ランク hyper-connection (hc=4, lr=320, 要素ごと sigmoid ゲート) | **mHC** (hc=4, 24 本の mix から pre/post/comb, comb は **Sinkhorn 20 回反復**) |
| 密 FFN 層 | なし | **層 0〜2 の 3 層** (ff 12,288, Q8_0, 1 層 160 MB) |
| MoE 層 | 48 層 × 512 experts, top-10, softmax ルータ | **42 層 × 288 experts, top-8**, **sigmoid ルータ + 選択バイアス `exp_probs_b`**, 重みを正規化して **×2.5** |
| expert の形 | 2560 → 640 → 2560 | **4096 → 2048 → 4096** (1 expert = 25.2M パラメータ) |
| expert 総数 | 24,576 | **12,096** |
| 共有 expert | 1 本, スカラーゲート `ffn_gate_inp_shexp` 付き | 1 本, **ゲートなし** (単純加算) |
| SwiGLU | silu(g)·u | **clamp 付き**: `silu(min(g,10)) · clamp(u,−10,10)` (expert, 共有 expert とも limit 10) |
| PLE (n-gram 埋め込み, SSD 28.8 GB) | あり (層 1) | **なし** |
| MTP ドラフト層 | あり (別途 BF16 から取得) | この GGUF には **nextn テンソルなし**。llama.cpp 側も `DECODER_MTP` 未実装 |
| コンテキスト | 262,144 | **1,048,576** |
| トークナイザ | gpt2 BPE, pre `qwen35` | gpt2 BPE, pre **`glm4`** (llama.cpp では `CHATGLM4` 正規表現 + `ignore_merges = true`) |
| チャットテンプレート | Qwen (`<think>`, `<tool_call>` XML) | `[gMASK]<sop>` 始まり, `reasoning_effort` は low/high/max |

### 1.1 GGUF の中身 (量子化と容量)

| テンソル群 | 型 | 容量 |
| --- | --- | --- |
| routed experts (gate/up/down) 36 層 | **Q2_K** | 85.6 GB (1 expert 8.26 MB) |
| routed experts 3 層 | Q4_K | 12.2 GB (1 expert 14.2 MB) |
| routed experts 3 層 | Q3_K | 9.3 GB (1 expert 10.8 MB) |
| **experts 合計** | | **107.2 GB** |
| KDA の q/k/v 射影 (`attn_q/k/v`, 4096→8192) | Q8_0 | 3.6 GB |
| `attn_output` (8192→4096, 45 層) | Q8_0 | 2.0 GB |
| 共有 expert (42 層) | Q8_0 | 1.1 GB |
| 密 FFN (3 層) | Q8_0 | 0.48 GB |
| MLA 射影 (`attn_q_a` Q8_0, `attn_q_b` Q8_0, `attn_kv_a_mqa` BF16, `attn_k_b`/`attn_v_b` BF16) | | 0.78 GB |
| indexer (`indexer.attn_q_b` 等, F32) | F32 | 0.33 GB |
| KDA のゲート射影 (`ssm_f_a/f_b/g_a/g_b/beta`, BF16), conv, norm | | 0.25 GB |
| mHC (`hc_attn_fn`, `hc_ffn_fn` 16384×24, F32) | F32 | 0.14 GB |
| ルータ (`ffn_gate_inp` F32) | F32 | 0.20 GB |
| `output` + `token_embd` | Q8_0 | 1.35 GB |
| **非 expert 合計** | | **10.3 GB** |

Strata の既存の GPU expert カーネル (`native_mmvq`) は Q3_K/Q4_K/Q5_K/Q6_K を持つが **Q2_K を持たない**。
CPU 側は ggml-cpu の `vec_dot` が全型をカバーする (ただし AVX2 の多トークンカーネル `kq_avx2` は Q4_K/Q5_1/Q8_0 のみ)。
prefill の MMQ も Q2_K/Q3_K のインスタンスを追加する必要がある (ggml の `mmq.cuh` にはある)。

### 1.2 トークン 1 個あたりの重み読み出し量

| | Qwen (Q2_0) | GLM (3.0bit) |
| --- | --- | --- |
| routed experts | 10 × 48 × 1.38 MB = **0.66 GB** | 8 × 42 × 8.26 MB (Q2_K 層) ≒ **2.8 GB** |
| 密な部分 (毎トークン) | ≒ 3.5 GB | ≒ 10 GB |

GLM は expert の読み出しが Qwen の 4 倍以上。Strata の設計 (GPU に乗らない expert を CPU で計算) をそのまま
使うと、AVX2 8 コアの CPU が Q2_K を 1 トークンあたり数百 MB も計算することになり、ここが最大のボトルネックになる。
開発機では **2 枚の W7900 (96 GB) に expert の 7〜8 割を常駐させ、ミス分は PCIe DMA で GPU に運んで GPU で計算する**
のが正しい配分 (推定; 第 4 節)。

---

## 2. 1 層の計算 (llama.cpp `glm5-next.cpp` から)

毎層、アテンション半分と FFN 半分がそれぞれ mHC で包まれる。

```
R: [n_embd=4096, hc=4]  (4 本の残差ストリーム)

--- アテンション半分 ---
x, post, comb = hc_pre(R, hc_attn_fn, hc_attn_scale, hc_attn_base)
x = rms_norm(x) * attn_norm
y = KDA(x)  または  DSA(x)
R = hc_post(y, R, post, comb)

--- FFN 半分 ---
x, post, comb = hc_pre(R, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x = rms_norm(x) * ffn_norm
y = dense_ffn(x)  (層 0..2)   または   moe(x) + shared_expert(x)
R = hc_post(y, R, post, comb)

--- 出力 ---
h = mean over 4 streams of R
logits = output @ rms_norm(h) * output_norm
```

### 2.1 mHC (DeepSeek V4 流 hyper-connection)

`ggml_dsv4_hc_comb / hc_pre / hc_post` の定義 (`ggml/include/ggml.h:2680`):

```
flat  = R を 16384 次元に並べて rms_norm(eps=1e-6)       ← 重みなしの RMSNorm
mixes = hc_fn @ flat                                       ([24])   F32 の 16384×24 行列
pre [h]      = sigmoid(mixes[h]      * scale[0] + base[h])      + eps      (h = 0..3)
post[h]      = sigmoid(mixes[4+h]    * scale[1] + base[4+h])    * 2
comb[d,s]    = mixes[8 + d + 4s] * scale[2] + base[8 + d + 4s]  → softmax over d, +eps,
               列正規化, (行正規化, 列正規化) × 19 回        ← Sinkhorn 20 反復
x[i]         = Σ_h R[i,h] * pre[h]                          (hc_pre)
R'[i,d]      = y[i] * post[d] + Σ_s R[i,s] * comb[d,s]      (hc_post)
```

Qwen の GR (`gr_read`/`gr_write`: 低ランク 320 のボトルネック、要素ごとの sigmoid ゲート、inject) とは
計算が全く別物。Strata の `fused_gr.cu` は再利用できず、**新しい fused カーネル 2 本** (hc_pre+norm, hc_post) になる。
計算量は小さい (16384×24 GEMV + 4×4 の Sinkhorn)。

### 2.2 KDA (Kimi Delta Attention) 層 ×34

```
q = silu(conv1d_q(attn_q @ x))     k = silu(conv1d_k(attn_k @ x))     v = silu(conv1d_v(attn_v @ x))
    ※ q/k/v が別々の 8192 ch 射影 + 別々の conv (Qwen は 10240 ch をひとつの conv)
    64 heads × 128 dim, h_k == h_v == 64 (GQA なし)
q, k = l2_norm(q), l2_norm(k)      (eps 1e-6)
g1 = f_b @ (f_a @ x) + dt_bias                       ([8192] = 64 heads × 128)
g1 = gate_lower_bound * sigmoid(-(g1 * A))           (lower_bound = -5, A = ssm_a[h])   ← 要素ごとの減衰
beta = sigmoid(beta_w @ x)                            ([64])
再帰 (ggml_gated_delta_net の kda 分岐, ggml-cpu/ops.cpp:11123):
    S[i,:] *= exp(g1[i])            ← Qwen はヘッド単位のスカラー、KDA は i ごとのベクトル
    d[j]   = (v[j] - Σ_i S[i,j] k[i]) * beta
    S[i,j] += k[i] d[j]
    o[j]   = Σ_i S[i,j] q[i] / sqrt(128)
g2 = g_b @ (g_a @ x)                                 ([8192])
y  = rms_norm(o, ssm_norm) * sigmoid(g2)
out = attn_output @ y                                 (8192 → 4096)
```

状態: 128×128×64 = 1M float (4 MB/層, 34 層で 142 MB)、conv 状態 3 × 3 × 8192。
Strata の `gdn_step` との差: (1) 減衰がベクトル、(2) 3 本の conv、(3) ゲートの式、(4) GQA なし。
`gdn.hpp` の「1 スレッドが (j,h) 列を持つ」並列化はそのまま使える。**新カーネル `kda_step` (単トークン / 検証窓用の
multi) と prefill 用のチャンク再帰** が必要。

### 2.3 DSA (DeepSeek Sparse Attention, nope MLA) 層 ×11

```
qr  = rms_norm(attn_q_a @ x) * attn_q_a_norm        ([1536])
q   = attn_q_b @ qr                                  ([64 heads × 256])
kvc = rms_norm(attn_kv_a_mqa @ x) * attn_kv_a_norm   ([512])  ← これが KV キャッシュの全部 (トークンあたり 512 値)
q_abs[h] = attn_k_b[h] @ q[h]                        ([64 heads × 512])   wk_b を q に吸収

--- indexer (k-pool) ---
iq  = indexer.attn_q_b @ qr                          ([32 heads × 128])
ik  = layer_norm(indexer.attn_k @ x) (weight+bias)   ([128])
ig  = indexer_compressor_gate @ x                    ([128])
w   = (indexer.proj @ x) / sqrt(128*32)              ([32])
キャッシュ行 = [ik | ig | pooled] の 3×128
4 トークンが揃うごとに 1 プール:
    pooled[d] = Σ_{j<4} softmax_j(ig_j[d] + ape[d,j]) * ik_j[d]      ← 次元ごとに 4 位置の softmax
score[pool] = Σ_h relu(iq_h · pooled) * w_h
上位 512 プール (= 2048 セル) + 未完成の末尾 (最大 3 セル) を選択
--- attention (MQA, 64 query heads が同じ 512 次元 latent を見る) ---
p   = softmax(q_abs[h] · kvc[selected] / sqrt(256))
o[h]= attn_v_b[h] @ (Σ p * kvc[selected])            (512 → 256)
out = attn_output @ concat(o)                         (64×256 = 16384 → 4096)
```

Qwen の QSA との対応: 「圧縮率 4 のブロックを indexer で採点して上位 2048 セルを選ぶ」という骨格は同じ。
Strata の `qsa_select` (block scores → top-k) と `kv_gather` の設計はそのまま持ち込める。違うのは
(1) プーリングの式 (QSA: ブロック完成時の pooling / DSA: ゲート付き softmax プール)、(2) K と V が同一の
512 次元 latent で RoPE なし (キャッシュが小さい: 11 層 × 512 × fp16 = 11 KB/トークン、262K で 2.9 GB)、
(3) ヘッドごとの `wk_b` 吸収と `wv_b` 展開、(4) スコアが 32 ヘッドの relu 和。

### 2.4 MoE ルータ (`llama-graph.cpp:2011` `build_moe_ffn`)

```
logits = ffn_gate_inp @ x                   (F32 重み)
probs  = sigmoid(logits)
sel    = probs + exp_probs_b                ← 選択は バイアス込み、重みは バイアスなし
ids    = top-8 of sel
w      = probs[ids];  w /= max(Σw, 2^-14);  w *= 2.5
out    = Σ w_e * down_e( silu(min(gate_e,10)) * clamp(up_e,-10,10) )  +  shared(x)
```

Strata の `router_top10` (softmax → top-k → 正規化) は **sigmoid+bias 版**を追加する必要がある。
expert カーネル (GPU `native_mmvq` の gate/up 融合と CPU `native_gu_rows`) に **clamp** を入れる。

---

## 3. Strata で再利用できるもの / 新規に必要なもの

| 領域 | 再利用 | 新規 / 改修 |
| --- | --- | --- |
| GGUF 読み込み (`gguf_reader`, `gguf_split`) | ○ そのまま | glm5-next のキー読み取り |
| パック形式 (`index.txt`, `native_experts.txt`, `experts.bin`) | ○ 形式は汎用 (層ごと可変 blob, `n_expert` はヘッダ) | `tools/iq_pack.py` に glm5-next の層構成 (dense 3 層, 288 experts, Q2_K/Q3_K/Q4_K blob) |
| `ModelGeometry` (`layout.hpp`) | × 数値がハードコード | **モデル記述を GGUF メタデータから構築**し、Qwen/GLM の 2 系統を持つ |
| `LayerType {GDN, QSA}` (`graph.hpp`) | 拡張 | `{GDN, QSA, KDA, DSA}` + dense/MoE フラグ |
| 残差 `gr.cu` / `fused_gr.cu` | × | **`mhc.cu`**: hc_pre(+rms_norm) と hc_post(+Sinkhorn), 単トークン/多トークン |
| `gdn.cu` / `fused_gdn.cu` / prefill `gdn_conv`/`gdn_recurrence` | 設計のみ | **`kda.cu`** (ベクトル減衰, 3 conv), prefill 用チャンク再帰 |
| QSA (`qsa_select`, `kv_gather`, `qsa_decode_attn`, `qsa_prompt_attn`) | 選択と gather の骨格 | **`dsa_*`**: k-pool プーリング, 32 ヘッド relu スコア, latent (512) 上の MQA, `wk_b` 吸収 / `wv_b` 展開 |
| KV キャッシュ (fp16/int8/q4, streaming) | ○ ページ表と streaming は流用 | latent 1 本の形に合わせる (K=V, RoPE なし) |
| ルータ `router_top10` | 拡張 | sigmoid + bias + 正規化 + scale |
| 共有 expert `shared_expert` | 拡張 | スカラーゲートなし版 + clamp |
| GPU expert カーネル `native_mmvq` (Q3_K/Q4_K) | ○ | **Q2_K を追加** (ggml-cuda vecdotq から転記), clamp |
| 多トークン expert (`native_expert_grouped`) | ○ (型追加) | Q2_K |
| CPU expert (`native_expert.cpp`, ggml-cpu) | ○ | Q2_K/Q3_K の AVX2 多トークンカーネル (任意, 性能) |
| expert 階層 (`expert_source`, `expert_cache`, pinned arena, プロファイル, 適応スワップ, peer, layer split) | ○ ほぼそのまま (n_layers×n_expert は汎用) | 密 FFN 層 (expert なし) をスキップできるように |
| prefill (`prefill.cpp` のチャンク/ストリーミング/MMQ) | ○ 骨格 | 層の計算を GLM に差し替え, MMQ に Q2_K/Q3_K, mHC/KDA/DSA の多トークン版 |
| 検証窓 `verify.cpp` / `verify_kernels.cu` | ○ 骨格 | 新カーネルの multi 版 |
| MTP `mtp.cpp` | × (テンソルがない) | 当面は `--spec 0` + prompt lookup (`suffix_drafter`)。後で nextn を HF から取得 |
| PLE `ple*`, `ngram` | 不要 | GLM では無効化 |
| サーバ `serve/server.py` | ○ | テンプレート (`[gMASK]<sop>`, reasoning_effort の値), `OutputParser` の tool 形式 |
| トークナイザ `tools/strata_tokenizer.py` | ○ 骨格 | `glm4` 正規表現 + `ignore_merges` |
| `setup.py` | ○ | GLM ファミリの追加 (後回し) |

---

## 4. 開発機でのメモリ配分 (推定)

| 置き場所 | 内容 | 容量 |
| --- | --- | --- |
| VRAM (2 × 48 GB) | 密な重み 10.3 GB (layer split なら各カードに自分の層分だけ) | ≒ 10 GB |
| | KV latent + indexer キャッシュ (262K, fp16) | ≒ 4 GB |
| | KDA 状態, 作業バッファ, prefill 借用分 | ≒ 4〜8 GB |
| | **expert キャッシュ** | **≒ 70〜75 GB → 12,096 個中 8,500〜9,000 個 (約 70%)** |
| RAM (125 GB) | expert 全部を pinned (107 GB) は **他と合わせて入らない** | → 低 RAM モード (`--resident-experts`: GPU に無い 3 割 ≒ 32 GB だけ RAM に常駐) |
| SSD | GGUF 本体 (mmap) | 117 GB |

ヒット率はプロファイルで 70% 以上、適応スワップで更に上がる見込み。ミスした expert (8.26 MB) は
PCIe 4.0 x16 (≒ 26 GB/s) で 0.3 ms。1 トークンあたり 8×42 = 336 expert のうち 2〜3 割 ≒ 80 個がミスとして
DMA で 0.66 GB → 約 25 ms。CPU (AVX2 8 コア, Q2_K) で同じ量を計算するより DMA の方が速いので、
**GLM では `--pcie-frac` を 1.0 近く (ミスは全部 GPU で計算) が既定になる**と予想する。
いずれにせよ測ってから決める。

---

## 5. 実装計画 (フェーズ)

各フェーズは「llama.cpp と同じ入力で同じ数値が出る」ことを合格条件にする。Strata の流儀どおり、
基準 (oracle) に対する parity テストのないカーネルは入れない。

### P0. オラクル環境 (1〜2 日)

- llama.cpp を HIP (gfx1100) でビルドし、GLM-5.3-Flash を 2 GPU + CPU で動かす (`-ngl` 分割)。これが
  トークン列と中間テンソルの基準になる。
- `llama-eval-callback` (または `GGML_SCHED` デバッグ) で、短いプロンプトについて層ごとの `hc_attn_pre`,
  `attn_norm`, `kda_scan_out`, `indexer_score`, `kq_soft_max`, `ffn_moe_out`, `l_out` をダンプするスクリプトを
  `tools/glm_oracle.py` として用意する。
- `ref/glm5.py`: numpy で 1 層を素直に書いた参照 (mHC, KDA 1 ステップ, k-pool, MoE ルータ)。llama.cpp の
  ダンプと一致させる。Strata の `ref/gdn.py` 等が果たしていた役割。

### P1. モデル記述の一般化とローダ (3〜5 日)

- `ModelGeometry` を「GGUF メタデータから埋める構造体」に変え、`arch` (qwen4exp / glm5-next) と
  層種別の配列 (`is_recr`, `is_dense`, `has_indexer`) を持たせる。Qwen の挙動は変えない
  (既存の parity テストと `--serve` のゴールデンで確認)。
- `tools/iq_pack.py` に glm5-next を追加: 密テンソルは GGUF の型のまま (`--native` 相当)、experts は
  `native_experts.txt` v4 で層ごとに Q2_K/Q3_K/Q4_K の blob (gate|up|down)。`experts.bin` は作らず
  GGUF をその場で読む (低 RAM モードの流儀)。
- `tools/strata_tokenizer.py`: `glm4` pre (CHATGLM4 正規表現, `ignore_merges`) を追加し、llama.cpp の
  `llama-tokenize` と往復一致を確認。
- 新しい `strata-glm` の起動が「重みを読み、形を検証し、layer 0 の mHC まで走る」ところまで。

### P2. カーネル群 (2〜3 週間, 並列可能)

それぞれ `src/kernels/*_parity.cpp` を伴う。

| カーネル | 内容 | 基準 |
| --- | --- | --- |
| `mhc.cu` | `mhc_pre` (flat rms_norm → 24 mix → pre/post/comb(Sinkhorn) → x), `mhc_post` | `ref/glm5.py`, ggml-cpu `dsv4_hc_*` |
| `kda.cu` | q/k/v 3 本の conv + silu, l2 norm, ゲート (lower bound 版), `kda_step` (ベクトル減衰), 出力 norm × sigmoid(g2) | ggml-cpu `gated_delta_net` (kda 分岐) |
| `dsa_indexer.cu` | ik の layer_norm, ig, k-pool プーリング (4 トークン完成時), 32 ヘッド relu スコア + 末尾 | `ref/glm5.py` |
| `dsa_attn.cu` | `wk_b` 吸収, 選択セル上の MQA softmax, `wv_b` 展開 (単トークン / 検証窓) | ggml 非 fused 経路 |
| `router_sigmoid` | sigmoid + bias top-8, 正規化, ×2.5 | `build_moe_ffn` |
| `native_mmvq` Q2_K | ggml-cuda の `vec_dot_q2_K_q8_1` 転記 + swiglu clamp | ggml-cuda |
| `shared_expert` 変種 | ゲートなし + clamp | ggml |
| dense FFN | 既存の Q8_0 GEMV + clamp なしの silu | ggml |

HIP での注意: `qsa_select` の tf32 mma と同じく、W7900 (gfx1100) には WMMA bf16/f16 があるが、まずは
FP32 の素直な版で一致を取り、速度は後。

### P3. 単トークン decode の貫通 (1〜2 週間)

- `layer.cpp` に `kda_layer`, `dsa_layer`, `dense_ffn_layer`, `mhc` の pre/post を追加し、`block_layer_pre/post`
  を層種別で分岐。`session.cpp` の doorbell と CPU プールへの引き渡しはそのまま。
- まず `--no-capture` (グラフなし) で、expert は全部 CPU (`--no-pool` 逆の構成でもよい) で、llama.cpp と
  **greedy で同じトークン列**を出す。この時点で速度は問わない。
- 次に CUDA/HIP グラフ捕捉 (`session_capture_token`) を通す。

### P4. expert 階層の統合 (1 週間)

- プロファイル作成 (`--dump-routing` → `tools/make_profile.py`、`n_expert=288` に対応)。
- expert キャッシュ (層ごと可変 blob)、PCIe DMA 分担、低 RAM `--resident-experts`、2 GPU の layer split
  または peer tier を GLM で動かす。密 FFN 層 (expert なし) をスキップ。
- 開発機で VRAM 70 GB 超のキャッシュを張り、ヒット率と tok/s を測る。

### P5. prefill (2 週間)

- `prefill.cpp` のチャンクループに GLM の層計算を載せる: mHC の多トークン版、KDA のチャンク再帰
  (`gdn_recurrence` の KDA 版)、DSA の prompt attention (チャンク内 causal + 選択)、k-pool のバッチ構築。
- MMQ に Q2_K/Q3_K のインスタンス (`STRATA_MMQ_KQUANTS` 相当のオプション)。
- 32K プロンプトで llama.cpp と logits を比較 (許容差は fused 経路と同じ基準で文書化)。

### P6. 検証窓と投機 (1 週間)

- `verify.cpp` に新カーネルの multi 版を接続。MTP がないので `suffix_drafter` (prompt lookup) のみで
  `--spec` を動かす。
- 後日: GLM-5.3-Flash の nextn (MTP) テンソルを HF の safetensors から `tools/mtp_fetch.py` 流に取得し、
  `mtp.cpp` の GLM 版 (`eh_proj`, `enorm`, `hnorm`, 1 層) を作る。これは 1.5〜1.8 倍の価値がある。

### P7. サーバとツール (数日)

- `serve/chat_template.jinja` 相当は GGUF 内のテンプレートをそのまま使う。`frontend.py` の `EFFORT` 写像
  (GLM は low/high/max) と `OutputParser` の tool-call 形式を確認。
- `setup.py` に `glm` ファミリ (ダウンロード元、RAM 要件、`--kv` 既定) を追加。

### P8. 性能 (継続)

- 測って効く順に: GPU で計算するミス expert の割合、Q2_K の多トークン GPU カーネル (検証窓)、hipBLASLt 表、
  KDA の fused 化、DSA 選択の WMMA 化、prefill の fused int8 (`moe_fused_iq` の k-quant 版)。
- 数字は `docs/DETAILS.md` の流儀 (機械・設定・エンジン版を併記、推定は推定と書く) で残す。

---

## 6. 検証の基準

- **トークン列**: greedy で llama.cpp と一致 (短・中・長プロンプト、日本語/英語/コード)。
- **カーネル単体**: parity テストで、基準との最大絶対誤差と、一致しない場合はその理由を各ヘッダに明記
  (Strata の既存ルール)。ビット一致を求める箇所: KDA 再帰 (累積誤差が出るため)、ルータ (選択が変わるため)。
- **expert の算術**: 同じ bytes に対する GPU/CPU の結果が ggml の `vec_dot` と一致 (Strata の `native_expert_parity`
  に Q2_K を追加)。

## 7. リスク

| リスク | 対応 |
| --- | --- |
| expert 1 個 8 MB で CPU 計算が間に合わない | GPU 常駐率を上げる (2 GPU)、ミスは DMA で GPU 計算。Q2_K の GPU 多トークンカーネルを優先 |
| RAM 125 GB に experts が入らない | 最初から `--resident-experts` (GPU 補集合だけ RAM) を既定にする |
| Sinkhorn 20 反復の数値一致 | ggml-cpu の実装を一行ずつ転記、F32、反復順を固定 |
| k-pool の末尾/キャッシュ更新のタイミング | llama.cpp `set_input_kpool` のセマンティクスを `ref/glm5.py` に先に写す |
| HIP 固有の差 (`hipErrorInvalidValue` 等) | Qwen で踏んだ回避策 (`hip_compat`) を流用、gfx1100 でのみ検証 |
| nextn (MTP) が無く投機が効かない | P6 の後半で HF から取得。それまでは prompt lookup のみ |

## 8. 最初の一歩 (次の作業)

1. llama.cpp を HIP でビルドし、GLM-5.3-Flash が動く設定 (`-ngl`, `--split-mode`) を見つけてトークン列を保存する。
2. `ref/glm5.py` を書き、`llama-eval-callback` のダンプと層 0〜3 (dense FFN と最初の KDA) で一致させる。
3. `ModelGeometry` の一般化 (P1) に着手する。Qwen 側の回帰は既存テストで守る。

---

## MTP (NextN) 層の仕様 (2026-10-04 確定)

ファイル: `/home/client/Projects/llm/GLM-5.3-Flash-MTP-Q4_K.gguf` (4.58 GB, `blk.45`)。中身は DSA 層 1 枚 (nope MLA、indexer 付き、
`index_share_mtp = true`) + 288 experts top-8 (gate/up Q4_K、down Q5_K、4.4 GB) + 共有 expert + `nextn.eh_proj` (8192→4096, Q8_0)、
`nextn.enorm`、`nextn.hnorm`、`nextn.shared_head_norm`。`embed_tokens` と `shared_head_head` は無く本体の `token_embd` / `output` を使う。
**`hc_*` は無い** (HF の safetensors index でも layer 45 に hyper-connection の重みは存在しない)。

オラクル: llama.cpp は glm5-next の MTP グラフ未実装 (`"GLM5-Next NextN graph not implemented yet"`)、HF transformers にも MTP クラス無し。
計算式は vLLM の実装 (`ref/vllm_glm5next_mtp.py.txt`、`ref/vllm_glm5next_model.py.txt`、ユーザー提供) から確定した。検証は採択率で行う。

```
h_prev   = output_norm( mean_streams(R_45層後) )            本体の result_norm と同じテンソル (vLLM: hc_contract = 平均, その後 self.norm)
e        = token_embd[tok_{p+1}];  位置 0 では e = 0         (vLLM fused_eh_norm: pos == 0 の埋め込みをゼロにしてから norm)
x        = eh_proj( concat( rms_norm(e)*enorm, rms_norm(h_prev)*hnorm ) )      eps = rms_eps、順序は [enorm 側 | hnorm 側]
a        = DSA_45( rms_norm(x)*attn_norm )                   ← mHC なし、普通の pre-norm 残差 (vLLM: is_mtp_layer → "KDA + MoE without HC" 分岐)
x1       = x + a
m        = MoE_45( rms_norm(x1)*ffn_norm ) + shared_expert   (sigmoid ルータ + bias、正規化 ×2.5、clamp 10: 本体と同じ)
h_next   = rms_norm(x1 + m) * shared_head_norm               ← ドラフト logits にも、次のドラフトステップの h_prev にもこれをそのまま使う
logits   = output @ h_next                                   (本体の出力ヘッド、再 norm なし)
```

ドラフトを複数ステップ連ねるときは h_prev ← h_next、tok ← argmax(logits) で繰り返す。vLLM は indexer の top-k をステップ 0 だけ計算し
以降は再利用する (`index_share_for_mtp_iteration`)。MTP 層は自前の latent KV と indexer キャッシュを持つので、プロンプト全位置について
(h_p, tok_{p+1}) を流す MTP 用 prefill が要る (Strata 本体の `draft_kv` に相当)。

実装メモ: expert 4.4 GB は VRAM 常駐 (`native_mmvq` の Q4_K/Q5_K と一括版がそのまま使える)。採択 2.5 トークン/パスなら
密な重み 11 ms を 2.5 トークンで割れるので実効 15〜18 ms/token (55〜65 tps) が見込み (推定)。

**GLM の thinking オフ (2026-10-06)**: pack の `chat_template.jinja` (GGUF 由来) は生成プロンプトを常に `<|assistant|><think>` で終えるため、
`reasoning_effort: none` / `enable_thinking: false` でもモデルが思考を始め、サーバはそれを本文として返していた。pack のテンプレートを
`enable_thinking is false` なら `<think></think>` を出すよう修正 (`~/Projects/Strata-data/glm53/pack/tokenizer/chat_template.jinja`、元は `.orig`)。
`reasoning_effort: low/high` はテンプレートの `Reasoning Effort:` 行で元から有効。サーバの done 行に停止トークン id を出すようにした。

## solo の高速化 (2026-10-06 夜、進行中)

**prompt 窓の chunk 化 (`--prefill-window` 最大 8192)**: prefill は 2048 トークン窓で PCIe (非常駐 42 GB/窓 ≈ 3.0 s) と MMQ 計算
(≈2.5 s) の両方が床に近い。窓を広げても窓あたりの転送量は同じなので、4096 窓ならトークンあたりの PCIe は半分になる。エンジンは
`t_max_` (窓) と `t_chunk_ = min(t_max_, 2048)` (計算 chunk) を分け、層ごとに窓を chunk 単位で回す (スクラッチは chunk サイズ、
残差ストリーム R / bounce / h_win / tok_rows だけ窓サイズ)。リングバッファの解放と次層の prefetch は窓の最後の chunk の後だけ
(`moe(..., last_chunk)`)。MTP の追従窓も chunk で回す。VRAM: R が 2 倍 (+134 MB/台 @4096)。

**decode の先読み (lookahead) の再設計**: 旧実装 (`--lookahead P`) は (1) prompt 窓で予測を出さないのに待ってハング、(2) `prefetch()` の
フラグが書かれない、(3) 追い出しがプール前提 (exchange swap 後はホスト側コピーを失いファイル読みに落ちる: 4.3 tok/s) の 3 点で死んでいた。
exchange swap で常駐化する版は往復 18 MB/expert の PCIe がラウンドあたり 1.9 GB になり 19.6 tok/s (基準 30.8)。現在の版は
**staging への片道先読み** (`prefetch_stage`: 予測上位の非常駐 expert を層あたり `STRATA_GLM5_PF_CAP` 本 (既定 3) だけ staging の
上半分へ DMA、サービス時に当たった分は staging 内 d2d コピーで使う、常駐は変えない)。

結果: staging 片道版も効かない (cap 3: 27.6 tok/s、cap 2: 30.3、なし 30.7)。層あたり 2〜3 本 (18〜28 MB) の先読みが PCIe 14 GB/s で
1.3〜2 ms かかり、層の処理時間 (≈1.7 ms) を超えて次層のフラグを遅らせる。ホスト RAM 42 GB/s を CPU 側 (30 GB/s) と分け合う前提では
PCIe 先読みは CPU 計算に勝てない → **打ち切り** (`--lookahead` は既定 0 のまま; コードはハングを直した状態で残す)。
4096 窓も同様に撤回 (計算律速、上記)。

## 長文の検証: NIAH と 256K (2026-10-06 深夜)

ハーネス `scratchpad/niah.py` (リポジトリの docs を詰め物に、針 1 文を深さ 10/50/90% に置き、thinking オフで直答させる; CLI `--prompts`)。

| 文脈 | DSA | ページ平均ルーティング (page 64, top 128/64) | page 16 / top 512 |
|---|---|---|---|
| 8K | 3/3 (490 tok/s) | 3/3 (508 tok/s) | |
| 32K | 3/3 (273 tok/s) | 3/3 (445 tok/s) | |
| 64K | **3/3** (218 tok/s) | **0/3** (434 tok/s) | **3/3** (375 tok/s) |
| 128K | (計測中) | 0/3 (420 tok/s) | |
| 236K | | 0/3 (404 tok/s、動作自体は正常) | |

窓単位モードは 64K 以上で針を拾えず (「そんな合言葉は知らない」)、DSA は拾う。exact 量 (8192) を変えずにページを 16 に細かくすると 64K で
回復するので、原因は **64 トークン平均の希釈** (針 1 文が他 63 トークンに埋もれ、`max_h q·mean` で上位に来ない)。
対策として **トークン単位ルーティング**を実装 (`STRATA_GLM5_LOD_ROUTE=tok` 既定): クエリをヘッド群 (`STRATA_GLM5_LOD_QGROUPS`=8) で
平均した q̄_g と各トークンの潜在 c の内積の max をページの点にする (prefill は MMQ で T×G 行 × 全候補セル、列 16384 ずつ; decode は
ページごとのブロックでキャッシュを全読み)。要約 (ページ平均) はそのまま。コストは 128K で prefill 窓あたり ~0.1 s、decode ~1 ms/層。
256K: 236K トークンの prefill 583 s (404 tok/s、平坦)、decode 18 tok/s (CLI、ドラフトなし)、落ちなし。

結果 (トークン単位ルーティング、page 64, top 128/64, G=8): **64K 3/3 (440 tok/s)、128K 3/3 (405 tok/s)** — DSA (218 / 179 tok/s) と同じ正答で
2.0〜2.3 倍速い。ページ平均ルーティングは 0/3 だったので、長文の窓単位モードはトークン単位ルーティングが必須 (既定に)。

**128K prefill の内訳** (トークン単位ルーティング、2048 窓 × 58、平均 5.0 s/窓 = 410 tok/s): moe_hits 1.26 s (25%) + moe_down 0.70 (14%) =
expert MMQ 39%、`dsa_attend` 1.22 s (24%: top 128 で圧縮オペランド ≈12.4K 列、S 行列のパス)、`dsa_score` 0.43 s (9%: ルーティングの
MMQ + colmax、G=8)、KDA 0.85 s (17%)、mhc 0.17。100K prefill の伸びしろはこの順。

**expert MMQ (Q2_K) の調査**: `scratchpad/mmq_bench_new` (288 expert、2048 トークン、偏った分布) で Q2_K は J=16 で 9.3 TMAC/s、J=24〜48 で
1.5 (6 倍遅い)、J=64 で 8.0。Q4_K は J=24 で 15.9。`-Rpass-analysis=kernel-resource-usage` で見ると Q2_K の `mul_mat_q<J=32>` は VGPR 256 +
スクラッチ 1584 B/lane、J=48 は 2676 B (レジスタ溢れ)、J=16 は 8 B。RDNA3 の設定表 (`mmq-config-rdna3.cuh`) のスレッド数/I を変えても溢れは
消えない (MMA 版 vec-dot の構造)。Q2_K だけ dp4a 版に切り替える実験は未着手。現状の engine は Q2_K に J=8 (→ J=16) を使っており、これが上限。

**ルーティングの比較 (NIAH、針 3 本、top 128 / decode 64)** — モデルの訓練文脈長は 1M (rope なし):

| 文脈 | DSA | ページ平均 | トークン (G=8) | indexer ルーティング |
|---|---|---|---|---|
| 64K | 3/3 (218 tok/s) | 0/3 | 3/3 (440) | 3/3 (436) |
| 128K | 3/3 (179) | 0/3 | 3/3 (405) | 3/3 (408) |
| 236K | 3/3 (142) | 0/3 | 1/3 (365) | **2/3 (373)** |

top を下げる実験 (トークン版): 64K は top 32 でも 3/3 (509 tok/s、attention 0.47 s/窓)、128K は top 32 で 0/3、top 64 で 1/3 → 128K では top 128 が要る。
**indexer ルーティング** (`STRATA_GLM5_LOD_ROUTE=idx`): DSA の indexer 経路 (idx_q/k/gate/proj、ik/ig キャッシュ、k-pool) を窓単位モードでも維持し、
prefill は iq (T×32 行、128→256 にゼロ詰め) × プール鍵 (Q8_0) の MMQ → `Σ_h relu(S) w_h` の max → ページ max、decode は `dsa_score_rows` → ページ max。
コストは 128K で `dsa_score` 0.4 s/窓 (8%)、236K で 0.7 s (13%)。長文ではこれを既定にする。

DSA は 236K でも 3/3 (142 tok/s、1 プロンプト 28 分) → モデルは拾える。窓単位側の残差は選択粒度 (64 トークンページで top 64/128) で、
DSA は 4 トークンプールで top 512。exact 集合もプール単位にする版 (`STRATA_GLM5_LOD_ROUTE=pool`: indexer の上位 `--lod-top-pools`
プール (prefill 2048、decode 512) を exact、プールが 1 つも開かないページだけ要約) を実装、236K で検証中。

Q2_K MMQ の設定表の試行: J=32/48 を 256/2/128 や 128/1/64 にしても溢れは同じ (1.6〜2.7 KB/lane)、J=64 を I=64 にすると 6.7 TMAC/s (元の 8.0 より悪化)。
→ 設定表では直らない (MMA 版 vec-dot のレジスタ構造)。pinned ツリーは元に戻した。engine は Q2_K に J=16 (9.5 TMAC/s) のまま。

**結果 (pool 選択、2026-10-07)**: 64K 3/3 (438 tok/s)、**236K 3/3 (373 tok/s)** — DSA (142 tok/s) と同じ正答で 2.6 倍。indexer ルーティングの
128K は top 64 ページでも 3/3 (449 tok/s)、top 32 は 2/3。既定ルートを `pool` に (`STRATA_GLM5_LOD_ROUTE=pool`、prefill 2048 プール = 8192
トークン exact、decode 512 プール = 2048 トークン exact、要約は開かなかったページ)。MoE の Q2_K MMQ (窓の 39%) が残る最大の伸びしろ。

prefill のプール数 1024 (exact 4096 トークン) でも 128K 3/3 (448 tok/s)、236K 3/3 (**405 tok/s**、2048 プール: 373) → 既定を 1024 に
(`STRATA_GLM5_LOD_TOP_POOLS=1024`)。236K の窓 5.0 s: MoE 39%、attention 0.89 s (18%)、indexer 採点 0.71 s (14%)、KDA 17%。

- indexer 鍵の MMQ 幅 (`STRATA_GLM5_LOD_IDXPAD`) 128 と 256 は同結果・同速 (12K で dsa_score 0.13 s/窓): 256 のまま。
- prefill のプール数 512 (exact 2048 トークン = DSA と同じ) でも 128K は 3/3、477 tok/s (attention 0.53 s/窓)。236K を検証中。
- **KDA scan の列カーネル** (`k_kda_step_col`、prompt 窓のみ、`STRATA_GLM5_KDA_COL=0` で旧カーネル): 旧カーネルはヘッドあたり 1 ブロック
  (32 ブロック) で 1 トークンにつき __syncthreads 4 回、ゲートの exp を 128×128 回。新カーネルは列 1 本を隣接 2 レーンで持ち (64 状態レジスタ、
  溢れなし)、exp/k/q を 1 回だけ共有し、内積はレーン内 + shuffle 1 回、同期 1 回/トークン。12K で A/B 中。
  結果: 12K で kda_scan 371 → 221 ms/窓 (−40%)、prefill 24.1 → 23.4 s (+3%)。縮約順が変わるので出力は同点付近で分岐する (既定オン)。
- 512 プールは 236K で 2/3 (436 tok/s) → 236K の既定は 1024 のまま。KDA 列カーネル込みで 236K の窓は 5.0 → 4.7 s。
- **要約のアブレーション** (`STRATA_GLM5_LOD_NOSUM=1`: indexer の選択だけで要約を足さない = 窓共有 DSA) を 12K/236K で実行中。
  差が出なければ GLM ではページ要約は不要で、速さの源泉は「窓共有の選択 + 圧縮オペランド + 融合 decode カーネル」という実装構造になる。

**アブレーションの結果**: 要約なし (窓共有 DSA、1024 プール) で 236K **3/3、433 tok/s** (要約あり 405 tok/s、attention 0.89 → 0.69 s/窓)、12K は 527 tok/s。
GLM-5.3-Flash (DSA で訓練) では ページ要約は NIAH に寄与せず、列が増える分 ~7% 遅い。得られた高速化 (DSA 142 → 433 tok/s @236K) の源泉は
「窓で選択を共有し [選択セル | 窓自身] を圧縮オペランドにして MMQ に通す」+「融合 decode カーネル」という実装構造。
→ サーバの既定は `--attn lod` + `STRATA_GLM5_LOD_ROUTE=pool` + `STRATA_GLM5_LOD_NOSUM=1` (= 窓共有 DSA)。ページ要約は密注意モデル向けのオプションとして残す。

## 層ごとの選択の重要度 (2026-10-08、64K NIAH 3 本)

`STRATA_GLM5_LOD_ROUTE_LAYERS="3=pool,7=tok,..."` で層ごとにルートを指定 (pool = indexer 選択、tok = indexer なしのトークン採点。
全層 tok は 64K で 3/3、128K 3/3、236K 1/3 だった)。

| 実験 | 結果 |
|---|---|
| keep-one (層 k だけ indexer、他 10 層 tok) | 層 3: 2/3、層 7〜43 の各単独: **3/3** (10/10 層) |
| leave-one-out (層 k だけ tok、他 indexer) | 層 23: 2/3、他: 3/3 |

→ 64K では「どこか 1 層が正確に選べれば十分」。層 23 (device 1 の最初の DSA 層) だけ外すと 1 本落ちた点は要再測 (針 3 本では統計的に弱い)。
indexer 採点のコストは窓の 7〜14% なので、速度面の意味は「残りの層の予算を小さくできるか」(次の実験: indexer 1 層 + 他層 top 32)。

**indexer 1 層 (23) + 他層トークン採点の速度試験**:

| 文脈 | 他層の予算 | NIAH | prefill |
|---|---|---|---|
| 64K | top 32 | 3/3 | 522 tok/s (全層 indexer 1024 プール: 436) |
| 128K | top 64 | 3/3 | 462 (448) |
| 236K | top 128 | **1/3** | 382 (433) |

64K/128K は速くて正答するが、236K では「1 層正確」では足りず、しかも全層トークン採点の列が増えて遅い。236K では全層 indexer
(プール選択) が最良のまま。文脈長で切り替えるのが実用的: ≤128K は層 23 indexer + tok、それ以上は全層 pool。

**Q2_K MMQ、dp4a レイアウトの試行 (2026-10-08)**: Q2_K だけ WMMA を切った翻訳単位 (dp4a タイル) を作って計測 → ベンチは WMMA 版と同一
(J=16 9.6 TMAC/s、J=24〜48 1.6)。レジスタ使用も同じ (J=32 で 520 B/lane、J=48 1.9 KB、J=64 2.5 KB のスクラッチ)。溢れの原因は
タイルのレイアウトではなく `mul_mat_q` の累積レジスタ (J×I/wave の float 和) が wave32 で J ≥ 24 だと 256 VGPR を超えること。
→ 設定表・レイアウトでは直らない。直すならカーネル本体 (J を 2 分割して 2 パス、または累積を LDS へ) か、Strata の融合 expert
カーネルの Q2_K/WMMA 版 (gfx1100 WMMA は int8 16×16×16; Q2_K の 16 値サブブロックのスケールを内積に畳む) を書く必要がある。
なお expert 1 本あたりの行数は平均 57、最大 ~300 なので J=16 のタイルは行数的には妥当で、J を上げても満杯にならない。
真の効率問題は I (重み行) 方向: I=64/128 のタイルを 2048 行 (n_ff) × K=4096 で回す構造が Q2_K の展開コストで律速。

## Q2_K expert の WMMA カーネル (2026-10-08)

`src/glm5/moe_q2k.cu`: prompt 窓の expert 積 (gate|up, down) を gfx11 の int8 WMMA (`wmma_i32_16x16x16_iu8`, wave32) で計算する
Strata 独自カーネル。MMQ の Q2_K (J≥24 でレジスタ溢れ、9.7 TMAC/s) を置き換える。
- 算術: w = d·sc_s·q − dmin·m_s (サブブロック 16 値)。A 行列 = sc_s·q を **バイト permute** (`__builtin_amdgcn_perm` で {0,sc,2sc,3sc} を
  2 bit コードで選ぶ、0..45 は int8 に収まる) で作り、K=16 の MMA 1 本がサブブロックにちょうど一致。min 項 Σ_s m_s·suma_s は
  活性の 16 値和 suma を下位/上位バイトに分けた 2 本の MMA (A = m_s)。活性は 256 値ブロックの int8 (スケール 1 本 + 16 値和)。
- フラグメント配置 (gfx1100 実測): C の lane l、要素 e = Σ_k (第 1 オペランドの lane 2e+l/16)·(第 2 オペランドの lane l%16)。
  活性を第 1 オペランドにすると lane = 重み行 l%16、要素 = 活性列 2e+l/16 になる (最初は逆に組んで誤答、one-hot プローブで判明)。
- ブロック = 8 wave × 16 重み行 = 128 行、列タイル 32 (2 × 16)、grid (行タイル, 列タイル, expert)。VGPR 170、溢れなし。
- 検証: ggml の `quantize_row_q2_K_ref`/`dequantize_row_q2_K` の double 参照と最大誤差 2.4e-5 (|ref| 245)。
- 速度 (288 expert、2048 トークン × 8、偏り分布): gate|up 28.3 → **12.3 ms**、22.4 TMAC/s (MMQ 9.7)、down 6.1 ms。
- engine: Q2_K の層 (36/42) で `moe_mmq` の積と活性量子化を置き換え (`STRATA_GLM5_NOQ2K=1` で MMQ に戻す)。Q3_K/Q4_K の層は MMQ のまま。
- engine での A/B (12K、pool/nosum): prefill 23.0 → **20.6 s (598 tok/s)**、窓 3.81 → 3.40 s。moe_hits 1221 → 599 ms、moe_down 682 → 366 ms
  (MoE 計算 1.9 → 0.97 s)。代わりに `moe_wait` 503 ms が現れた: 非常駐 expert の PCIe 転送 (≈35 GB/窓 ≈ 2.5 s) が計算に隠れきらなくなった
  → 次は PCIe 側 (ring 4 本、または 4096 窓で転送あたりのトークンを倍に)。1064 トークンの出力は活性の量子化ブロック (256 vs 128) の差で
  同点付近のトークンが変わる (MMQ 比)。

**4096 トークン窓 (2026-10-08)**: ハングの原因は doorbell の層別テーブル (routing id / miss slot / 予測) を chunk サイズで確保しながら
窓サイズをストライドにしていたこと (層 23 以降で領域外に書き「never published」)。ストライドを chunk に統一して解決 (3365985)。
WMMA で計算が速くなり PCIe が露出していたので、転送あたりのトークンを倍にする 4096 窓が効く:

| プロンプト | 2048 窓 + MMQ (朝) | 2048 窓 + WMMA | **4096 窓 + WMMA** |
|---|---|---|---|
| 4K | 7.6 s (531 tok/s) | | **5.4 s (764)** |
| 12K | 23.0 s (533) | 20.6 s (598) | **17.9 s (686)** |
| 128K (NIAH 3/3) | 251 s (470) | | **200 s (592)** |

出力は 2048 窓と一致 (4K 32/32、12K 16/16)。128K の窓 (4096) は 6.9 s: MoE 1.9 s、attention 1.0 s、PCIe 待ちは消えた。
→ サーバの既定を `--prefill-window 4096` に (R/bounce が +134 MB/台)。

## prompt 窓の 2 GPU パイプライン (2026-10-08、5ba5d30)

層は device 0 (0〜21) と device 1 (22〜44) に分かれているので、窓 k を device 1 が処理している間、窓 k+1 を device 0 で始められる
(窓 k+1 の層 L は窓 k の層 L の後に走ればよく、device の境界がその順序を保証する)。実装:
- **ジョブ** (`Engine::Job`: slot, pos0, T, tok, parity, トークン, device ごとの prefetch 位置, 次のジョブ): 層コードが読む「窓の状態」
  (`cur_slot_`/`tok_`/`ring_parity_`/`win_T_`) を `d.job` 経由に置き換え (`jslot/jtok/jparity/jT`、`stj(d, il)`)。device ごとに
  いま処理中のジョブを持つので、2 つの窓が同時に走れる。
- **ステージ** `stage_run(d, j, il0, il1)`: 1 device の層範囲を 1 窓ぶん (chunk ループ込み)。先頭ステージは埋め込み、末尾は head。
  受け渡しは残差ストリーム R の host bounce (窓の parity で 2 面) + device 間イベント (2 面)。逐次経路 (`forward_window`、decode 窓) も
  同じ `stage_run` を順に呼ぶ。
- **ステージスレッド**: device ごとに 1 本 (`stage_main`)、キューでジョブを受け渡す。各ステージの host 同期 (MMQ の max_rows) は自分の
  device だけを待つ。先頭ステージは末尾ステージより 1 窓以上先行しない (bounce/イベントの 2 面を 2 窓後に再利用するため)。
- **ring prefetch の連続**: `prefetch_up_to(d, j, limit)` はジョブ j の層が尽きたら `j.next` の層へ続ける (device ごとの mutex)。
  これで device 1 が窓 k を計算している間に窓 k+1 の最初の層の expert が ring に届く。フラグは層ごと (窓ではない) だが、
  DMA は前の読み手のイベントを待つので、窓 k の `moe_wait` が通ってからしか k+1 の値に進まない。
- **MTP の追従**: 窓ごとのコールバック (`prompt_windows(slot, toks, n, wsize, on_window)`) は末尾ステージのスレッドから呼ばれ、
  `h_win` がストリーム順で有効なうちに MTP 窓を同じストリームへ積む。最後の窓だけ `argmax_last` (device 1 の同期)。
- serve: スロットが decode 中でないときだけパイプライン (それ以外は従来の窓ごとループ + batch_step の割り込み)。CLI は常にパイプライン。
- 無効化: `STRATA_GLM5_NOPIPE=1`。条件: 2 device、prompt モード (窓 ≥ 256)、窓が 2 本以上。

### 結果 (stage35、72a90f7)

| ケース | 逐次 (stage33/34) | パイプライン | 出力 |
|---|---|---|---|
| 4K、窓 2048 ×2 | 6.73 s (608 tok/s) | 5.47 s (748 tok/s) | 32 トークン一致 |
| 12K、窓 4096 ×3 | 17.9 s (686 tok/s) | 12.23 s (1005 tok/s) | 16 トークン一致 |
| 128K NIAH、窓 4096 | 200 s (592 tok/s) | 111 s (1060 tok/s) | 3/3 |
| 1K 逐次 / slots-test | 変化なし | — | 24 トークン一致 / 0 mismatches |

窓が多いほど定常状態 (両 GPU が常に別の窓を計算) に近づき、128K で 1.8 倍。12K は 3 窓なので先頭/末尾の片肺ぶんが残る。
decode は無関係 (従来どおり)。

初回実装で踏んだバグ (全部 72a90f7 で修正):
- **ring の継続**: ring バッファ番号が `moe_seq % nbuf` (層ごと固定) だったので、device あたり MoE 層数が奇数 (19) のとき、次の窓の
  最初の層が今の窓の最後の層 (まだ読んでいない) と同じバッファに DMA されていた (→ ゴミ expert → NaN ルーティング → 不正アクセス)。
  窓ごとに device 別の ring 起点 `Job::rbase` を持たせ、`(rbase + moe_seq) % nbuf` に。逐次経路も同じ式 (窓間で起点が進む)。
- **bounce の再利用**: parity p の bounce を 2 窓後に書き直す前に、読み手 (次 device の h2d) のイベント `ev_in2[p]` を待つ。
  先頭ステージは末尾ステージが窓 k-1 を始めるまで窓 k を始めない (読み手の enqueue 順序の保証)。
- **末尾コールバックの窓状態**: `argmax_last` は `win_T_` で行数を決めるので、末尾ステージがコールバック前に `win_start_/win_T_` を設定。
- スレッド安全: `mmq_fits_` キャッシュ (map) と pin 切れ expert の読み込みプール (`Pool::run`) に mutex。
- `STRATA_GLM5_SYNC=1` が prompt モードで doorbell 待ちに入って固まる件: prompt モードでは待たない (診断に使えるように)。

np=2 の扱い: ジョブは slot を持ち、状態表は `stj(d, il)` で slot 別。serve はどの slot も decode 中でないときだけパイプライン
(decode 中は従来の窓ごとループ + batch_step 割り込み)。2 本の prompt を同時にパイプラインへ流す (窓を交互に) のは未実装。

## tool call (2026-10-08、ab27dec)

GLM-5.3 Flash はテンプレートどおり `<tool_call>NAME<arg_key>P</arg_key><arg_value>V</arg_value>...</tool_call>` を出すが、
serve/frontend.py の出力パーサは `<function=NAME><parameter=P>` 形式しか解釈せず HTTP 400 "malformed tool call" になっていた。
GLM 形式を追加 (非 streaming / streaming、schema が string の引数は文字列、それ以外は JSON デコード、値の中の `</arg_value>` は
次の `<arg_key>` か `</tool_call>` が続く位置で判定)。往復 (呼び出し → tool 結果 `<|observation|>` → 回答) と streaming を実機で確認。

## サーバ構成 (2026-10-08 時点)

| 設定ファイル | GPU | ctx | parallel | expert budget | 備考 |
|---|---|---|---|---|---|
| strata-glm53.json | 0,1 | 262144 | 2 | 28.5,25.8 | キャッシュ 3 セット |
| strata-glm53-np1.json | 0,1 | 262144 | 1 | 31.0,29.5 | 60.5 GiB 常駐、空き 1.2/1.55 GiB |
| strata-glm53-1gpu.json | 0 | 262144 | 1 | 17.9 (+ --pin-gib 96) | decode 15 tok/s、prefill 490 tok/s |

1 GPU では非常駐分 (82 GiB) を全部 pin しないと prompt の prefetch が staging slot 不足で止まる (`--pin-gib` 既定 64)。

## decode: Q2_K 積を WMMA で (2026-10-09)

decode 1 ラウンド (窓 3 行、spec 2、約 80 ms) の位相別実測: 常駐 expert の積 約 30 ms (5.5 GB を約 180 GB/s で読む)、密な層 約 25 ms、
CPU ミス待ち 6〜12 ms、残り約 10 ms。1 層ぶんのマイクロベンチ (20 expert × 1〜3 列、scratchpad `dec_bench`): grouped MMVQ 161 / 114 GB/s
(gate / down)、prompt 用 WMMA 268 / 252、D2D コピー 680。Q2_K はメモリではなく命令数律速 (素の Q2_K MMVQ はキャッシュに乗っていても 233 GB/s、
Q4_K は 500)。32 重み単位で整数累積する dp4a カーネルを先に試したが 3 列で 100 GB/s (重み 1 バイトあたり VALU/LDS 命令 6 個: 算術だけでその時間)。

int8 WMMA は 16 列ぶんの内積を 1 命令に詰めるので、decode カーネルは WMMA 版 (`dec_wmma`、moe_q2k.cu): グループあたり 16 列タイル 1 つ
(decode 窓では 1 expert に最大 8 対)、活性化の和を (列, 超ブロック) ごとに `pack_sums` で事前パック (prompt カーネルはレーン・超ブロックごとに
詰め直していた)、grouped 表 (`grp_w`, `col0`, `ncols`, `orow`) をそのままインターフェースに。ベンチで 550〜565 GB/s、同じオペランドで
prompt カーネルとビット一致。エンジンでは対の活性化を prompt 経路と同じ 256 単位で量子化 (`quantize_act`) するので decode と prompt の
量子化が統一された。`STRATA_GLM5_NODECQ2K=1` で grouped MMVQ に戻る。

| | 旧 | 新 |
|---|---|---|
| 1064 + 96 トークン、spec 2 | 28.6 tok/s (86 ms/ラウンド) | 30.8 tok/s (82 ms/ラウンド) |
| spec 3 | - | 30.8 tok/s (3.00 トークン/ラウンド) |
| slots-test、solo ドラフトなし | 88.5 ms/トークン | 76.5 ms/トークン |
| slots-test、2 スロット ドラフトなし | 67.3 ms/窓 | 57.4 ms/窓 |
| 1064 + 24 greedy | 参照 | 23/24 一致 (量子化の変更) |

カーネルは 3 倍なのにエンジンでは 8% の理由: CPU のミス計算 (ラウンドあたり約 180 expert、ピン留め RAM から 1.5 GB を約 34 GB/s = 44 ms) が
常駐 expert の積と並行していて、積を隠していた。積が 8 ms になると CPU 待ちが露出する (moe_wait 12 → 23 ms)。ラウンドは GPU だけの仕事
約 45 ms (密な層、router、MTP) と CPU の仕事 約 44 ms が層ごとに直列で、CPU 側はメモリバスが床。単独ストリームに残る手段は密な層
(9.8 GiB の重みに 25 ms、約 400 GB/s) と層ごとのオーバーヘッド。構造的な手段は 2 リクエストの層単位インターリーブ (一方の CPU 相を他方の
GPU 相に重ねる、np=2 で合計 約 1.6 倍の見込み)。

## decode: 貸出プール (2026-10-09)

**まず計測。** 新規のプロンプト 8 本 (日英の説明文、Python / TypeScript / Rust の課題、思考あり) では静的 profile のヒット率は 67% で、
decode は投機ありで 28〜30 tok/s (`--prompts` 経路はドラフトなしで 18 tok/s)。ルーティングログ (`STRATA_GLM5_ROUTE_LOG=<path>`: 常駐集合と
decode 窓ごとの id) を常駐方式に通して再生 (scratchpad `route_sim.py`):

| 同じルーティングでの方式 | ヒット | 窓あたりの新規ミス |
|---|---|---|
| 静的 profile | 66.9% | 2.65 |
| このストリームに最適な固定集合 | 82.9% | |
| LRU、ミスを全部常駐に (帯域無限) | 90.6% | 0.75 |
| EMA 移送、層あたり 1〜4 expert/窓 | 90.0〜90.7% | 0.74〜0.80 |
| 静的 + 貸出プール 10 / 20 / 30 スロット/層、LRU | 84.5 / 89.1 / 92.0% | 1.24 / 0.87 / 0.64 |

作業集合はリクエストとともに動く: 固定集合は 83% が上限、時間適応なら 90% を超える。

**貸出プール。** decode 中、prompt 窓の prefetch ring (2 枚で 2×1.34 + 2×1.20 GiB = 8.26 MB × 660 スロット) を expert store に貸し、
ミスした expert の LRU キャッシュにする: スロット表の `kLoanBit | j` が `loan_ptr[j]` を指す (grouping カーネルの第 2 のアドレス空間)、
store は `slot_of` と別に `loan_of` を持ち、表の更新は層ごとのリストを `k_loan_apply` が grouping の前に適用する。prompt 窓が ring を
取り戻す: `drop_loans()` がデバイスを同期して表を消す。保護: 今の窓で使ったスロットは犠牲にしない、コピー中のスロットも犠牲にしない、
追い出しの表更新は追加より先に適用。`STRATA_GLM5_LOAN=0` で無効、`STRATA_GLM5_LOAN_PER_WIN` で呼び出しあたりの上限。

**効かなかったものと理由。** CPU で計算したミスもホストから貸出スロットへコピーする方式 (`STRATA_GLM5_LOAN_H2D=1`) はヒット率を 85% に
上げたが decode は遅くなった。診断は 3 段: (1) コピーをサービス経路の copy ストリームに積んでいて各層のフラグが遅れた (修正: device ごとの
loan ストリーム); (2) 表の更新が compute ストリーム上の小さな H2D memcpy で、コピーエンジンの待ち行列で 8 MB の貸出コピーの後ろに並び
毎層止まっていた (moe_wait 11 → 42〜78 ms; 修正: apply カーネルがマップ済みピン留めのエントリを直接読む、CPU の結果行もカーネルでコピー);
(3) 残ったのはメモリバス: CPU で計算するミスは blob 8.3 MB を 1 回読み、貸出コピーがもう 1 回読むので、定常状態ではコピーの代金が
増えたヒットの利得と相殺する (同じプロンプトで 25.6 vs 28.5 tok/s)。そこで既定は CPU/PCIe 分担が PCIe に送ったミスだけを貸し出す:
blob はバスを 1 回 (staging へ) しか渡らず、貸出は D2D コピー。分担を PCIe 寄りに偏らせる (`STRATA_GLM5_DMA_MS`) と PCIe 側が長くなって
また負ける (31.9〜32.4 vs 33.6)。

| プロンプト (300 トークン、spec 2) | 貸出なし | 貸出 (PCIe 経由分) | ヒット なし → あり |
|---|---|---|---|
| Python CLI ツール (3) | 28.5 | **33.6** | |
| 日本の四季 (1) | 36.1 | **39.2** | 86.6 → 89.3% |
| TypeScript (5) | 30.2 | **36.0** | 74.3 → 79.2% |
| Rust (8) | 31.9 | **34.0** | 73.6 → 78.8% |

1064 + 24 greedy: 参照と 24/24 一致; `--slots-test 2`: 0 mismatches。

**decode の現在地。** 1 ラウンド (3 行) は GPU だけの仕事 約 45 ms (密な層 25、router、MTP) と、CPU のミス (1 件 約 0.2 ms = 8.3 MB を
ピン留め RAM からバスの 42 GB/s で) が層ごとに直列。残る手段: 貸出容量の拡大 (prompt scratch、カードあたり約 4 GiB 増)、GPU 側の基礎時間の
削減、2 リクエストなら層インターリーブ。短いプロンプト (T < 256、中サイズ窓の経路) は ring の PCIe コピー待ち (39 トークンで 1.1 s) で、
CPU 分担はそこにも適用できる。

**それが起こしたクラッシュ (修正済み、e97a32a)。** サーバのエンジンが長い回答の約 1,800 トークンで落ちた: serve_cpu のルーティング
ループでの segfault (カーネルログのアドレスをシンボル化、gdb でも窓 803 で再現) = 0..287 の外の routing id、つまり活性化がゴミに
なっていた。ホストとデバイスの貸出の見え方に穴が 2 つ: (1) ホストは割り当て時点で常駐扱い、デバイスは次の grouping の表更新からなので、
片側でヒット・片側でミスになり対の行配置が食い違う; (2) LRU が同じ窓のまだ後の層のスロットを追い出せて、その層の (未公開の) grouping が
読む最中に新しいコピーが上書きする。いまは貸出は適用後にだけ `loan_live`、犠牲はこの窓で処理済みの層のものだけ、範囲外 id はメッセージで
停止。2,500 トークン × 2、一致テスト、8 プロンプト交互が通過。

## decode: ラウンドあたりのドラフト本数 (2026-10-10、不採用)

新規プロンプト (400 トークン、貸出あり) で固定本数を比較: 2 / 3 / 4 本で TypeScript 32.6 / 34.2 / 31.0 tok/s、Rust 34.1 / 31.8 / 28.1。
3 本目は片方で得、片方で損 (採択率 82% vs 71%)、4 本目は常に損。適応方式を 2 つ試した: 位置ごとの採択率のしきい値 (保守的すぎて
TypeScript 32.4) と、本数ごとの実測レート (トークン数と所要時間の EMA、8 ラウンドに 1 回は隣を試す) で、後者は `--spec 4` で両方の
回答で固定の最良値に並んだ (34.2 / 34.1)。ただし利得は 400 トークン計測のばらつき (約 3%) の範囲で、サーバの最初のリクエストでは
方針が 2 本に落ち着いた。decode ループを単純に保つため撤回し、設定は `--spec 2` のまま。サーバの実トラフィックの採択データ
(188 リクエスト): 中央値 72%、1/3 のリクエストが 65% 未満 (日本語とコードの回答)、2.57 トークン/ラウンド。

## 小さな改善、第 1 回 (2026-10-13)

このビルドの位相プロファイル (40 トークンのプロンプト 1.81 s; decode ラウンドの GPU 位相 73〜74 ms; 4K 窓 5.22 s) と、各項目の結果:

| 項目 | 発見 | 結果 |
|---|---|---|
| 行ごとの重み再読 | F32 GEMV (router、indexer q_b)、mHC の混合、DSA の absorb/expand が窓の行ごとに行列を読み直していた (grid y/z = t): decode 窓では 3 倍のバイト | T ≤ 8 は 1 回読みの形 (ビット一致): decode 27.0 → 28.4 tok/s。広い窓は行ごとの grid のまま (8 行の形はブロック数が足りず mhc 374 → 680 ms @4K) |
| 中サイズ窓 (9 ≤ T < 256) | ミスが全部 PCIe (serve_ring): 40 トークンのプロンプト 1.81 s のうち 1.27 s が待ち | `serve_mid`: decode と同じ CPU/PCIe 分担 (CPU は対の少ない expert、最大 256 行)、CPU の行を MMQ の配置に書く (grouping カーネルが 2048 対までトークン順を保つ: `det_n`)、デバイス側の積はマスクで飛ばす (Q2_K WMMA カーネル)。分担は Q2_K 層のみ: このモデルの最初の MoE 層は Q4_K / Q3_K で汎用 MMQ (マスク無し) を通るため、初回は古い ring データを計算して活性化が壊れた。40 トークン 1.81 → 1.16 s、147 トークン 3.02 → 2.03 s。`STRATA_GLM5_MID_CPU=0` で無効 |
| prompt 用 WMMA カーネル | 活性化の和をレーン・超ブロックごとに詰め直し (タイルあたり 16 ロード + 32 命令) | decode カーネルと同じ (列, 超ブロック) ごとの事前パック: 4K 窓 5.42 → 5.36 s |
| shared expert | gate と up が別々の積 | ロード時に連結 (`concat_rows`)、1 回の積、`swiglu_clamp_gu_t` (層あたり起動 1 つ減、起動あたりの行 2 倍) |
| 広い窓の mHC 混合 | GEMM 形式のカーネル (fn のスライスを LDS に、残差行は 1 回読み) | 正しい (値 ~200 に対し最大差 1e-4) が行ごとの grid より速くない (4K で 371 vs 374 ms) 上に greedy トークンが動く: 既定オフ (`STRATA_GLM5_MHC_WIDE=1`) |
| 中サイズ窓の小さな BF16 行列 | T=40 で "kda_gates" 195 ms は BLAS の遅い呼び出しに見えた | プロセス最初の GEMM 呼び出しに乗る 1 回限りの BLAS ウォームアップ (約 170 ms) で、行列を別経路に移しても位相が移るだけ。変更なし。起動時のウォームアップで最初のリクエストから外せる |
| doorbell の失敗 | doorbell スレッドの失敗でデバイスが `moe_wait` に永久停止 (今日 2 回) | `release_waits()`: 全層のフラグを窓のトークンに; ホストは次の同期で throw |
| 範囲外の routing id | 以前は segfault | 層と窓つきで報告 |

測れなかったもの: HIP ランタイムの起動ログ (`AMD_LOG_LEVEL`) はこの ROCm では何も出さず、ラウンドあたりの起動数 (推定 ~1,200) は推定のまま。
430 トークンの prompt モード窓は 256 トークン以上のプロンプトの床を示す: routing と無関係に非常駐 expert を全部 prefetch (PCIe 2.3 s)。
その窓への分担の拡張が 256〜2048 トークンのプロンプトの次の手段。

## 小さな改善、第 2 回 (2026-10-13)

- **起動時の BLAS ウォームアップ**: GEMM オブジェクト作成時に device ごとに小さな bf16 / f16 GEMM を 1 回ずつ。プロセス最初の prompt 窓が
  約 170 ms を払わなくなった (prompt モードで 1064 トークン 3.42 → 3.27 s、430 トークン 3.35 → 3.21 s)。
- **256 トークン超の単一窓プロンプトへの分担拡張**: そうした窓は非常駐 expert を全部 prefetch していた (430 トークンで PCIe 2.3 s)。
  中サイズ経路を `STRATA_GLM5_MID_MAX_T` (640) トークンまでに: ミスは層ごとに routing で供給、CPU は対の少ない expert (上限 層あたり
  1024 行、ピン留め結果ブロックと公開した MoE 入力は層の偶奇で 2 面)、grouping カーネルは expert ごとの走査で窓の対をトークン順に
  (`rank_scratch`)。クロスオーバー: 430 トークン 3.21 → 2.58 s、646 トークン 3.23 → 3.04 s、802 トークン 3.23 vs 3.30 s、1064 トークン
  3.27 vs 3.76 s (約 700 トークンを超えるとほぼ全 expert が多数の対で routing され、CPU の取り分が縮む一方で先読みの重なりを失う)。
  出力は prefetch-all 経路と CPU の丸めぶんだけ異なる (分担ありの 1064 + 24 テストは元の参照と 24/24 一致)。40 トークンのプロンプト
  0.97 s (この一連の作業の開始時 1.81 s)、slots-test 0 mismatches。

## 小さな改善、第 3 回: decode ラウンド (2026-10-13)

この回の前に decode ラウンド (3 行、spec 2) のカーネル調査 (rocprofv3) を取った: 2 枚のカードでラウンドあたり約 3,430 回の起動、
カーネル約 46 ms、CPU 供給の expert を待つ doorbell の空転約 50 ms、カーネル間の隙間約 13 ms (1 回 3〜4 us: RDNA 3 の in-order
キューが依存する起動の間に置く)。この回は起動数を減らし、待ちの下に仕事を移す。変更はすべて環境変数で戻せる (併記)。slots-test
(solo と batch 行の比較、CPU 供給 off) は 0 mismatches のまま。

- **KDA 層** (`STRATA_GLM5_KDA_FUSE=0` で元に): q | k | v を積んだ 1 つの Q8_0 積と 1 回の conv 起動 (`wqkv`、`conv_qkv`; prompt 窓は
  3 つの行スライスを順に)、f_a | g_a | beta を 1 つの BF16 積 (`fab`、列はストライド読み)、scan と前後の処理を 1 カーネルに
  (`k_kda_step_f`: L2 正規化、ゲート、beta の sigmoid、再帰、出力の RMS norm とゲート; ゲートの exp は状態行ごとでなく要素ごとに 1 回)。
  置き換えた連鎖とビット一致。層あたり 26 起動 → 10。
- **mHC の decode 末尾** (`STRATA_GLM5_MHC_DEC=0`): rs + 非線形 + pre-mix + 続く RMS norm をトークンごと 1 ブロック (`k_mhc_finish_t`、
  1024 スレッド); mixes カーネルはスケールなしで走り、和にスケールを掛ける (この回で唯一の丸めの変更)。層の半分あたり 6 起動 → 2。
- **MoE の順序** (`STRATA_GLM5_MOE_EARLY=0`): 常駐 expert の swiglu / down 積と共有 expert を、供給されるミスの待ちより前に (そこで
  GPU は遊んでいた); combine が共有の行を足して層出力へ直接書く (`add_rows` + memcpy が消えた)。ビット一致。
- **生成側で q8 行を作る** (`STRATA_GLM5_Q8_FUSE=0`): `xn`、KDA 出力 `y`、共有 expert の `h` を書くカーネルが q8_1 行も書く
  (`native_quantize_q8_1` と同じビット、32 要素ブロックごとに 1 warp) ので、そこから始まる Q8_0 射影は別の quantize 起動を省く
  (`gemv_q8`)。dense 層の gate | up も積んだ。
- **細かいもの**: CPU 供給の行のコピーを層ごと 1 回の記述子駆動の起動に (expert ごとだった); Q2_K の活性量子化が自分で和を詰める
  (decode 経路から `pack_sums` が消えた); routing カーネルの直列部は 288 全部でなく routed された expert (shared memory でソート) を
  歩く (38 → 16 us); 残差ストリームの初期化と平均を窓ごと 1 起動に; draft 巻き戻しの 136 回の状態 memcpy をカードごと 1 回の
  テーブル駆動起動に; f_b / g_b 向けの狭い K の BF16 GEMV (出力行ごと 1 スレッド、列は shared memory) (`STRATA_GLM5_NARROW_BF16=0`);
  MoE の q8 入力は非 Q2_K 層でだけ作る。
- **貸出プールを prompt スクラッチへ** (`STRATA_GLM5_LOAN_SCRATCH=0`): decode 窓が使わないバッファ (dense attention と MMQ / GEMM の
  スクラッチは丸ごと、対・射影・attention のバッファは 16 行分より後ろ) も貸出スロットに刻む: リング 622 + スクラッチ 892 スロット
  (7.4 GB 増)。prompt 窓 (T >= 9) はバッファに触る前に貸出を落とす仕組みがもともとある。

decode ラウンドあたり (40 トークンのプロンプト後の 200 トークン; ラウンドあたりのトークン数は受理された draft で変わるので
ラウンド時間で測る): この回の前 79.0 ms → 76.7 (KDA) → 74.7 (mHC、MoE 順序) → 73.6 (q8 行、routing、コピー) → 69.4 ms (スクラッチの
貸出プール) → 67.3 ms (狭い BF16 GEMV、和を詰める量子化、巻き戻しテーブル、広い mHC 末尾); ラウンドあたりの起動 3,430 → 2,295
(貸出変更の前)。試して戻したもの: llama.cpp の 2 行/ブロックの多列 MMVQ 配置 (`STRATA_GLM5_MMVQ_UPSTREAM=1`): 67.7 ms、利得なし。
この回ではほかに: 貸出コピーをコピーエンジンでなくシェーダのカーネルで (`STRATA_GLM5_LOAN_MEMCPY=1` で元に)、doorbell の publish の
ホスト書き込みを 16 バイト単位に。未解決: CPU 供給 off (`STRATA_GLM5_CPU=0`、slots-test のモード) では、この回の後の solo decode が
全スイッチ off でも 75 → 83 ms/token と測れ、GPU の各フェーズは doorbell の待ち以外すべて同じ; 同じ routing でヒットが減る
(VRAM から 82.5 → 79.8%、adaptive swap と thrash が増える) - 先読みの staging と swap の方針がタイミングの変化に反応している。
サーバが走る CPU 供給モードは一貫して速い。slots-test の solo decode (CPU 供給 off): 84.0 → 74.7 ms/token。
プロンプトは変わらず (40 トークン 0.97 s、430 トークン 2.59 s、1064 トークン 3.26 s、1064 トークンの出力は同一)。残るラウンド:
CPU 供給 expert の待ち約 43 ms (メモリバス律速)、Q8_0 dense 積約 12 ms、Q2_K expert 積約 10 ms、起動の隙間約 11 ms。

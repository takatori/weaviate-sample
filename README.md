# Weaviate Multi-Vector Search Experiments

Weaviateを使用したマルチベクトル検索（ColBERT/MUVERA）およびシングルベクトル検索の性能評価プロジェクトです。

## データセット

本プロジェクトでは、**JQaRA (Japanese Question-Answer Retrieval Assessment)** データセットを使用しています。

- **リポジトリ**: https://github.com/hotchpotch/JQaRA
- **概要**: RAG（検索拡張生成）を用いた日本語質問応答の精度を評価するためのデータセット
- **構成**: 1,667件のテストクエリ、各クエリに100件の候補文書（正解ラベル付き）
- **ベースデータ**: JAQKETのQ&A + 日本語Wikipedia

## プロジェクト構成

```
.
├── encoder.py                  # ColBERTマルチベクトルエンベディング生成スクリプト
├── notebook/
│   ├── muvera.ipynb           # MUVERA (ColBERT) 評価ノートブック
│   ├── single-vector.ipynb    # シングルベクトル評価ノートブック
│   ├── tutorial.ipynb         # Weaviate基本チュートリアル
│   └── weaviate-cloud.ipynb   # Weaviate Cloud実験
├── data/
│   └── jqara_test.jsonl       # JQaRAテストデータ
├── outputs/                    # エンベディング出力ディレクトリ
│   ├── jacolbert/             # JaColBERT v1 エンベディング
│   └── jacolbert-v2.5/        # JaColBERT v2.5 エンベディング
├── docker-compose.yaml         # Weaviate + Ollama
└── muvera-memo.md             # ColBERT/MUVERAアルゴリズムメモ
```

## セットアップ

### 環境構築

```bash
# 依存関係インストール
uv sync

# 仮想環境有効化
source .venv/bin/activate
```

### Weaviate起動

```bash
docker compose up -d
```

Weaviateは `localhost:8080` で起動します。

## 使用方法

### 1. ColBERTエンベディング生成

```bash
# JaColBERT v1
python encoder.py \
  --input ./data/jqara_test.jsonl \
  --output_dir ./outputs \
  --model jacolbert \
  --batch_size 128

# JaColBERT v2.5
python encoder.py \
  --input ./data/jqara_test.jsonl \
  --output_dir ./outputs \
  --model jacolbert-v2.5 \
  --batch_size 128
```

出力先: `outputs/{model_name}/batch_*.pt`

### 2. MUVERA評価

`notebook/muvera.ipynb` を開いて実行します。

```python
# モデル選択
SELECTED_MODEL = "jacolbert"  # または "jacolbert-v2.5"
```

### 3. シングルベクトル評価

`notebook/single-vector.ipynb` を開いて実行します。

```python
# モデル選択
SELECTED_MODEL = "multilingual-e5-small"  # small/base/large
```

## 対応モデル

### ColBERT (マルチベクトル)

| モデルID | HuggingFace |
|----------|-------------|
| `jacolbert` | bclavie/JaColBERT |
| `jacolbert-v2.5` | answerdotai/JaColBERTv2.5 |

### Sentence Transformers (シングルベクトル)

| モデルID | HuggingFace | 次元 |
|----------|-------------|------|
| `multilingual-e5-small` | intfloat/multilingual-e5-small | 384 |
| `multilingual-e5-base` | intfloat/multilingual-e5-base | 768 |
| `multilingual-e5-large` | intfloat/multilingual-e5-large | 1024 |

## 評価指標

ranxライブラリを使用して以下の指標を計算：

- NDCG@1, @3, @5, @10
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Precision@10
- Recall@10, @100

## 参考資料

- [JQaRA](https://github.com/hotchpotch/JQaRA) - 日本語質問応答リトリーバル評価データセット
- [Weaviate Multi-Vector Documentation](https://weaviate.io/developers/weaviate/concepts/vector-index#multi-vector)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) - Late Interaction検索モデル
- [MUVERA](https://arxiv.org/abs/2405.19504) - Multi-Vector Retrieval via Fixed Dimensional Encodings

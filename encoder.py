# jqara_to_colbert_multivectors.py
import os
import json
import glob
import argparse
from typing import Iterator, Tuple, List

import torch
from tqdm import tqdm
import pandas as pd

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


# ---------- I/O: JQaRA ローダ（JSONL/JSON/CSV） ----------
def detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jsonl", ".jl"]:
        return "jsonl"
    if ext in [".json"]:
        return "json"
    if ext in [".csv"]:
        return "csv"
    # フォールバック：行を見て判定
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline().strip()
        if head.startswith("{") and head.endswith("}"):
            return "jsonl"  # 1行1JSON とみなす
    return "jsonl"


def stream_records(path: str) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (id, title, text) for each record.
    """
    fmt = detect_format(path)
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                _id = str(obj.get("id"))
                title = obj.get("title", "") or ""
                text = obj.get("text", "") or ""
                if _id is None:
                    continue
                yield _id, title, text
    elif fmt == "json":
        data = json.load(open(path, "r", encoding="utf-8"))
        # data が list か dict（{"data":[...]}) を想定
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        for obj in data:
            _id = str(obj.get("id"))
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            if _id is None:
                continue
            yield _id, title, text
    elif fmt == "csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            _id = str(row.get("id"))
            title = str(row.get("title")) if not pd.isna(row.get("title")) else ""
            text = str(row.get("text")) if not pd.isna(row.get("text")) else ""
            if _id is None:
                continue
            yield _id, title, text
    else:
        raise ValueError(f"Unsupported format detected: {fmt}")


def count_records(path: str) -> int:
    fmt = detect_format(path)
    if fmt == "jsonl":
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    elif fmt == "json":
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return len(data)
    elif fmt == "csv":
        return sum(1 for _ in open(path, "r", encoding="utf-8")) - 1  # header 除く
    return 0


# ---------- 進捗・再開管理 ----------
def list_done_ids(output_dir: str) -> set:
    """
    既存バッチファイル(*.pt)から処理済み id を収集。
    """
    done = set()
    for fp in glob.glob(os.path.join(output_dir, "batch_*.pt")):
        try:
            payload = torch.load(fp, map_location="cpu")
            # payload: {'ids': [...], 'embeddings': List[torch.Tensor]}
            ids = payload.get("ids", [])
            done.update(map(str, ids))
        except Exception:
            # 壊れたファイルは無視（必要なら手動削除）
            pass
    return done


# ---------- 埋め込み（ColBERT） ----------
@torch.inference_mode()
def encode_texts_with_colbert(
    ckpt: Checkpoint,
    texts: List[str],
    bsize: int = 32,
) -> List[torch.Tensor]:
    """
    ColBERT の doc エンコーダで "multi-vector"（トークン表現列）を得る。
    返り値は各文書ごとに shape [n_tokens, dim] の Tensor（CPUテンソル推奨）。
    """
    # ColBERT Checkpoint API 想定: docFromText
    # keep_dims=False で doc 単位の list 取得
    result = ckpt.docFromText(texts, bsize=bsize, keep_dims=False)

    # 返り値の形式を確認して処理
    # keep_dims=False の場合、通常は List[torch.Tensor] を返すが、
    # 場合によってはタプルや3Dテンソルを返す可能性がある
    if isinstance(result, tuple):
        # タプルの場合は最初の要素を取得
        embs = result[0]
    else:
        embs = result

    # 3Dテンソル [batch_size, seq_len, dim] の場合はリストに変換
    if isinstance(embs, torch.Tensor) and embs.dim() == 3:
        # 各文書ごとに [seq_len, dim] のテンソルに分割
        cpu_embs = []
        for i in range(embs.shape[0]):
            doc_emb = embs[i]  # [seq_len, dim]
            # パディング部分を除去（全て0の行を削除）
            mask = ~(doc_emb == 0).all(dim=1)
            doc_emb = doc_emb[mask]
            cpu_embs.append(doc_emb.detach().to("cpu").contiguous())
        return cpu_embs

    # リストの場合
    if isinstance(embs, list):
        cpu_embs: List[torch.Tensor] = []
        for e in embs:
            if isinstance(e, torch.Tensor):
                # 既にテンソルの場合はそのままCPUへ
                cpu_embs.append(e.detach().to("cpu").contiguous())
            elif isinstance(e, (list, tuple)):
                # リストやタプルの場合はテンソルに変換
                try:
                    tensor = torch.tensor(e, dtype=torch.float32)
                    cpu_embs.append(tensor.to("cpu").contiguous())
                except (ValueError, TypeError):
                    # 変換できない場合はスキップ（デバッグ用に警告）
                    print(
                        f"Warning: Could not convert embedding to tensor: {type(e)}, shape: {getattr(e, 'shape', 'N/A')}"
                    )
                    raise
            else:
                # その他の型は試行
                try:
                    tensor = torch.as_tensor(e, dtype=torch.float32)
                    cpu_embs.append(tensor.to("cpu").contiguous())
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert embedding to tensor: {type(e)}")
                    raise
        return cpu_embs

    # 予期しない形式
    raise ValueError(
        f"Unexpected return type from docFromText: {type(result)}, value: {result}"
    )


def concat_title_text(title: str, text: str, sep: str = "\n\n") -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if title and text:
        return f"{title}{sep}{text}"
    return title or text or ""


# ---------- メイン処理 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="JQaRA の JSONL/JSON/CSV ファイルパス"
    )
    parser.add_argument(
        "--output_dir", default="outputs", help="出力ディレクトリ（バッチ保存）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="ColBERT 推論用バッチサイズ"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=512,
        help="保存単位（この件数ごとに1ファイル）",
    )
    parser.add_argument("--sep", default="\n\n", help="title と text の結合セパレータ")
    parser.add_argument("--fp16", action="store_true", help="FP16 推論（GPU推奨）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ColBERT Checkpoint 準備
    # Appleシリコン対応: MPS > CUDA > CPU の順で選択
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # MPS使用時はFP16とAMPを完全に無効化（ColBERTのAMPがCUDA専用のため）
    use_fp16 = args.fp16 and device == "cuda"

    cfg = ColBERTConfig(
        nbits=2,  # 量子化ビット（保存時ではなくインデックス時に影響。推論はそのまま）
        doc_maxlen=300,  # 必要に応じて調整（長文はトークン数↑→ベクトル数↑）
        query_maxlen=64,
        mask_punctuation=True,
        dim=128,  # jina-colbert-v2 既定の埋め込み次元（モデル側に従う）
        amp=use_fp16,  # 自動混合精度（CUDA時のみ有効）
        gpus=[0] if device == "cuda" else [],  # CUDA時のみGPU設定
    )
    ckpt = Checkpoint("jinaai/jina-colbert-v2", colbert_config=cfg)
    ckpt = ckpt.to(device)

    # MPS使用時は警告を表示
    if device == "mps":
        print(
            "Using Apple Silicon GPU (MPS). FP16 and AMP are disabled for compatibility."
        )

    total = count_records(args.input)
    done_ids = list_done_ids(args.output_dir)
    already = len(done_ids)

    print(f"Total records in input: {total}")
    print(f"Already processed (found in {args.output_dir}): {already}")
    remaining = total - already
    if remaining <= 0:
        print("All records already processed. Nothing to do.")
        return

    # 再開：既存の batch_*.pt の最大インデックスから続きの番号を振る
    existing_batches = sorted(glob.glob(os.path.join(args.output_dir, "batch_*.pt")))
    next_batch_idx = 0
    if existing_batches:
        try:
            last = existing_batches[-1]
            # outputs/batch_00012.pt -> 12
            base = os.path.basename(last)
            num = int(base.replace("batch_", "").replace(".pt", ""))
            next_batch_idx = num + 1
        except Exception:
            pass

    ids_buf: List[str] = []
    texts_buf: List[str] = []
    pbar = tqdm(total=remaining, desc="Encoding (resume-aware)", unit="doc")

    for _id, title, text in stream_records(args.input):
        if _id in done_ids:
            continue
        merged = concat_title_text(title, text, sep=args.sep)
        if not merged:
            # 空文書はスキップ（空の埋め込みを避ける）
            done_ids.add(_id)
            pbar.update(1)
            continue

        ids_buf.append(_id)
        texts_buf.append(merged)

        # シャード境界でエンコード＆保存
        if len(ids_buf) >= args.shard_size:
            # エンコード
            embs = encode_texts_with_colbert(ckpt, texts_buf, bsize=args.batch_size)
            # 念のため対応数チェック
            assert len(embs) == len(ids_buf), "Mismatch between ids and embeddings!"

            # 保存（id→マルチベクトル配列 の形を保つ）
            payload = {
                "ids": ids_buf,  # List[str]
                "embeddings": embs,  # List[Tensor[n_tokens, dim]]  可変長
                "meta": {
                    "model": "jinaai/jina-colbert-v2",
                    "dim": int(embs[0].shape[-1]) if embs else cfg.dim,
                    "doc_maxlen": cfg.doc_maxlen,
                    "created_on": torch.tensor([])
                    .new_empty(0)
                    .dtype.__str__(),  # 単なる占位情報
                },
            }
            out_path = os.path.join(args.output_dir, f"batch_{next_batch_idx:05d}.pt")
            torch.save(payload, out_path)
            next_batch_idx += 1

            # 進捗更新
            pbar.update(len(ids_buf))
            done_ids.update(ids_buf)

            # バッファクリア
            ids_buf.clear()
            texts_buf.clear()

    # 端数があれば保存
    if ids_buf:
        embs = encode_texts_with_colbert(ckpt, texts_buf, bsize=args.batch_size)
        assert len(embs) == len(ids_buf), "Mismatch between ids and embeddings!"
        payload = {
            "ids": ids_buf,
            "embeddings": embs,
            "meta": {
                "model": "jinaai/jina-colbert-v2",
                "dim": int(embs[0].shape[-1]) if embs else cfg.dim,
                "doc_maxlen": cfg.doc_maxlen,
            },
        }
        out_path = os.path.join(args.output_dir, f"batch_{next_batch_idx:05d}.pt")
        torch.save(payload, out_path)
        pbar.update(len(ids_buf))

    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()

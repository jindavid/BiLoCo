"""Text datasets used for calibration and evaluation (downloaded on demand from the HF Hub)."""
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def _c4(shard: str):
    return load_dataset("json", data_files=[hf_hub_download("allenai/c4", shard, repo_type="dataset")], split="train")


def load_text_dataset(name: str, max_samples: int = 0, offset: int = 0):
    """name: wikitext (WikiText-2 test), c4 (validation shard 0), c4_train (train shard 0),
    fineweb (fineweb-edu sample-10BT shard 0). Keeps documents longer than 50 characters
    (fineweb: non-empty), then applies offset and max_samples."""
    if name == "wikitext":
        ds, min_len = load_dataset("wikitext", name="wikitext-2-raw-v1", split="test"), 50
    elif name == "c4":
        ds, min_len = _c4("en/c4-validation.00000-of-00008.json.gz"), 50
    elif name == "c4_train":
        ds, min_len = _c4("en/c4-train.00000-of-01024.json.gz"), 50
    elif name == "fineweb":
        path = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset")
        ds, min_len = load_dataset("parquet", data_files=[path], split="train"), 0
    else:
        raise ValueError(f"unknown dataset {name}")
    ds = ds.filter(lambda x: len(x.get("text", "").strip()) > min_len)
    if offset > 0:
        ds = ds.select(range(offset, len(ds)))
    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def collate_text(batch):
    return {"text": [item["text"] for item in batch]}

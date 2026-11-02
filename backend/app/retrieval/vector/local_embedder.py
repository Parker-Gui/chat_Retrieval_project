"""Local all-MiniLM-L6-v2 embedder that does not depend on the ST auto-loader.

`sentence-transformers` 6.2 with `transformers` 5.22 cannot load this cached model
offline: the loader calls `AutoProcessor.from_pretrained` unconditionally and
transformers 5.x raises because the repository stores only a tokenizer. This module
reproduces exactly what the SentenceTransformers pipeline of that checkpoint does -
mean pooling over the attention mask, then L2 normalisation - so vectors stay
compatible with the `KnowledgeChunk` index that was built with it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from neo4j_graphrag.embeddings.base import Embedder
from transformers import AutoModel, AutoTokenizer


DEFAULT_MAX_LENGTH = 256


def resolve_snapshot_path(cache_dir: Path, model_name: str) -> Path:
    """Return the local snapshot directory for a Hugging Face model cache entry.

    The cache folder name uses the fully qualified repository id, so a bare model
    name such as ``all-MiniLM-L6-v2`` is also tried under the
    ``sentence-transformers`` organisation.
    """
    candidates_names = [model_name]
    if "/" not in model_name:
        candidates_names.append(f"sentence-transformers/{model_name}")
    searched = []
    for candidate in candidates_names:
        folder = "models--" + candidate.replace("/", "--")
        snapshots = cache_dir / folder / "snapshots"
        searched.append(str(snapshots))
        if not snapshots.is_dir():
            continue
        usable = [path for path in snapshots.iterdir() if (path / "config.json").is_file()]
        if usable:
            return max(usable, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(
        f"No cached snapshot for {model_name} under {searched}. "
        "Download the embedding model once while the network is available."
    )


class LocalMiniLMEmbedder(Embedder):
    """Mean-pooled, L2-normalised sentence embedder compatible with the index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str | Path | None = None,
        device: str = "cpu",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if cache_folder is None:
            raise ValueError("cache_folder is required to locate the offline model.")
        self.model_path = str(resolve_snapshot_path(Path(cache_folder), model_name))
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed_query(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            [str(text)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        normalised = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalised[0].to("cpu").tolist()

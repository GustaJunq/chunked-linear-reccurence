"""
Data loader: os arquivos .bin ficam no disco, e a gente lê
janelas aleatórias via memmap (numpy), sem carregar o corpus inteiro na RAM.
Funciona bem mesmo com corpora de dezenas de GB.

O dtype dos tokens é lido do meta.json (escrito pelo prepare_data.py):
uint8 pro tokenizer byte-level, uint32 pro Qwen3 (vocab grande demais pra
uint8/uint16). Se o meta.json não existir, cai pra uint8 por compatibilidade
com dados preparados antes dessa mudança.
"""

import json
import os
import numpy as np
import torch


def _load_dtype(data_dir):
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return np.dtype(meta["dtype"]), meta.get("vocab_size")
    return np.dtype(np.uint8), None  # fallback pra dados antigos (só byte-level)


class BinDataset:
    def __init__(self, data_dir, split, block_size):
        assert split in ("train", "val")
        self.path = os.path.join(data_dir, f"{split}.bin")
        self.block_size = block_size
        self.dtype, self.vocab_size = _load_dtype(data_dir)
        # reabrimos o memmap a cada get_batch pra evitar leak de memória
        # em workers/DDP de longa duração (é o mesmo truque do nanoGPT)

    def get_batch(self, batch_size, device):
        data = np.memmap(self.path, dtype=self.dtype, mode="r")
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

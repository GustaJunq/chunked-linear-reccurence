"""
Prepara um corpus de texto (.txt) em arquivos binários de tokens (train.bin,
val.bin), no estilo nanoGPT — carregados via memmap no data.py.

Dois tokenizadores disponíveis:

  --tokenizer byte   (padrão) byte-level UTF-8, vocab_size=256, zero
                      dependências. Bom pra prototipar/validar a arquitetura.

  --tokenizer qwen3   usa o tokenizer BPE do Qwen3 via `transformers`
                      (baixa o tokenizer.json do Hugging Face Hub na
                      primeira execução — precisa de internet nessa hora).
                      Vocabulário ~151.9k tokens, então os .bin passam a
                      ser gravados em uint32 (não cabe mais em uint8).

Tudo isso (vocab_size, dtype, nome do tokenizer) é gravado em
`meta.json` dentro de --out_dir, e data.py / train.py leem esse arquivo
automaticamente — não precisa passar --vocab_size na mão no train.py.

Uso:
    python prepare_data.py --input corpus.txt --out_dir data/meucorpus
    python prepare_data.py --input corpus.txt --out_dir data/meucorpus \
        --tokenizer qwen3 --hf_model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import numpy as np


def dtype_for_vocab(vocab_size):
    if vocab_size <= 256:
        return np.uint8
    if vocab_size <= 65536:
        return np.uint16
    return np.uint32


def load_qwen3_tokenizer(hf_model):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "precisa de `transformers` instalado pra usar --tokenizer qwen3: "
            "pip install transformers"
        ) from e
    print(f"carregando tokenizer '{hf_model}' (baixa do HF Hub se não tiver em cache)...")
    tok = AutoTokenizer.from_pretrained(hf_model)
    return tok


def iter_line_chunks(path, lines_per_chunk=20000):
    """Lê o corpus em pedaços de N linhas por vez, pra não precisar tokenizar
    o arquivo inteiro de uma tacada só (corpora de dezenas de GB não cabem
    tokenizados na RAM de uma vez)."""
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= lines_per_chunk:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def tokenize_byte_level(input_path, out_path, dtype):
    """Byte-level: encode direto, sem chunking (é rápido e não precisa de
    modelo carregado; se o corpus for gigante, isso ainda escreve em uma
    tacada só via numpy — trocar por streaming se virar gargalo real)."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(dtype)
    data.tofile(out_path)
    return len(data)


def tokenize_qwen3(input_path, out_path, tok, eos_id):
    """BPE via HF tokenizers: processa em chunks de linhas, tokeniza em lote
    (mais rápido que chamada por linha) e vai escrevendo no .bin incrementalmente
    (append), sem acumular tudo tokenizado na RAM."""
    n_tokens = 0
    with open(out_path, "wb") as out_f:
        for chunk_text in iter_line_chunks(input_path):
            ids = tok.encode(chunk_text, add_special_tokens=False)
            if eos_id is not None:
                ids.append(eos_id)  # separa "documentos" (chunks de linhas) com EOS
            arr = np.array(ids, dtype=np.uint32)
            arr.tofile(out_f)
            n_tokens += len(arr)
            print(f"  ... {n_tokens:,} tokens processados", end="\r")
    print()
    return n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="arquivo .txt de entrada")
    ap.add_argument("--out_dir", type=str, default="data/default")
    ap.add_argument("--val_fraction", type=float, default=0.001)
    ap.add_argument("--tokenizer", type=str, default="byte", choices=["byte", "qwen3"])
    ap.add_argument("--hf_model", type=str, default="Qwen/Qwen3-8B",
                     help="checkpoint HF cujo tokenizer usar (só importa a versão do "
                          "tokenizer — modelos densos do Qwen3 compartilham o mesmo BPE)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    full_bin = os.path.join(args.out_dir, "_full.bin")

    if args.tokenizer == "byte":
        vocab_size = 256
        dtype = dtype_for_vocab(vocab_size)
        n = tokenize_byte_level(args.input, full_bin, dtype)
        tokenizer_name = "byte"
    else:
        tok = load_qwen3_tokenizer(args.hf_model)
        vocab_size = len(tok)
        dtype = dtype_for_vocab(vocab_size)
        eos_id = tok.eos_token_id
        n = tokenize_qwen3(args.input, full_bin, tok, eos_id)
        tokenizer_name = args.hf_model

    # separa train/val no nível de token (mesmo esquema de antes)
    data = np.memmap(full_bin, dtype=dtype, mode="r")
    n_val = max(1, int(n * args.val_fraction))
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    np.asarray(data[:-n_val]).tofile(train_path)
    np.asarray(data[-n_val:]).tofile(val_path)
    del data
    os.remove(full_bin)

    meta = {
        "vocab_size": int(vocab_size),
        "dtype": np.dtype(dtype).name,
        "tokenizer": tokenizer_name,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # mantido por compatibilidade com versões antigas do pipeline
    with open(os.path.join(args.out_dir, "vocab_size.txt"), "w") as f:
        f.write(str(vocab_size))

    n_train = n - n_val
    print(f"tokenizer: {tokenizer_name} | vocab_size: {vocab_size:,} | dtype: {meta['dtype']}")
    print(f"total de tokens: {n:,}")
    print(f"train: {n_train:,} tokens | val: {n_val:,} tokens")
    print(f"salvo em: {args.out_dir} (meta.json + train.bin + val.bin)")


if __name__ == "__main__":
    main()

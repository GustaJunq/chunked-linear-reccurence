# CLR — Chunked Linear Recurrence

Uma arquitetura de LLM sem atenção: mistura tokens via **recorrência linear
com gate de decaimento dependente do input**, mais um branch de convolução
causal curta para contexto local. Parente de Mamba / RWKV / GLA / HGRN, só que mais simples — estado diagonal por canal em vez de matriz por cabeça.

Duas propriedades que vêm de graça desse design:

- **Sem complexidade quadrática.** Custo linear em relação ao tamanho da
  sequência, tanto em treino quanto em inferência.
- **Sem KV cache que cresce.** Cada camada mantém um estado de tamanho
  **fixo** (`B, n_heads, head_dim`). Gerar o próximo token custa `O(1)`,
  não `O(n)` — o cache não incha conforme o contexto cresce, ao contrário
  de atenção padrão.

## Como funciona

Cada camada resolve, por canal, a recorrência

```
s_t = g_t · s_{t-1} + (1 - g_t) · v_t
```

onde `g_t` (o gate de decaimento) e `v_t` são projeções **dependentes do
token atual** — o modelo aprende, para cada canal e cada posição, o quanto
manter do estado anterior vs. o quanto substituir pelo valor novo.

- **Em treino:** resolvida em paralelo via *chunked scan* (log-space),
  sem loop Python token a token — vira operação matricial em blocos.
- **Em inferência:** resolvida como recorrência real, passo a passo, com
  estado de tamanho fixo por camada.

Os dois modos são matematicamente equivalentes (mesmos pesos, mesma saída
a menos de erro de ponto flutuante) e podem ser encadeados: o `generate()`
faz o *prefill* do prompt em modo paralelo (rápido) e só entra em modo
recorrente para os tokens novos gerados.

Um branch de convolução causal curta (`conv_kernel`, padrão 4) captura
n-gramas locais antes da recorrência — mesma ideia usada em Mamba/H3. O
estado da convolução (últimos `conv_kernel - 1` tokens) também é mantido
entre chamadas, então ele funciona igual em modo paralelo e em modo cache.

Cada bloco: `RMSNorm → CLR mixer → residual → RMSNorm → SwiGLU → residual`.

## Estrutura do repo

| Arquivo | O que faz |
|---|---|
| `model.py` | A arquitetura (`CustomMixer`, `Block`, `CustomLM`) |
| `prepare_data.py` | Corpus `.txt` → `train.bin` / `val.bin` + `meta.json` |
| `data.py` | Data loader via memmap (não carrega o corpus inteiro na RAM) |
| `train.py` | Treino DDP multi-GPU (bf16, grad accum, grad checkpointing) |

## Instalação

```bash
pip install torch numpy
# só se for usar --tokenizer qwen3 no prepare_data.py:
pip install transformers
```

## Uso

### 1. Preparar os dados

Tokenizador byte-level (padrão, zero dependências, bom pra prototipar):

```bash
python prepare_data.py --input corpus.txt --out_dir data/meucorpus
```

Tokenizador BPE do Qwen3 (~152k vocab, precisa de internet na primeira
chamada pra baixar o tokenizer do Hugging Face Hub):

```bash
python prepare_data.py --input corpus.txt --out_dir data/meucorpus \
    --tokenizer qwen3 --hf_model Qwen/Qwen3-8B
```

Isso escreve `train.bin`, `val.bin` e `meta.json` (com `vocab_size` e
`dtype` corretos) em `--out_dir`. O `train.py` lê o `meta.json`
automaticamente — não precisa passar `--vocab_size` na mão.

> Com vocabulário grande (Qwen3), o embedding + lm_head (weight-tied)
> domina os parâmetros em modelos pequenos. Com `dim=1024`/`n_layers=16`
> (defaults) já é ~43% dos parâmetros totais. Vale aumentar `dim` se quiser
> diluir essa proporção.

### 2. Treinar (1 nó, 6 GPUs)

```bash
torchrun --standalone --nproc_per_node=6 train.py --data_dir data/meucorpus
```

Principais flags (todas com default, veja `train.py` pra lista completa):

| Flag | Default | O quê |
|---|---|---|
| `--dim` | 1024 | dimensão do modelo |
| `--n_layers` | 16 | número de blocos |
| `--n_heads` | 16 | cabeças no mixer |
| `--conv_kernel` | 4 | tamanho do kernel da conv causal |
| `--chunk_size` | 64 | tamanho do chunk no scan paralelo |
| `--block_size` | 1024 | contexto em treino |
| `--grad_checkpointing` | off | troca memória por recomputação |

Roda em bf16 nativo (Ada Lovelace pra cima) sem `GradScaler`.

### 3. Gerar texto

```python
import torch
from model import CustomLM

ckpt = torch.load("checkpoints/ckpt_10000.pt")
m = CustomLM(vocab_size=ckpt["args"]["vocab_size"], **{
    k: ckpt["args"][k] for k in ("dim", "n_layers", "n_heads", "conv_kernel", "chunk_size")
})
m.load_state_dict(ckpt["model"])
m.eval()

idx = torch.zeros((1, 1), dtype=torch.long)  # ponto de partida
out = m.generate(idx, max_new_tokens=200, temperature=0.8, top_k=50)
```

## Caveats conhecidos

- **Numérico:** o chunked scan usa `exp(-cumsum(log(gate)))` internamente.
  Se o gate ficar perto de zero por muitos passos dentro de um chunk, isso
  pode estourar em bf16/fp16. Com `chunk_size=64` deve se comportar bem na
  prática; se aparecer `NaN` em treino real, tente `chunk_size` menor
  (16–32) primeiro.
- **Expressividade por camada:** estado diagonal (vetor `head_dim`), não
  matriz cheia como Mamba/S4 — mais barato, compensa-se com mais
  camadas/cabeças.

"""
Arquitetura própria da SynastrIA Networks — mixer de tokens por recorrência linear com gate
(decaimento dependente do input), sem atenção O(n^2).

Ideia central do bloco de mixagem (CustomMixer):
  - Um "gate" de decaimento por canal, dependente do token (input-dependent),
    controla quanto do estado anterior é mantido a cada passo.
  - Um branch de convolução causal curta captura n-gramas locais (contexto
    de vizinhança imediata, tipo Mamba/H3).
  - Em TREINO: computado em paralelo via "chunked scan" (log-space, sem loop
    Python token a token) — roda rápido em GPU como uma operação matricial.
  - Em INFERÊNCIA: computado como recorrência real, com estado de tamanho
    FIXO por camada (B, n_heads, head_dim) — nada de KV cache que cresce
    com o contexto. Cada token novo gerado custa O(1), não O(n).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = int(dim * hidden_mult * 2 / 3)  # padrão SwiGLU (estilo LLaMA)
        hidden = 64 * ((hidden + 63) // 64)       # arredonda pra múltiplo de 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CustomMixer(nn.Module):
    """
    Mixer de tokens: recorrência linear com gate de decaimento dependente do
    input, + branch de convolução causal curta para contexto local.

    Estado por cabeça: vetor de tamanho head_dim (diagonal — não é matriz
    cheia tipo Mamba/S4. Mais simples e barato, ao custo de menos
    expressividade por camada; compensa-se com mais camadas/cabeças.)
    """

    def __init__(self, dim, n_heads=8, conv_kernel=4, chunk_size=64):
        super().__init__()
        assert dim % n_heads == 0, "dim precisa ser divisível por n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.chunk_size = chunk_size

        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_gate = nn.Linear(dim, dim, bias=False)       # controla decaimento (0,1)
        self.to_out_gate = nn.Linear(dim, dim, bias=False)   # gate de saída (tipo "receptance" do RWKV)

        # branch de convolução causal curta (n-gramas locais). padding=0:
        # o padding causal é feito manualmente no forward (com um buffer de
        # estado), pra funcionar de forma IDÊNTICA em modo paralelo e cache.
        self.conv_kernel = conv_kernel
        self.conv = nn.Conv1d(dim, dim, kernel_size=conv_kernel, groups=dim, padding=0)

        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, state=None, use_cache=False):
        """
        x: (B, T, D)
        state: (ssm_state, conv_state) do passo anterior, ou None
            ssm_state:  (B, n_heads, head_dim)     — estado recorrente
            conv_state: (B, D, conv_kernel - 1)    — últimos tokens (pré-conv)
        use_cache=True  -> modo inferência (recorrência real, passo a passo)
        use_cache=False -> modo treino/prefill (chunked parallel scan)
        Os dois modos são numericamente equivalentes e podem ser encadeados
        (ex.: prefill em paralelo, depois geração em cache) desde que o
        estado retornado por um seja passado para o outro.
        """
        B, T, D = x.shape
        K = self.conv_kernel
        ssm_state, conv_state = state if state is not None else (None, None)

        x_t = x.transpose(1, 2)  # (B, D, T)
        if conv_state is None:
            conv_state = x_t.new_zeros(B, D, K - 1)
        x_padded = torch.cat([conv_state, x_t], dim=2)   # (B, D, K-1+T) — contexto causal real
        x_conv = self.conv(x_padded).transpose(1, 2)      # (B, T, D)
        new_conv_state = x_padded[:, :, -(K - 1):] if K > 1 else x_t.new_zeros(B, D, 0)

        v = self.to_v(x_conv)
        gate = torch.sigmoid(self.to_gate(x_conv))
        out_gate = torch.sigmoid(self.to_out_gate(x))

        v = v.view(B, T, self.n_heads, self.head_dim)
        gate = gate.view(B, T, self.n_heads, self.head_dim)

        if use_cache:
            if ssm_state is None:
                ssm_state = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                g_t = gate[:, t]
                v_t = v[:, t]
                ssm_state = g_t * ssm_state + (1 - g_t) * v_t
                outs.append(ssm_state)
            y = torch.stack(outs, dim=1)
            new_ssm_state = ssm_state
        else:
            y, new_ssm_state = self._chunked_scan(v, gate, ssm_state)

        y = y.reshape(B, T, D)
        y = y * out_gate
        # sempre devolve o estado completo (mesmo em modo paralelo/treino) —
        # necessário para encadear "prefill paralelo + geração sequencial"
        return self.out_proj(y), (new_ssm_state, new_conv_state)

    def _chunked_scan(self, v, gate, state):
        """
        Resolve s_t = g_t * s_{t-1} + (1-g_t) * v_t em paralelo, em blocos
        (chunks), pra evitar instabilidade numérica de cumprod/cumsum em
        sequências longas. Só há loop sequencial ENTRE chunks
        (T / chunk_size passos), nunca token a token.
        """
        B, T, H, Dh = v.shape
        C = self.chunk_size
        n_chunks = math.ceil(T / C)

        if state is None:
            state = torch.zeros(B, H, Dh, device=v.device, dtype=v.dtype)

        outs = []
        eps = 1e-6
        for i in range(n_chunks):
            s, e = i * C, min((i + 1) * C, T)
            g_c = gate[:, s:e].clamp(eps, 1 - eps)
            v_c = v[:, s:e]
            b_c = (1 - g_c) * v_c

            log_g = torch.log(g_c)
            log_g_cumsum = torch.cumsum(log_g, dim=1)
            decay_from_start = torch.exp(log_g_cumsum)

            carry = decay_from_start * state.unsqueeze(1)

            weighted_b = torch.exp(-log_g_cumsum) * b_c
            internal = decay_from_start * torch.cumsum(weighted_b, dim=1)

            chunk_out = carry + internal
            outs.append(chunk_out)
            state = chunk_out[:, -1]

        return torch.cat(outs, dim=1), state


class Block(nn.Module):
    def __init__(self, dim, n_heads, conv_kernel, chunk_size, mlp_mult=4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = CustomMixer(dim, n_heads, conv_kernel, chunk_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(self, x, state=None, use_cache=False):
        mixed, new_state = self.mixer(self.norm1(x), state, use_cache)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, new_state


class CustomLM(nn.Module):
    def __init__(self, vocab_size, dim=768, n_layers=12, n_heads=8,
                 conv_kernel=4, chunk_size=64, mlp_mult=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, conv_kernel, chunk_size, mlp_mult) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        # ligado externamente (train.py) via model._grad_checkpointing = True
        self._grad_checkpointing = False

        self.apply(self._init_weights)
        # escala as projeções de saída de cada bloco por 1/sqrt(2*n_layers),
        # estilo GPT-2 — evita que a variância do stream residual cresça
        # linearmente com a profundidade
        for block in self.blocks:
            for proj in (block.mixer.out_proj, block.mlp.w3):
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, states=None, use_cache=False):
        x = self.tok_emb(idx)
        do_ckpt = self._grad_checkpointing and self.training and not use_cache
        new_states = []
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            if do_ckpt:
                # use_reentrant=False: mais seguro com DDP/autocast, não exige
                # que todo tensor de entrada tenha requires_grad
                x, new_s = torch_checkpoint.checkpoint(
                    block, x, s, use_cache, use_reentrant=False
                )
            else:
                x, new_s = block(x, s, use_cache)
            new_states.append(new_s)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # devolve os estados sempre — em modo treino (use_cache=False) eles
        # vêm do chunked scan e servem, por ex., para prefill de geração
        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Geração autoregressiva.

        Prefill do prompt em modo PARALELO (use_cache=False -> chunked scan),
        que é muito mais rápido que rodar o prompt inteiro token a token.
        Só a partir daí entra no modo recorrente (use_cache=True), onde cada
        novo token custa O(1) e o estado tem tamanho fixo — sem KV cache que
        cresce com o contexto.
        """
        logits, _, states = self(idx, states=None, use_cache=False)
        logits = logits[:, -1]

        out = idx
        for _ in range(max_new_tokens):
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok], dim=1)
            logits, _, states = self(next_tok, states=states, use_cache=True)
            logits = logits[:, -1]

        return out

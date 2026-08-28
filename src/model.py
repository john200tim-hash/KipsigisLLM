import torch
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.dropout_p = config.get('dropout', 0.1)

        # Manual attention projection layers (gives us full control over the causal mask)
        self.q_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.k_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.v_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.out_proj = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(self.dropout_p)
        self.resid_drop = nn.Dropout(self.dropout_p)

        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(self.dropout_p),
        )

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_head

        # Layernorm first (Pre-LN is more stable than post-LN)
        x_norm = self.ln1(x)

        # Project to Q, K, V and split into heads
        q = self.q_proj(x_norm).view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention with CAUSAL MASK
        # This is the critical fix: tokens can only attend to PAST tokens, not future ones
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Combine heads and project back
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        # Residual connections
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCustomLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.pos_emb = nn.Embedding(config['block_size'], config['n_embd'])
        self.drop = nn.Dropout(config.get('dropout', 0.1))
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # Weight tying: share embedding and output head weights (saves parameters, improves quality)
        self.head.weight = self.tok_emb.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config['block_size'], "Sequence length exceeds block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

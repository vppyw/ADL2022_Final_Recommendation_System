import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
   AutoTokenizer,
   AutoModel
)

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.softmax = F.softmax
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch):
        attn_w = self.softmax(self.W(batch).squeeze(-1), dim=-1).unsqueeze(-1)
        ret = torch.sum(batch * attn_w, dim=1)
        return ret

class XLMRobertaExtractor(nn.Module):
    def __init__(self, model_path, max_length, cache_dir="./cache"):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(model_path,
                                                 output_hidden_states=True,
                                                 cache_dir=cache_dir)

        self.pooling = SelfAttentionPooling(input_dim=768)
        self.MLP = nn.Sequential(
                       nn.Linear(768, 1024),
                       nn.ReLU(),
                       nn.Linear(1024, 1024),
                       nn.ReLU(),
                       nn.Linear(1024, 768)
                   )
    
    def forward(self, seqs, device="cpu"):
        seqs = list(seqs)
        with torch.no_grad():
            tokenized_seqs = self.tokenizer(
                seqs,
                add_special_tokens=False,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt"
            )
            ids = tokenized_seqs["input_ids"].to(device)
            masks = tokenized_seqs["attention_mask"].to(device)
            outputs = self.encoder(ids, masks)
        hidden_states = outputs.last_hidden_state
        pooling = self.pooling(hidden_states)
        mlp_out = self.MLP(pooling)
        norm = mlp_out.norm(p=2, dim=1, keepdim=True)
        mlp_out_norm = mlp_out.div(norm.expand_as(mlp_out))
        return mlp_out_norm 

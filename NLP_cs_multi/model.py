import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
   AutoTokenizer,
   AutoModel
)

COURSE_NUM=728
SUBGROUP_NUM=149
FEATURE_NUM=312
# 312 for bert-chinese-tiny
# 768 for pert-chinese-base

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.softmax = F.softmax
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch):
        attn_w = self.softmax(self.W(batch).squeeze(-1), dim=-1).unsqueeze(-1)
        ret = torch.sum(batch * attn_w, dim=1)
        return ret

class BertExtractor(nn.Module):
    def __init__(self, model_path, max_length, cache_dir="./cache"):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(model_path,
                                                 output_hidden_states=True,
                                                 cache_dir=cache_dir)

        self.pooling = SelfAttentionPooling(input_dim=FEATURE_NUM)
        self.norm = nn.LayerNorm(FEATURE_NUM)
    
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
        output = self.pooling(hidden_states)
        output_norm = self.norm(output)
        return output_norm 

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, output_dim)
                  )

    def forward(self, x):
        return self.fc(x)

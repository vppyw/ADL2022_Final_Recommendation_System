import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch):
        attn_w = F.softmax(self.W(batch).squeeze(-1), dim=-1).unsqueeze(-1)
        ret = torch.sum(batch * attn_w, dim=1)
        return ret

class TransRec(nn.Module):
    def __init__(self,
                 num_feature,
                 output_dim,
                 hidden_dim,
                 embed_dim,
                 nhead,
                 num_layers,
                 padding_idx,
                 dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(num_feature+1,
                                  embed_dim,
                                  padding_idx=padding_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.pool = SelfAttentionPooling(input_dim=embed_dim)
        self.MLP = nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim)
                   )
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.embed.weight, mean=0, std=0.1)

    def forward(self, feature):
        """
        input: batch * user feature size
        output: batch * predict class size
        """
        embed = self.embed(feature.to(torch.int64))
        mask = (feature == 0)
        enc_out = self.encoder(embed,
                               src_key_padding_mask=mask)
        pool_out = self.pool(enc_out)
        output = self.MLP(pool_out)
        return output

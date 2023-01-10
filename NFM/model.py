import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_NFM(nn.Module):
    def __init__(self,
                 num_feature,
                 output_dim,
                 hidden_dim,
                 embed_dim,
                 padding_idx,
                 dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(num_feature+1,
                                        embed_dim,
                                        padding_idx=padding_idx)
        self.MLP = nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                   )
        self.linear = nn.Linear(num_feature, output_dim)
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.embed.weight, mean=0, std=0.1)

    def forward(self, feature):
        """
        input: batch * user feature size
        output: batch * predict class size
        """
        nonezero_embed = self.embed(feature.to(torch.int64))
        bi_inter = 0.5 * (nonezero_embed.sum(dim=1).pow(2) \
                    - nonezero_embed.pow(2).sum(dim=1))
        mlp_out = self.MLP(bi_inter)
        feature[torch.where(feature>0)] = 1.0
        linear_out = self.linear(feature)
        output = linear_out + mlp_out
        return output

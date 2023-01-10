import torch
import torch.nn as nn
import torch.nn.functional as F
import json
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
                 embed_dim,
                 nhead,
                 num_layers,
                 padding_idx,
                 dropout=0.0):
        super().__init__()
        
        ### Parameter
        item_feat_path = "./course_with_feature.json"
        self.item_feat_embed_dim = embed_dim // 4 * 3
        item_feat = json.load( open(item_feat_path, "r") )
        self.item_feat = torch.empty((0, 251)).cuda()
        for item in item_feat:
            self.item_feat = torch.cat(
                                ( self.item_feat, 
                                  torch.FloatTensor(item["feature"]).cuda().unsqueeze(dim=0)
                                ), 0 )
        self.output_dim = output_dim
        
        ### Embedding 
        self.feat_embed = nn.Embedding(num_feature+1,
                                       embed_dim,
                                       padding_idx=padding_idx)
        
        self.item_embed = nn.Embedding(output_dim,
                                       embed_dim - self.item_feat_embed_dim)
        self.item_feat_embed = nn.Embedding( 251+1,    
                                             self.item_feat_embed_dim,
                                             padding_idx=padding_idx)
        
        ### Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        
        item_feat_encoder_layer = nn.TransformerEncoderLayer(d_model=self.item_feat_embed_dim,
                                                            nhead=nhead,
                                                            dropout=dropout,
                                                            batch_first=True)
        self.item_feat_encoder = nn.TransformerEncoder(item_feat_encoder_layer,
                                                        num_layers=num_layers)
        
        self.pool = SelfAttentionPooling(input_dim=embed_dim)
        self.item_feat_pool = SelfAttentionPooling(input_dim=self.item_feat_embed_dim) 
        
        self.norm = nn.LayerNorm(embed_dim)
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.feat_embed.weight,
                        mean=0, std=0.1)
        nn.init.normal_(self.item_embed.weight,
                        mean=0, std=0.1)

    def forward(self, feature):
        """
        input: batch * user feature size
        output: batch * predict class size
        """
        # Encode User Information
        embed = self.feat_embed(feature.to(torch.int64))
        mask = (feature == 0)
        enc_out = self.encoder(embed,
                               src_key_padding_mask=mask)
        pool_out = self.pool(enc_out)
        pool_out_norm = self.norm(pool_out)
        # print(feature.shape, embed.shape, enc_out.shape, pool_out.shape)

        # Encode Course Information
        item_feat = self.item_feat
        item_feat_embed = self.item_feat_embed( item_feat.to(torch.int64) )
        mask = (item_feat == 0)
        item_enc_out = self.item_feat_encoder(item_feat_embed,
                                                  src_key_padding_mask = mask)
        item_pool_out = self.item_feat_pool(item_enc_out)

        item_out = self.item_embed(
                         torch.arange(0, self.output_dim)\
                        .to(feature.device)
                   )
        
        item_cat_out = torch.cat( (item_pool_out, item_out), 1 )
        # print( pool_out_norm.shape, item_pool_out.shape , item_out.shape, item_cat_out.shape)
        item_out_norm = self.norm(item_cat_out)
        # exit(0) 
        # Consine Similarity
        output = pool_out_norm.mm(
                    item_out_norm.transpose(0, 1)
                 )
        return output

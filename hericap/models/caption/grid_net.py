import torch
from torch import nn
from torch.nn import functional as F
from models.common.attention import MultiHeadAttention
from models.common.pos_embed import FeedForward


class TransformerLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__()

        self.mhatt = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

    def forward(self, q, k, v, mask=None):
        out = self.mhatt(q, k, v, mask)
        out = self.pwff(out)
        return out


class GridFeatureNetwork(nn.Module):

    def __init__(self, n_layers, d_in=1024, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_model) # 1024---->512
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads, d_ff, dropout, n_memories=n_memories) for _ in range(n_layers)]) # 3层

    # input:  outputs['gri_feat'] = [b (h w) c]    torch.Size([16, 60, 1024])    [B, H4*W4 ,C4 ]    
    # mask:  outputs['gri_mask'] = [b 1 1 (h w)]  torch.Size([16, 1, 1, 60])                         
    def forward(self, input, mask=None):
        out = self.layer_norm(self.dropout(F.relu(self.fc(input)))) # [b (h w) d_model]   1024---->512 

        outs = []
        for layer in self.layers:
            out = layer(out, out, out, mask)  
            outs.append(out.unsqueeze(1))     # [[b 1 (h w) d_model],.....]

        # torch.Size([16, 3, 60, 512])
        outs = torch.cat(outs, 1)  #  [[b 1 (h w) d_model],......] ---> [b, 3, (h w), d_model]
        
        #    [b, 3, (h w), d_model]     [b 1 1 (h w)]
        return outs, mask   
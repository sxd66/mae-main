from functools import partial

import torch
import torch.nn as nn
import random
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class Mae(nn.Module):
    def __init__(self,image_size=32,patch_size=4,in_c=3,embed_dim=64,depth=4,
                 num_head=4,mlp_ratio=2,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 mask_rate=0.75     ):
        super(Mae, self).__init__()
        self.patch=PatchEmbed(img_size=image_size,patch_size=patch_size,
                              embed_dim=embed_dim)
        self.tok=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.num_tok=(image_size//patch_size)**2
        self.mask_rate=mask_rate
        self.pos_emb=nn.Parameter(torch.zeros(1,self.num_tok+1,embed_dim))
        self.blk=nn.ModuleList([
            Block(embed_dim,num_head,mlp_ratio,qkv_bias=True,norm_layer=norm_layer)
           for i in range(depth)             ] )
        self.norm=norm_layer(embed_dim)
    def random_msak(self,x,mask_ratio):
        num=x.shape[1]
        mask_num=int(num*mask_ratio)
        random_numbers = sorted(random.sample(range(num), mask_num))
        x=x[:,random_numbers,:]
        return x,random_numbers

    def forward_encoder(self,x):
        x=self.patch(x)
        x=x+self.pos_emb[:,1:,:]
        x,mask=self.random_msak(x,mask_ratio=self.mask_rate)
        tok=self.tok+self.pos_emb[:,:1,:]
        tok=tok.expand(x.shape[0],-1,-1)
        x=torch.cat((tok,x),dim=1)

        for blk in self.blk:
            x=blk(x)
        x=self.norm(x)
        return x,mask


if __name__=="__main__":
    x=torch.rand(64,3,32,32)
    model=Mae()
    y=model.forward_encoder(x)
    print(y)

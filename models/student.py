import torch
from torch import nn
import torch.nn.functional as F
from sgof.geometry.transforms import pose9_to_c2w, relative_to_first


class Student(nn.Module):
    """Vision-only Eq.8--12 reference model; teacher embeddings never enter forward()."""
    def __init__(self,cfg):
        super().__init__()
        cfg.validate(); self.cfg=cfg
        d=cfg.embed_dim
        self.patch=nn.Conv2d(3,d,cfg.patch_size,stride=cfg.patch_size)
        n=(cfg.image_size//cfg.patch_size)**2
        self.position=nn.Parameter(torch.zeros(1,n,d))
        nn.init.trunc_normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(d,cfg.attention_heads,4*d,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,cfg.transformer_layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d)
        pose_layers=[nn.Linear(d,cfg.pose_hidden),nn.ReLU()]
        if cfg.pose_layers==3:
            pose_layers.extend([nn.Linear(cfg.pose_hidden,cfg.pose_hidden),nn.ReLU()])
        pose_layers.append(nn.Linear(cfg.pose_hidden,9))
        self.pose=nn.Sequential(*pose_layers)
        nn.init.zeros_(self.pose[-1].weight)
        with torch.no_grad():
            self.pose[-1].bias.copy_(torch.tensor([1.,0.,0.,0.,1.,0.,0.,0.,0.]))
        # Optional calibration extension, excluded from Eq.13 unless weight > 0.
        self.intrinsics=nn.Linear(d,4)
        nn.init.zeros_(self.intrinsics.weight); nn.init.zeros_(self.intrinsics.bias)
        self.seed=nn.Linear(d,cfg.seed_channels*4*4*4)
        decoders=[]; channels=cfg.seed_channels
        for c in cfg.decoder_channels:
            decoders.append(nn.Sequential(nn.Conv3d(channels,c,3,padding=1),nn.BatchNorm3d(c),nn.ReLU(),
                                          nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c),nn.ReLU()))
            channels=c
        self.decoders=nn.ModuleList(decoders)
        self.occupancy=nn.Conv3d(channels,1,1)

    def forward(self,images):
        if images.ndim!=5 or images.shape[2:]!=(3,self.cfg.image_size,self.cfg.image_size):
            raise ValueError("Expected (B,V,3,image_size,image_size)")
        b,v=images.shape[:2]
        if self.cfg.input_normalization=="imagenet":
            mean=images.new_tensor([.485,.456,.406])[None,None,:,None,None]
            std=images.new_tensor([.229,.224,.225])[None,None,:,None,None]
            images=(images-mean)/std
        tokens=self.patch(images.reshape(b*v,*images.shape[2:])).flatten(2).transpose(1,2)
        embedding=self.norm(self.encoder(tokens+self.position)).mean(1).reshape(b,v,-1)
        absolute=pose9_to_c2w(self.pose(embedding))
        poses=relative_to_first(absolute)
        scene=embedding.mean(1)
        volume=self.seed(scene).reshape(b,self.cfg.seed_channels,4,4,4)
        for stage,size in zip(self.decoders,(self.cfg.grid_size//4,self.cfg.grid_size//2,self.cfg.grid_size)):
            volume=stage(F.interpolate(volume,size=(size,)*3,mode="trilinear",align_corners=False))
        intr=self.intrinsics(embedding)
        intr=torch.cat([intr[...,:2].exp(),intr[...,2:].sigmoid()],dim=-1)
        return {"logits":self.occupancy(volume).squeeze(1),"c2w":poses,"intrinsics_normalized":intr}

import torch
from torch import nn


class SaZMod(nn.Module):
    """Eq.3--7, shared per-slice refinement, affine language modulation and residual."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        c=cfg.teacher_channels
        self.refine=nn.Sequential(nn.Conv3d(c,c,(1,3,3),padding=(0,1,1)),nn.ReLU(),
                                  nn.Conv3d(c,c,(1,3,3),padding=(0,1,1)),
                                  nn.BatchNorm3d(c) if cfg.batch_norm else nn.Identity())
        z=cfg.grid_size if cfg.modulation=="zwise" else 1
        self.affine=nn.Sequential(nn.Linear(cfg.text_dim,cfg.modulation_hidden),nn.ReLU(),
                                  nn.Linear(cfg.modulation_hidden,2*z*c))

    def forward(self,geometry,text):
        if self.cfg.modulation=="none":
            return geometry
        b,c,z,y,x=geometry.shape
        if z!=self.cfg.grid_size or text.shape!=(b,self.cfg.text_dim):
            raise ValueError("SaZ volume/text dimensions differ from configuration")
        # No cross-Z convolution; shared BN statistics are computed over folded slices.
        slices=geometry.permute(0,2,1,3,4).reshape(b*z,c,1,y,x)
        refined=self.refine(slices).reshape(b,z,c,y,x).permute(0,2,1,3,4)
        nz=z if self.cfg.modulation=="zwise" else 1
        affine=self.affine(text).reshape(b,2,nz,c)
        gamma,beta=[affine[:,i].permute(0,2,1)[:,:,:,None,None] for i in (0,1)]
        output=gamma*refined+beta
        return output+geometry if self.cfg.residual else output


class OccupancyDecoder(nn.Module):
    """Supplementary calibrated decoder; native VGGT has no scalar occupancy head."""
    def __init__(self,channels):
        super().__init__()
        hidden=max(8,channels//2)
        self.network=nn.Sequential(nn.Conv3d(channels,hidden,1),nn.ReLU(),nn.Conv3d(hidden,1,1))

    def forward(self,volume):
        return self.network(volume).squeeze(1)

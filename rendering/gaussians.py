import math
import torch
from sgof.geometry.transforms import grid_centers


def sample_primitives(occupancy,bounds,tau=.5,scale=.1,max_points=None,seed=0):
    """Eq.1 initialization: one voxel-center point per sigma > tau, no silent pruning."""
    if occupancy.ndim!=3 or not 0<=tau<=1:
        raise ValueError("Expected scalar ZYX occupancy and tau in [0,1]")
    if not torch.isfinite(occupancy).all() or (occupancy<0).any() or (occupancy>1).any():
        raise ValueError("Occupancy must be finite probabilities")
    xyz=grid_centers(bounds,occupancy.shape,occupancy.device,occupancy.dtype)
    mask=occupancy>tau
    positions=xyz[mask]; confidence=occupancy[mask]
    original=len(positions)
    if max_points is not None and len(positions)>max_points:
        if max_points<1:
            raise ValueError("Point cap must be positive")
        generator=torch.Generator(device=occupancy.device).manual_seed(seed)
        index=torch.randperm(len(positions),generator=generator,device=occupancy.device)[:max_points]
        positions,confidence=positions[index],confidence[index]
    return {"positions":positions,"confidence":confidence,
            "scales":torch.full_like(positions,scale),"original_count":original,"retained_count":len(positions)}


def colors_from_views(positions,images,c2w,intrinsics):
    """Average RGB projected from context views; unseen points use neutral gray."""
    if not len(positions):
        return positions.clone()
    v,_,h,w=images.shape
    total=torch.zeros_like(positions); counts=torch.zeros(len(positions),device=positions.device)
    for i in range(v):
        world_to_camera=torch.linalg.inv(c2w[i])
        p=positions@world_to_camera[:3,:3].T+world_to_camera[:3,3]
        pixel=p@intrinsics[i].T
        xy=pixel[:,:2]/pixel[:,2:3].clamp_min(1e-8)
        x,y=xy[:,0].round().long(),xy[:,1].round().long()
        valid=(p[:,2]>1e-4)&(x>=0)&(x<w)&(y>=0)&(y<h)
        total[valid]+=images[i,:,y[valid],x[valid]].T
        counts[valid]+=1
    return torch.where(counts[:,None]>0,total/counts.clamp_min(1)[:,None],torch.full_like(total,.5))


def rasterize_reference(positions,colors,opacities,scales,c2w,intrinsics,height,width,
                        background=(0.,0.,0.),pixel_chunk=2048,max_gaussians=4096):
    """Differentiable anisotropic projection and front-to-back alpha compositing.

    O(M*H*W) reference backend, not the tiled CUDA 3DGS renderer. No densification,
    SH view dependence or visibility culling acceleration. For tests/small scenes only.
    """
    if len(positions)>max_gaussians:
        raise ValueError("Reference renderer point budget exceeded; use an official CUDA backend for real experiments")
    device,dtype=positions.device,positions.dtype
    background=torch.tensor(background,device=device,dtype=dtype)
    if not len(positions):
        return background[:,None,None].expand(3,height,width).clone()
    w2c=torch.linalg.inv(c2w)
    p=positions@w2c[:3,:3].T+w2c[:3,3]
    valid=p[:,2]>1e-4
    p,colors,opacities,scales=p[valid],colors[valid],opacities[valid],scales[valid]
    if not len(p):
        return background[:,None,None].expand(3,height,width).clone()
    order=p[:,2].argsort()
    p,colors,opacities,scales=p[order],colors[order],opacities[order],scales[order]
    x,y,z=p.unbind(-1)
    fx,fy,cx,cy=intrinsics[0,0],intrinsics[1,1],intrinsics[0,2],intrinsics[1,2]
    means=torch.stack([fx*x/z+cx,fy*y/z+cy],-1)
    zeros=torch.zeros_like(z)
    jac=torch.stack([fx/z,zeros,-fx*x/z**2,zeros,fy/z,-fy*y/z**2],-1).reshape(-1,2,3)
    covariance=torch.diag_embed(scales.square())
    rotation=w2c[:3,:3]
    projected=jac@(rotation@covariance@rotation.T)@jac.transpose(-1,-2)
    projected=projected+torch.eye(2,device=device,dtype=dtype)[None]*.3
    precision=torch.linalg.inv(projected)
    yy,xx=torch.meshgrid(torch.arange(height,device=device,dtype=dtype),torch.arange(width,device=device,dtype=dtype),indexing="ij")
    pixels=torch.stack([xx,yy],-1).reshape(-1,2)
    parts=[]
    for chunk in pixels.split(pixel_chunk):
        delta=chunk[None]-means[:,None]
        quadratic=torch.einsum("mpi,mij,mpj->mp",delta,precision,delta)
        alpha=(opacities[:,None]*torch.exp(-.5*quadratic)).clamp(0,.99)
        transmission=torch.cumprod(torch.cat([torch.ones(1,len(chunk),device=device,dtype=dtype),1-alpha],0),0)
        rgb=(alpha*transmission[:-1]).T@colors+transmission[-1,:,None]*background
        parts.append(rgb)
    return torch.cat(parts).reshape(height,width,3).permute(2,0,1)


def export_gaussian_ply(path,primitives,colors,initial_opacity=.1):
    """Degree-0 PLY compatible with original 3DGS load_ply, isotropic initial covariance."""
    from pathlib import Path
    import numpy as np
    p=primitives["positions"].detach().cpu().numpy()
    s=primitives["scales"].detach().cpu().numpy()
    if np.any(s<=0) or not 0<initial_opacity<1:
        raise ValueError("Positive scales and opacity strictly in (0,1) required")
    dc=(colors.detach().cpu().numpy()-.5)/.28209479177387814
    names=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="ascii") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(p)}\n")
        for name in names:f.write(f"property float {name}\n")
        f.write("end_header\n")
        for point,color,scale in zip(p,dc,s):
            row=[*point,0,0,0,*color,math.log(initial_opacity/(1-initial_opacity)),*np.log(scale),1,0,0,0]
            f.write(" ".join(f"{x:.9g}" for x in row)+"\n")

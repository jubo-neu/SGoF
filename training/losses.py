import torch
import torch.nn.functional as F
from sgof.geometry.transforms import c2w_to_pose9


def distillation_loss(pred,target,cfg):
    occ=F.binary_cross_entropy_with_logits(pred["logits"],target["occupancy"])
    # Eq.15 averages over views/samples, sums over the pose-vector coordinates.
    pose=((c2w_to_pose9(pred["c2w"])-c2w_to_pose9(target["c2w"]))**2).sum(-1).mean()
    intr=torch.zeros_like(occ)
    if cfg.lambda_intrinsics:
        intr=F.mse_loss(pred["intrinsics_normalized"],target["intrinsics_normalized"])
    total=cfg.lambda_occ*occ+cfg.lambda_pose*pose+cfg.lambda_intrinsics*intr
    return total,{"occupancy":float(occ.detach()),"pose":float(pose.detach()),"intrinsics":float(intr.detach())}


def ssim_torch(x,y,window=11):
    """Gaussian-window SSIM in RGB [0,1], valid window region."""
    if x.shape!=y.shape or x.ndim!=4 or min(x.shape[-2:])<window:
        raise ValueError("SSIM requires matching BCHW images with size >= window")
    r=torch.arange(window,device=x.device,dtype=x.dtype)-(window-1)/2
    g=torch.exp(-r*r/(2*1.5**2)); g=g/g.sum()
    kernel=(g[:,None]*g[None,:]).expand(x.shape[1],1,window,window)
    filt=lambda a:F.conv2d(a,kernel,groups=x.shape[1])
    mx,my=filt(x),filt(y)
    vx,vy=filt(x*x)-mx*mx,filt(y*y)-my*my
    cov=filt(x*y)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()


def rendering_loss(pred,target,lambda_ssim=.2):
    l1=(pred-target).abs().mean()
    dssim=(1-ssim_torch(pred,target))/2
    return (1-lambda_ssim)*l1+lambda_ssim*dssim

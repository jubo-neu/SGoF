import math
import numpy as np
import torch
from sgof.training.losses import ssim_torch


def psnr(pred,target,mask=None):
    if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[0]!=3:
        raise ValueError("PSNR expects matching RGB CHW")
    squared=(pred-target).square()
    if mask is not None:
        if mask.shape!=pred.shape[-2:]:raise ValueError("Mask shape mismatch")
        if not mask.any():return None
        squared=squared[:,mask]
    mse=float(squared.mean())
    return math.inf if mse==0 else -10*math.log10(mse)


class ImageMetrics:
    def __init__(self,lpips=False,device="cpu"):
        self.device=device;self.lpips=None
        if lpips:
            try:
                import lpips as package
                self.lpips=package.LPIPS(net="alex").to(device).eval()
            except Exception as error:
                raise RuntimeError("Official LPIPS dependencies/weights unavailable. Install requirements-optional.txt; no proxy LPIPS is reported.") from error

    def __call__(self,pred,target,masks=None):
        pred,target=pred.to(self.device),target.to(self.device)
        if min(pred.shape[-2:])<11:raise ValueError("Metric images must be at least 11x11")
        with torch.inference_mode():
            result={"psnr":psnr(pred,target),"ssim":float(ssim_torch(pred[None],target[None])),"lpips":None}
            if self.lpips is not None:
                result["lpips"]=float(self.lpips(pred[None]*2-1,target[None]*2-1).mean())
            for name,mask in (masks or {}).items():
                result[f"{name}_psnr"]=psnr(pred,target,mask.to(self.device))
                result[f"{name}_pixels"]=int(mask.sum())
        return result


def paired_bootstrap(differences,seed=0,repeats=10000):
    values=np.asarray(differences,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite paired observations required")
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,size=len(values),replace=True).mean() for _ in range(repeats)])
    lo,hi=np.quantile(means,[.025,.975])
    return {"mean_difference":float(values.mean()),"ci95":[float(lo),float(hi)],"n_pairs":len(values),
            "caution":"Four evaluation scenes provide limited scene-level statistical power."}

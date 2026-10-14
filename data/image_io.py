from pathlib import Path
import cv2
import numpy as np
import torch


def read_rgb(path,size=None):
    image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f"Invalid image: {path}")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if size is not None:image=cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image.copy()).permute(2,0,1).float()/255


def write_rgb(path,image):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    image=(image.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    ok,array=cv2.imencode(path.suffix,cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    if not ok:raise ValueError("Image encoding failed")
    array.tofile(str(path))



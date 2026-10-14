from pathlib import Path
import csv
import math
import cv2
import numpy as np
from collections import defaultdict
from sgof.data.image_io import read_rgb
from sgof.utils.common import save_json
from sgof.evaluation.image_metrics import ImageMetrics


def mask_read(path,shape):
    raw=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape!=shape:raise ValueError(f"Invalid metric mask {path}; no silent resizing")
    import torch
    return torch.from_numpy(raw>0)


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return "inf" if value>0 else "-inf" if value<0 else None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    return value


def evaluate_renders(manifest,output,with_lpips=False,device="cpu"):
    path=Path(manifest).resolve()
    with path.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    required={"method","scene","view","seed","protocol","prediction","target"}
    if not rows or not required.issubset(rows[0]):raise ValueError("Incomplete evaluation manifest")
    calc=ImageMetrics(with_lpips,device);results=[];seen=set();groups=defaultdict(set)
    for row in rows:
        key=tuple(row[k] for k in ("method","scene","view","seed","protocol"))
        if key in seen:raise ValueError("Duplicate evaluation record")
        seen.add(key)
        groups[(row["method"],row["seed"],row["protocol"])].add((row["scene"],row["view"]))
        resolve=lambda value:Path(value) if Path(value).is_absolute() else path.parent/value
        prediction,target=read_rgb(resolve(row["prediction"])),read_rgb(resolve(row["target"]))
        if prediction.shape!=target.shape:raise ValueError("Prediction/target image shape mismatch")
        masks={name:mask_read(resolve(row[name+"_mask"]),target.shape[-2:]) for name in ("sky","far","boundary") if row.get(name+"_mask")}
        results.append({k:row[k] for k in ("method","scene","view","seed","protocol","category") if k in row}|calc(prediction,target,masks))
    # No missing views may be silently dropped from a paired leaderboard.
    by_protocol=defaultdict(list)
    for (method,seed,protocol),views in groups.items():by_protocol[protocol].append(views)
    if any(any(v!=sets[0] for v in sets[1:]) for sets in by_protocol.values()):
        raise ValueError("Methods/seeds must evaluate identical scene/view sets within each protocol")
    scene_groups=defaultdict(list)
    for r in results:scene_groups[(r["method"],r["scene"],r["seed"],r["protocol"])].append(r)
    aggregated=[]
    for (method,scene,seed,protocol),records in scene_groups.items():
        result={"method":method,"scene":scene,"seed":seed,"protocol":protocol,"views":len(records)}
        for metric in ("psnr","ssim","lpips","sky_psnr","far_psnr","boundary_psnr"):
            values=[r[metric] for r in records if r.get(metric) is not None]
            result[metric]=float(np.mean(values)) if values else None
            result[metric+"_valid_views"]=len(values)
        aggregated.append(result)
    save_json(Path(output)/"per_view.json",json_safe(results))
    save_json(Path(output)/"per_scene_seed.json",json_safe(aggregated))
    save_json(Path(output)/"evaluation_protocol.json",{"metric_domain":"sRGB [0,1]","ssim":"Gaussian 11x11 sigma1.5, valid windows, RGB mean",
               "lpips":"AlexNet official" if with_lpips else "not computed", "aggregation":"per-image, then scene; report scene-balanced seed means"})
    return aggregated

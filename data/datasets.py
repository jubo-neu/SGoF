from pathlib import Path
import torch
from torch.utils.data import Dataset
from sgof.utils.common import load_json, load_tensor_file
from sgof.data.contracts import validate_cache


class DistillationDataset(Dataset):
    def __init__(self,manifest,split,model_cfg,allow_demo=False):
        path=Path(manifest).resolve(); self.root=path.parent
        rows=load_json(path)["samples"]
        self.rows=[row for row in rows if row["split"]==split]
        if not self.rows:raise ValueError(f"No {split} samples")
        self.cfg=model_cfg;self.allow_demo=allow_demo
        all_ids=[r["id"] for r in rows]
        if len(all_ids)!=len(set(all_ids)):raise ValueError("Duplicate sample IDs")
        scene_splits={}
        for row in rows:scene_splits.setdefault(row["scene"],set()).add(row["split"])
        if any(len(s)>1 for s in scene_splits.values()):
            raise ValueError("Scene-disjoint distillation required; do not mix scene train/test windows")

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        path=Path(row["cache"]);path=path if path.is_absolute() else self.root/path
        cache=load_tensor_file(path);validate_cache(cache,self.cfg.grid_size)
        provenance=cache["provenance"]
        if provenance.get("kind")=="synthetic_demo" and not self.allow_demo:
            raise ValueError("Synthetic teacher cache requires --allow-demo")
        if provenance.get("uses_test_views",True):
            raise ValueError("Teacher targets must certify uses_test_views=false")
        images=cache["images"]
        if images.shape[-2:]!=(self.cfg.image_size,)*2:
            images=torch.nn.functional.interpolate(images,size=self.cfg.image_size,mode="bilinear",align_corners=False)
        return {"images":images,"occupancy":cache["occupancy"],"c2w":cache["c2w"],
                "intrinsics_normalized":cache["intrinsics_normalized"]}

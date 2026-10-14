"""Fuse real pretrained VGGT exports into a SaZ cache; does not fabricate VGGT outputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sgof.config import load_config
from sgof.geometry.transforms import fuse_features
from sgof.data.image_io import read_rgb
from sgof.utils.common import load_json, load_tensor_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry-npz",required=True,help="points Qx3, features QxC, depth Q, c2w Vx4x4, intrinsics_normalized Vx4")
    p.add_argument("--context",required=True);p.add_argument("--text-embedding",required=True)
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--vggt-checkpoint-id",required=True);p.add_argument("--clip-checkpoint-id",required=True)
    args=p.parse_args();cfg,_=load_config(args.config)
    manifest=Path(args.context).resolve();context=load_json(manifest)
    if context.get("uses_test_views",True):raise ValueError("Cannot export test views to teacher cache")
    geometry=np.load(args.geometry_npz,allow_pickle=False)
    required={"points","features","depth","c2w","intrinsics_normalized"}
    if not required.issubset(geometry.files):raise ValueError(f"NPZ missing {required-set(geometry.files)}")
    points,features,depth=[torch.from_numpy(geometry[k]).float() for k in ("points","features","depth")]
    if features.shape[1]!=cfg.teacher_channels:raise ValueError("Feature channels must match declared pretrained projection; no random projection")
    poses=torch.from_numpy(geometry["c2w"]).float()
    anchor=torch.linalg.inv(poses[0])
    points=points@anchor[:3,:3].T+anchor[:3,3]
    poses=anchor@poses
    bounds=context["bounds"]
    volume,support=fuse_features(points,features,depth,bounds,(cfg.grid_size,)*3)
    embedding=load_tensor_file(args.text_embedding)
    if not isinstance(embedding,torch.Tensor) or embedding.shape!=(cfg.text_dim,):raise ValueError("Expected actual frozen CLIP text embedding vector")
    images=torch.stack([read_rgb(manifest.parent/x,cfg.image_size) for x in context["images"]])
    cache={"features":volume,"support":support,"text_embedding":embedding,"images":images,"c2w":poses,
           "intrinsics_normalized":torch.from_numpy(geometry["intrinsics_normalized"]).float(),"bounds":torch.tensor(bounds),
           "provenance":{"kind":"pretrained_exports","scene_id":context.get("scene_id"),"uses_test_views":False,"vggt_checkpoint":args.vggt_checkpoint_id,
                         "clip_checkpoint":args.clip_checkpoint_id,"context_images":context["images"],
                         "feature_alignment":"User must verify points/features/depth correspond pixel-for-pixel."}}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(cache,output)
    print(f"Saved {output}; occupancy decoder calibration is a separate disclosed stage.")


if __name__=="__main__":main()

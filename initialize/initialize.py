from pathlib import Path
import torch
from sgof.config import ModelConfig
from sgof.models import Student
from sgof.data.image_io import read_rgb
from sgof.rendering.gaussians import sample_primitives, colors_from_views, export_gaussian_ply
from sgof.training.teacher import intrinsics_matrix
from sgof.utils.common import load_tensor_file, load_json, save_json


def initialize_student(checkpoint,context_manifest,output,device="cpu",tau=.5,max_points=None,allow_demo=False):
    state=load_tensor_file(checkpoint)
    if state.get("synthetic_demo") and not allow_demo:raise ValueError("Demo checkpoint requires --allow-demo")
    cfg=ModelConfig(**state["config"]["model"])
    model=Student(cfg).to(device);model.load_state_dict(state["model"],strict=True);model.eval()
    manifest=Path(context_manifest).resolve();context=load_json(manifest)
    if context.get("uses_test_views",True):raise ValueError("Context manifest must certify uses_test_views=false")
    paths=[Path(p) if Path(p).is_absolute() else manifest.parent/p for p in context["images"]]
    images=torch.stack([read_rgb(p,cfg.image_size) for p in paths]).to(device)
    with torch.inference_mode():pred=model(images[None])
    probability=pred["logits"][0].sigmoid();poses=pred["c2w"][0]
    # Bounds are a fixed experiment convention in first-camera coordinates, not test geometry.
    bounds=context.get("bounds",[[-1.,-1.,.1],[1.,1.,4.]])
    points=sample_primitives(probability,bounds,tau,max_points=max_points)
    source="predicted_calibration_extension"
    if "intrinsics_normalized" in context:
        intr=torch.tensor(context["intrinsics_normalized"],device=device,dtype=torch.float32)
        source="provided_calibration"
    else:
        if state["config"]["train"].get("lambda_intrinsics",0)==0:
            raise ValueError("Pure-RGB deployment needs the optional calibrated intrinsics head (lambda_intrinsics>0); otherwise provide known calibration explicitly")
        intr=pred["intrinsics_normalized"][0]
    matrices=intrinsics_matrix(intr,cfg.image_size,cfg.image_size)
    colors=colors_from_views(points["positions"],images,poses,matrices)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    export_gaussian_ply(output/"initial_gaussians.ply",points,colors)
    torch.save({"occupancy":probability.cpu(),"c2w":poses.cpu(),"intrinsics":matrices.cpu(),
                "positions":points["positions"].cpu(),"colors":colors.cpu(),"bounds":torch.tensor(bounds)},output/"initialization.pt")
    save_json(output/"initialization_report.json",{"points":points["retained_count"],"before_cap":points["original_count"],
               "status":"empty_occupancy_initialization" if points["retained_count"]==0 else "exported",
               "point_cap":max_points,"tau":tau,"scale_std":.1,"pose_representation":"SO3 rotation6D + translation3D; first-camera gauge",
               "intrinsics_source":source,"checkpoint_synthetic_demo":state.get("synthetic_demo",False),
               "note":"PLY does not contain scene cameras; official 3DGS needs the provided camera adapter and degree=0 on load."})

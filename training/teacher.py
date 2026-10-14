"""Cached teacher training, with explicit supplementary occupancy-decoder calibration.

VGGT, CLIP and LLaVA must run externally using their licensed pretrained weights.
This module never substitutes random geometry/text features in a real run.
"""
from pathlib import Path
import torch
from sgof.models import SaZMod, OccupancyDecoder
from sgof.training.losses import rendering_loss
from sgof.geometry.transforms import grid_centers
from sgof.rendering.gaussians import rasterize_reference
from sgof.utils.common import load_tensor_file, save_json, seed_everything


def load_geometry_cache(path,cfg,allow_demo=False):
    item=load_tensor_file(path)
    required={"features","support","text_embedding","images","c2w","intrinsics_normalized","bounds","provenance"}
    if not required.issubset(item):raise ValueError(f"Geometry cache missing {required-set(item)}")
    if item["features"].shape!=(cfg.teacher_channels,cfg.grid_size,cfg.grid_size,cfg.grid_size):
        raise ValueError("Cached geometry volume shape mismatch")
    if item["text_embedding"].shape!=(cfg.text_dim,):raise ValueError("Text shape mismatch")
    if item["provenance"].get("kind")=="synthetic_demo" and not allow_demo:raise ValueError("Demo features prohibited in real teacher runs")
    if item["provenance"].get("uses_test_views",True):raise ValueError("No held-out images may enter teacher caches")
    for k,v in item.items():
        if isinstance(v,torch.Tensor) and not torch.isfinite(v).all():raise ValueError(f"Nonfinite {k}")
    return item


def intrinsics_matrix(normalized,height,width):
    output=torch.zeros(*normalized.shape[:-1],3,3,device=normalized.device,dtype=normalized.dtype)
    output[...,0,0]=normalized[...,0]*width
    output[...,1,1]=normalized[...,1]*height
    output[...,0,2]=normalized[...,2]*width
    output[...,1,2]=normalized[...,3]*height
    output[...,2,2]=1
    return output


def calibrate_decoder(cache_paths,output,cfg,steps=100,device="cpu",allow_demo=False):
    """Supplementary stage absent from PDF: calibrate scalar decoder from geometry support.

    Support target = support/(support+1), an explicitly disclosed geometry proxy,
    not ground-truth occupancy and not an official VGGT pretrained occupancy decoder.
    """
    seed_everything(0)
    records=[load_geometry_cache(p,cfg,allow_demo) for p in cache_paths]
    decoder=OccupancyDecoder(cfg.teacher_channels).to(device)
    opt=torch.optim.Adam(decoder.parameters(),lr=1e-3)
    history=[]
    for step in range(steps):
        item=records[step%len(records)]
        support=item["support"].to(device)
        target=support/(support+1)
        opt.zero_grad(set_to_none=True)
        logits=decoder(item["features"][None].to(device))[0]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
        loss.backward();opt.step();history.append(float(loss.detach()))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.save({"decoder":decoder.state_dict(),"channels":cfg.teacher_channels,"proxy":"support/(support+1)",
                "supplementary_stage":True,"synthetic_demo":allow_demo},output/"decoder.pt")
    save_json(output/"decoder_calibration.json",{"loss":history,"warning":"Independent supplementary decoder, not the frozen native VGGT decoder claimed in the draft."})


def fit_teacher(cache_paths,decoder_checkpoint,output,cfg,steps=100,device="cpu",allow_demo=False,
                max_render_points=512,text_mode="correct",seed=0):
    if cfg.modulation=="none":
        steps=0
    seed_everything(seed)
    items=[load_geometry_cache(p,cfg,allow_demo) for p in cache_paths]
    state=load_tensor_file(decoder_checkpoint)
    if state.get("synthetic_demo") and not allow_demo:raise ValueError("Synthetic decoder prohibited in real runs")
    decoder=OccupancyDecoder(cfg.teacher_channels).to(device)
    decoder.load_state_dict(state["decoder"],strict=True);decoder.eval()
    for p in decoder.parameters():p.requires_grad_(False)
    modulator=SaZMod(cfg).to(device)
    optimizer=torch.optim.Adam(modulator.parameters(),lr=1e-4)
    history=[]
    if text_mode not in {"correct","zero","shuffled"}:raise ValueError("Unknown text control")
    texts=[item["text_embedding"] for item in items]
    if text_mode=="shuffled" and len(items)<2:raise ValueError("Shuffled-text control requires at least two scenes")
    shuffle_map=[]
    if text_mode=="shuffled":
        scenes=[item["provenance"].get("scene_id") for item in items]
        if any(scene is None for scene in scenes):raise ValueError("Shuffled-text control requires provenance.scene_id")
        for index,scene in enumerate(scenes):
            candidates=[i for i,other in enumerate(scenes) if other!=scene]
            if not candidates:raise ValueError("Shuffled text must originate from another scene")
            shuffle_map.append(candidates[index%len(candidates)])
    def embedding(index):
        if text_mode=="zero":return torch.zeros_like(texts[index])
        return texts[shuffle_map[index]] if text_mode=="shuffled" else texts[index]
    for step in range(steps):
        index=step%len(items);item=items[index]
        features=item["features"][None].to(device)
        text=embedding(index)[None].to(device)
        probability=decoder(modulator(features,text)).sigmoid()[0]
        # Fixed dense support subset: no hard sigma>tau gate during teacher training.
        positions=grid_centers(item["bounds"],probability.shape,device).reshape(-1,3)
        generator=torch.Generator(device=device).manual_seed(seed+step)
        select=torch.randperm(len(positions),device=device,generator=generator)[:max_render_points]
        poses=item["c2w"].to(device);images=item["images"].to(device)
        h,w=images.shape[-2:]
        k=intrinsics_matrix(item["intrinsics_normalized"].to(device),h,w)
        # Fixed RGB geometry cache used during teacher fitting; positions/poses frozen.
        colors=item.get("voxel_colors",torch.full((3,*probability.shape),.5)).to(device).permute(1,2,3,0).reshape(-1,3)[select]
        opacity=probability.reshape(-1)[select]
        scales=torch.full_like(positions[select],.1)
        rendered=[]
        for view in range(len(images)):
            rendered.append(rasterize_reference(positions[select],colors,opacity,scales,poses[view],k[view],h,w))
        loss=rendering_loss(torch.stack(rendered),images)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        history.append(float(loss.detach()))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.save({"modulator":modulator.state_dict(),"model_config":cfg.__dict__,"text_mode":text_mode,
                "decoder_checkpoint":str(decoder_checkpoint),"synthetic_demo":allow_demo},output/"teacher.pt")
    modulator.eval()
    with torch.inference_mode():
        for index,item in enumerate(items):
            occupancy=decoder(modulator(item["features"][None].to(device),embedding(index)[None].to(device))).sigmoid()[0].cpu()
            target={k:item[k] for k in ("images","c2w","intrinsics_normalized","bounds")}
            target["occupancy"]=occupancy
            target["provenance"]={**item["provenance"],"teacher_type":"supplementary_calibrated_decoder_reference",
                                  "text_mode":text_mode,"modulation":cfg.modulation,"uses_test_views":False}
            torch.save(target,output/f"target_{index:03d}.pt")
    save_json(output/"teacher_report.json",{"loss":history,"max_render_points":max_render_points,"text_mode":text_mode,
              "render_backend":"differentiable_CPU_reference_not_tiled_3DGS","pose_branch":"frozen cached VGGT poses; no language-conditioned pose decoder",
              "hard_threshold":"used only at inference; teacher fitting uses soft opacity", "synthetic_demo":allow_demo})

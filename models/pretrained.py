"""Local timm-style ViT-Tiny checkpoint import; no network or random fallback."""
import torch
from sgof.utils.common import load_tensor_file


def load_vit_tiny_encoder(student,path):
    state=load_tensor_file(path)
    state=state.get("state_dict",state.get("model",state))
    state={k.removeprefix("module."):v for k,v in state.items()}
    target=student.state_dict();mapping={"patch.weight":"patch_embed.proj.weight","patch.bias":"patch_embed.proj.bias",
             "norm.weight":"norm.weight","norm.bias":"norm.bias"}
    suffixes={"norm1.weight":"norm1.weight","norm1.bias":"norm1.bias","norm2.weight":"norm2.weight","norm2.bias":"norm2.bias",
              "self_attn.in_proj_weight":"attn.qkv.weight","self_attn.in_proj_bias":"attn.qkv.bias",
              "self_attn.out_proj.weight":"attn.proj.weight","self_attn.out_proj.bias":"attn.proj.bias",
              "linear1.weight":"mlp.fc1.weight","linear1.bias":"mlp.fc1.bias","linear2.weight":"mlp.fc2.weight","linear2.bias":"mlp.fc2.bias"}
    for i in range(student.cfg.transformer_layers):
        for a,b in suffixes.items():mapping[f"encoder.layers.{i}.{a}"]=f"blocks.{i}.{b}"
    replacements={}
    for a,b in mapping.items():
        if b not in state or target[a].shape!=state[b].shape:raise ValueError(f"Missing/mismatched pretrained key: {b}")
        replacements[a]=state[b]
    position=state.get("pos_embed")
    if position is None:raise ValueError("Missing pos_embed")
    if position.shape[1]==target["position"].shape[1]+1:position=position[:,1:]
    if position.shape!=target["position"].shape:raise ValueError("Position shape mismatch; use 224/p16 for standard ViT-Tiny")
    replacements["position"]=position
    target.update(replacements);student.load_state_dict(target,strict=True)
    return {"imported_parameters":len(replacements),"removed":"classifier and CLS token","normalization":"Caller must use pretrained input mean/std if required; base paper does not specify these."}

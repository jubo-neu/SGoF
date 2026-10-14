from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 192
    transformer_layers: int = 12
    attention_heads: int = 3
    seed_channels: int = 64
    decoder_channels: tuple = (32, 16, 8)
    grid_size: int = 32
    teacher_channels: int = 256
    text_dim: int = 512
    modulation_hidden: int = 256
    pose_hidden: int = 256
    pose_layers: int = 2
    modulation: str = "zwise"
    residual: bool = True
    batch_norm: bool = True
    input_normalization: str = "imagenet"

    def validate(self):
        if self.image_size % self.patch_size or self.embed_dim % self.attention_heads:
            raise ValueError("Image/patch and embedding/head divisibility required")
        if self.grid_size not in {16,32,64} or len(self.decoder_channels) != 3:
            raise ValueError("Supported grid sizes: 16/32/64; three decoder stages required")
        if self.modulation not in {"zwise", "global", "none"} or self.pose_layers not in {2,3}:
            raise ValueError("Invalid modulation or pose head depth")
        if self.input_normalization not in {"imagenet","none"}:raise ValueError("Invalid input normalization")
        if min(self.image_size,self.patch_size,self.embed_dim,self.transformer_layers,self.attention_heads,
               self.seed_channels,*self.decoder_channels,self.teacher_channels,self.text_dim,self.modulation_hidden,self.pose_hidden)<=0:
            raise ValueError("Model dimensions must be positive")


@dataclass
class TrainConfig:
    steps: int = 50000
    batch_size: int = 8
    learning_rate: float = 1e-4
    lambda_occ: float = 1.0
    lambda_pose: float = 0.1
    lambda_intrinsics: float = 0.0
    lambda_ssim: float = 0.2
    seed: int = 0
    device: str = "cpu"
    checkpoint_every: int = 1000

    def validate(self):
        if self.steps<1 or self.batch_size<1 or self.learning_rate<=0 or self.checkpoint_every<1:
            raise ValueError("Training budgets must be positive")
        if min(self.lambda_occ,self.lambda_pose,self.lambda_intrinsics)<0 or not 0<=self.lambda_ssim<=1:
            raise ValueError("Invalid loss weights")


def load_config(path):
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    model=ModelConfig(**raw.get("model",{}))
    train=TrainConfig(**raw.get("train",{}))
    model.validate(); train.validate()
    return model,train


def config_dict(model,train=None):
    return {"model":asdict(model),**({"train":asdict(train)} if train is not None else {})}

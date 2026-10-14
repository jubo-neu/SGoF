from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from sgof.models import Student
from sgof.data.datasets import DistillationDataset
from sgof.training.losses import distillation_loss
from sgof.utils.common import seed_everything, save_json, load_tensor_file
from sgof.config import config_dict


def train_student(manifest,output,model_cfg,train_cfg,allow_demo=False,resume=None,initial_weights=None,encoder_weights=None):
    seed_everything(train_cfg.seed)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dataset=DistillationDataset(manifest,"train",model_cfg,allow_demo)
    loader=DataLoader(dataset,batch_size=train_cfg.batch_size,shuffle=True,num_workers=0)
    model=Student(model_cfg).to(train_cfg.device)
    if encoder_weights:
        from sgof.models.pretrained import load_vit_tiny_encoder
        load_vit_tiny_encoder(model,encoder_weights)
    optimizer=torch.optim.Adam(model.parameters(),lr=train_cfg.learning_rate)
    start_step=0
    if initial_weights:
        state=load_tensor_file(initial_weights)
        model.load_state_dict(state.get("model",state),strict=True)
    if resume:
        state=load_tensor_file(resume)
        if state["config"]["model"]!=config_dict(model_cfg)["model"]:
            raise ValueError("Resume model configuration differs")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        start_step=state["step"]
    started=time.perf_counter();history=[];iterator=iter(loader)
    model.train()
    for step in range(start_step,train_cfg.steps):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(train_cfg.device) for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction=model(batch["images"])
        loss,parts=distillation_loss(prediction,batch,train_cfg)
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite distillation loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        history.append({"step":step+1,"total":float(loss.detach()),**parts})
        if (step+1)%train_cfg.checkpoint_every==0 or step+1==train_cfg.steps:
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step+1,
                        "config":config_dict(model_cfg,train_cfg),"synthetic_demo":allow_demo,
                        "pretrained_initialization":bool(initial_weights or encoder_weights)},output/"student.pt")
            save_json(output/"training_history.json",history)
    save_json(output/"training_report.json",{"steps":train_cfg.steps,"elapsed_s":time.perf_counter()-started,
              "parameters":sum(p.numel() for p in model.parameters()),"synthetic_demo":allow_demo,
              "config":config_dict(model_cfg,train_cfg),"resume_note":"Data-order and RNG-state continuation are not bit-exact."})
    return model

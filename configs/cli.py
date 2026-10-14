import argparse
from pathlib import Path
import torch
from sgof.config import load_config, ModelConfig
from sgof.utils.common import load_tensor_file, save_json


def main():
    parser=argparse.ArgumentParser(description="SGoF reference modules and reproducible experiment harness")
    sub=parser.add_subparsers(dest="command",required=True)
    train=sub.add_parser("train-student");train.add_argument("--manifest",required=True);train.add_argument("--config",required=True)
    train.add_argument("--output",required=True);train.add_argument("--allow-demo",action="store_true");train.add_argument("--resume");train.add_argument("--initial-weights")
    train.add_argument("--encoder-weights")
    init=sub.add_parser("initialize");init.add_argument("--checkpoint",required=True);init.add_argument("--context",required=True)
    init.add_argument("--output",required=True);init.add_argument("--device",default="cpu");init.add_argument("--tau",type=float,default=.5)
    init.add_argument("--max-points",type=int);init.add_argument("--allow-demo",action="store_true")
    for command in ("calibrate-decoder","fit-teacher"):
        p=sub.add_parser(command);p.add_argument("--caches",nargs="+",required=True);p.add_argument("--config",required=True)
        p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=100);p.add_argument("--device",default="cpu")
        p.add_argument("--allow-demo",action="store_true")
        if command=="fit-teacher":
            p.add_argument("--decoder",required=True);p.add_argument("--max-render-points",type=int,default=512)
            p.add_argument("--text-mode",choices=["correct","zero","shuffled"],default="correct")
    evaluate=sub.add_parser("evaluate");evaluate.add_argument("--manifest",required=True);evaluate.add_argument("--output",required=True)
    evaluate.add_argument("--lpips",action="store_true");evaluate.add_argument("--device",default="cpu")
    failures=sub.add_parser("failures");failures.add_argument("--metrics",required=True);failures.add_argument("--manifest",required=True)
    failures.add_argument("--output",required=True);failures.add_argument("--top-k",type=int,default=8)
    plan=sub.add_parser("plan");plan.add_argument("--output",required=True)
    abl=sub.add_parser("ablation-configs");abl.add_argument("--base",required=True);abl.add_argument("--output",required=True)
    run=sub.add_parser("run-external");run.add_argument("--task",required=True);run.add_argument("--execute",action="store_true")
    tables=sub.add_parser("tables");tables.add_argument("--results");tables.add_argument("--output",required=True);tables.add_argument("--protocol",default="native")
    profile=sub.add_parser("profile");profile.add_argument("--config",required=True);profile.add_argument("--checkpoint")
    profile.add_argument("--output",required=True);profile.add_argument("--views",type=int,default=8);profile.add_argument("--warmup",type=int,default=10)
    profile.add_argument("--repeats",type=int,default=50);profile.add_argument("--device",default="cpu")
    args=parser.parse_args()
    if args.command=="train-student":
        from sgof.training.student import train_student
        cfg,training=load_config(args.config)
        train_student(args.manifest,args.output,cfg,training,args.allow_demo,args.resume,args.initial_weights,args.encoder_weights)
    elif args.command=="initialize":
        from sgof.inference.initialize import initialize_student
        initialize_student(args.checkpoint,args.context,args.output,args.device,args.tau,args.max_points,args.allow_demo)
    elif args.command in {"calibrate-decoder","fit-teacher"}:
        from sgof.training.teacher import calibrate_decoder, fit_teacher
        cfg,_=load_config(args.config)
        if args.steps<1:parser.error("steps must be positive")
        if args.command=="calibrate-decoder":calibrate_decoder(args.caches,args.output,cfg,args.steps,args.device,args.allow_demo)
        else:fit_teacher(args.caches,args.decoder,args.output,cfg,args.steps,args.device,args.allow_demo,args.max_render_points,args.text_mode)
    elif args.command=="evaluate":
        from sgof.evaluation.evaluate import evaluate_renders
        evaluate_renders(args.manifest,args.output,args.lpips,args.device)
    elif args.command=="failures":
        from sgof.evaluation.failure_analysis import mine_failures
        mine_failures(args.metrics,args.manifest,args.output,args.top_k)
    elif args.command=="plan":
        from sgof.experiments.runner import plan_experiments
        tasks=plan_experiments(args.output);print(f"{len(tasks)} explicitly planned main/comparison tasks; not executed.")
    elif args.command=="ablation-configs":
        from sgof.experiments.runner import write_ablation_configs
        write_ablation_configs(args.base,args.output)
    elif args.command=="run-external":
        from sgof.experiments.runner import run_external_task
        print(run_external_task(args.task,args.execute))
    elif args.command=="tables":
        from sgof.experiments.tables import write_comparison_tables, write_tables_from_results
        if args.results:write_tables_from_results(args.results,args.output,args.protocol)
        else:write_comparison_tables(args.output)
    elif args.command=="profile":
        from sgof.models import Student
        from sgof.experiments.complexity import profile_student
        if args.views<1 or args.warmup<0 or args.repeats<1:parser.error("invalid profile budget")
        cfg,_=load_config(args.config);model=Student(cfg).to(args.device)
        if args.checkpoint:model.load_state_dict(load_tensor_file(args.checkpoint)["model"],strict=True)
        images=torch.zeros(1,args.views,3,cfg.image_size,cfg.image_size,device=args.device)
        save_json(args.output,profile_student(model,images,args.warmup,args.repeats))
    print(f"Finished {args.command}.")

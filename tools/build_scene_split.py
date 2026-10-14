"""Create a checked RGB split from official train/test lists; no automatic scene renaming."""
import argparse
from pathlib import Path
from sgof.utils.common import save_json


def read_list(path):return [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",required=True);p.add_argument("--images",required=True)
    p.add_argument("--train-list",required=True);p.add_argument("--test-list",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();train,test=read_list(args.train_list),read_list(args.test_list)
    if not train or not test or set(train)&set(test):raise ValueError("Nonempty, disjoint train/test lists required")
    if len(train)!=len(set(train)) or len(test)!=len(set(test)):raise ValueError("Repeated image IDs")
    root=Path(args.images).resolve()
    for name in train+test:
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file():raise ValueError(f"Missing/unsafe image {name}")
    save_json(args.output,{"scene":args.scene,"images_root":str(root),"train_images":train,"test_images":test,
                          "split_source":{"train":str(Path(args.train_list).resolve()),"test":str(Path(args.test_list).resolve())},
                          "teacher_context_rule":"Use train_images only, including narratives and all feature extraction."})


if __name__=="__main__":main()

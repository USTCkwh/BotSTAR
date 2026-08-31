import os
import argparse
import random
import yaml
import numpy as np
import torch



def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Twibot-20")  # Twibot-20
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num_heads_encoder", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--feat_drop", type=float, default=0.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.1,
                        help="attention dropout")
    parser.add_argument("--finetune_drop", type=float, default=0.3,
                        help="dropout rate of finetune model")
    parser.add_argument("--lr", type=float, default=0.015,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--feat_mask_rate", type=float, default=0.25)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--leave_unchanged_rate", type=float, default=0.0)
    parser.add_argument("--max_epoch_f", type=int, default=300)
    parser.add_argument("--lr_f", type=float, default=0.01, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--scheduler", action="store_true", default=True)
    parser.add_argument("--cluster_num", type=int, default=20)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--advanced_loss_rate", type=float, default=0.0005)
    parser.add_argument("--instance_prototype_loss_rate", type=float, default=0.01)
    parser.add_argument("--basevector_num", type=int, default=10)
    parser.add_argument("--use_align_loss", action="store_true", default=True)
    parser.add_argument("--align_loss_rate", type=float, default=0.13)
    parser.add_argument("--adapt_align_loss_rate", type=float, default=1.0,
                        help="下游 Adaptation 边扰动对齐损失权重，与预训练 align_loss_rate 区分")
    parser.add_argument("--edge_add_ratio", type=float, default=0.5,
                        help="下游 Adaptation 边扰动时加入虚构边的比例")
    parser.add_argument("--rep_sim_analysis", action="store_true", default=False,
                        help="开启后统计不同种子预训练模型对全图表征的跨种子余弦相似度")
    parser.add_argument("--pairs", type=str, nargs="+", default=None,
                        help="复现用 seed:max_epoch_f 键值对列表，如 8608:200 1272:250，每个种子用各自的 max_epoch_f")
    args = parser.parse_args()
    return args

def load_best_configs(args, path):
    with open(path, "r", encoding='utf-8') as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args




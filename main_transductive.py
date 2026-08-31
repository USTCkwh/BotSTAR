import numpy as np
from torch_geometric.transforms import TwoHop
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils import (
    build_args,
    set_random_seed,
    load_best_configs,
)
from torch import optim
from node_classification_evaluation import node_classification_evaluation
from models import build_model
import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import os


def pretrain(model, graph, optimizer, max_epoch, device, scheduler):
    graph = graph.to(device)
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss = model(graph.x, graph.edge_index, graph.edge_type, epoch)
        assert not torch.isnan(loss).any()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.10f}")
    return model


def pretrain_batched(model, graph, optimizer, max_epoch, device, scheduler, batch_size, num_neighbors):
    """
    用 NeighborLoader 对全图做子图采样分批预训练，避免全图 forward OOM。
    graph 留在 CPU 用于采样，每个 batch 子图单独 to(device) 前向。
    forward 内 compute_prototype=False 跳过未计入总 loss 的 instance_prototype_loss，
    避免每个 batch 都跑全图+kmeans 的无谓计算。
    """
    input_nodes = torch.arange(graph.x.size(0))
    loader = NeighborLoader(
        graph,
        num_neighbors=list(num_neighbors),
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=True,
    )
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            loss = model(batch.x, batch.edge_index, batch.edge_type, epoch, compute_prototype=False)
            assert not torch.isnan(loss).any()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {total_loss / max(n_batches, 1):.10f}")
    return model


def add_two_hop_edges(graph):
    th = TwoHop()
    original_edge_index = graph.edge_index
    original_edge_type = graph.edge_type
    edge_index_list = []
    edge_type_list = []
    for i in range(2):
        edge_index_i = original_edge_index[:, original_edge_type == i]
        graph.edge_index = edge_index_i
        graph = th(graph)
        graph.edge_type = torch.ones(graph.edge_index.shape[1], dtype=torch.long) * i
        edge_index_list.append(graph.edge_index)
        edge_type_list.append(graph.edge_type)
    graph.edge_index = torch.cat(edge_index_list, dim=1)
    graph.edge_type = torch.cat(edge_type_list, dim=0)
    return graph


def mask_train_nodes(graph, keep_rate=1.0, seed=0):
    train_mask = graph.train_mask
    print("mask train dataset, keep_rate: ", keep_rate)
    train_idx = torch.where(train_mask == 1)[0]
    train_num_nodes = train_idx.shape[0]
    perm = torch.randperm(train_num_nodes, generator=torch.Generator().manual_seed(seed))
    keep_nodes = int(keep_rate * train_num_nodes)
    train_idx = train_idx[perm][:keep_nodes]
    mask = torch.zeros_like(train_mask, dtype=torch.bool)
    mask[train_idx] = 1
    graph.train_mask = mask.bool()
    return graph


def load_dataset(dataset_name):
    name_to_path = {
        "Twibot-20": './dataset/processed/Twibot-20-n-hops-subgraph.pt',
        "MGTAB-FF": './dataset/processed/MGTAB-followers-friends.pt',
    }
    graph = torch.load(name_to_path[dataset_name])
    graph.name = dataset_name
    graph.num_classes = 2
    num_features = graph.num_features
    num_classes = graph.num_classes
    if dataset_name == "Twibot-20":
        graph = add_two_hop_edges(graph)
        graph = add_two_hop_edges(graph)
    return graph, (num_features, num_classes)


def visualize_embeddings(final_model, estp_model, graph, dataset_name, device):
    """
    保存模型编码数据到文件，供后续可视化脚本使用
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('visualizations', exist_ok=True)
    
    graph = graph.to(device)
    labels = graph.y
    
    # 获取 final 模型的编码
    final_model.eval()
    with torch.no_grad():
        final_model_module = final_model.module if hasattr(final_model, 'module') else final_model
        a1_final = final_model_module.gnn_encoder_pretrained(graph.x, graph.edge_index, graph.edge_type)
        a2_final = final_model_module.gnn_encoder(graph.x, graph.edge_index, graph.edge_type)
        final_embeddings = final_model_module.attention(torch.cat((a1_final.unsqueeze(1), a2_final.unsqueeze(1)), dim=1))
        final_embeddings = final_embeddings.squeeze(1)
    
    # 获取 estp 模型的编码
    estp_model.eval()
    with torch.no_grad():
        estp_model_module = estp_model.module if hasattr(estp_model, 'module') else estp_model
        a1_estp = estp_model_module.gnn_encoder_pretrained(graph.x, graph.edge_index, graph.edge_type)
        a2_estp = estp_model_module.gnn_encoder(graph.x, graph.edge_index, graph.edge_type)
        estp_embeddings = estp_model_module.attention(torch.cat((a1_estp.unsqueeze(1), a2_estp.unsqueeze(1)), dim=1))
        estp_embeddings = estp_embeddings.squeeze(1)
    
    # 只取测试集节点
    final_embeddings_test = final_embeddings[graph.test_mask].cpu().numpy()
    estp_embeddings_test = estp_embeddings[graph.test_mask].cpu().numpy()
    labels_test = labels[graph.test_mask].cpu().numpy()
    
    # 保存编码数据到 npz 文件
    # a1_final/a2_final: final epoch 两个编码器各自的原始输出（全图节点, N x num_hidden）
    os.makedirs(f'visualizations_data', exist_ok=True)
    data_path = f'visualizations_data/ours_{dataset_name}_{timestamp}.npz'
    np.savez(data_path,
             final_embeddings=final_embeddings_test,
             estp_embeddings=estp_embeddings_test,
             labels=labels_test,
             dataset_name=dataset_name,
             a1_final=a1_final.cpu().numpy(),
             a2_final=a2_final.cpu().numpy(),
             test_mask=graph.test_mask.cpu().numpy(),
             labels_full=labels.cpu().numpy())
    print(f"Saved embeddings data: {data_path}")
    
    return data_path


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    # --pairs 模式：每个种子用专属 max_epoch_f（seed:max_epoch_f 键值对）
    seed_epoch_map = None
    if getattr(args, 'pairs', None) is not None:
        seed_epoch_map = {}
        for pair in args.pairs:
            s, e = pair.split(':')
            seed_epoch_map[int(s)] = int(e)
        seeds = list(seed_epoch_map.keys())
        print(f"复现模式: 每个种子用专属 max_epoch_f: {seed_epoch_map}")
    num_hidden = args.num_hidden
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    use_scheduler = args.scheduler
    finetune_drop = args.finetune_drop
    basevector_num = args.basevector_num
    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features
    final_metric_list = []
    estp_metric_list = []
    # 跨种子表征相似度统计：流式累加 L2 归一化后的 gnn 表征，
    # 避免存储 S×N×D 的全部表征（种子数可达数百）
    rep_sim_analysis = getattr(args, 'rep_sim_analysis', False)
    rep_sum = None
    # 结果文件：每跑完一个种子实时追加写入（run_results/日期/运行开始时间.txt）
    run_start = datetime.now()
    result_dir = os.path.join('run_results', run_start.strftime('%Y%m%d'))
    os.makedirs(result_dir, exist_ok=True)
    for i, seed in enumerate(seeds):
        if seed_epoch_map is not None:
            max_epoch_f = seed_epoch_map[seed]
        print(f"####### Run {i} #######")
        set_random_seed(seed)
        # 标签效率分析：train_rate < 1 时在克隆图上按比例掩码训练节点（不污染原图）
        train_rate = getattr(args, 'train_rate', 1.0)
        if train_rate < 1.0:
            run_graph = mask_train_nodes(graph.clone(), train_rate, seed)
        else:
            run_graph = graph
        print(args)
        model = build_model(args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        else:
            scheduler = None

        model = pretrain(model, run_graph, optimizer, max_epoch, device, scheduler)
        if rep_sim_analysis:
            # 预训练模型对评估图(eval_graph)的 gnn 表征跨种子一致性（eval 禁用 dropout）
            model.eval()
            x = eval_graph.x.to(device)
            ei = eval_graph.edge_index.to(device)
            et = eval_graph.edge_type.to(device)
            gnn_rep, _ = model.embed(x, ei, et)
            z = F.normalize(gnn_rep, p=2, dim=1)
            rep_sum = z if rep_sum is None else rep_sum + z
            del x, ei, et
            if device != "cpu":
                torch.cuda.empty_cache()
        final_metric, estp_metric, final_model, estp_model = node_classification_evaluation(model, run_graph, num_hidden, lr_f, weight_decay_f,
                                                                   max_epoch_f, finetune_drop, device, dataset_name, basevector_num, args.bvlen_factor,
                                                                   adapt_align_loss_rate=args.adapt_align_loss_rate,
                                                                   edge_add_ratio=args.edge_add_ratio)
        final_metric_list.append(final_metric)
        estp_metric_list.append(estp_metric)

        # 实时追加该种子完整结果（不含中间训练日志）
        with open(result_path, 'a') as f:
            f.write(f"=== Run {i} seed={seed} max_epoch_f={max_epoch_f} ===\n")
            f.write(f"final:      acc={final_metric['accuracy']:.4f} prec={final_metric['precision']:.4f} "
                    f"rec={final_metric['recall']:.4f} f1={final_metric['f1']:.4f}\n")
            f.write(f"early-stop: acc={estp_metric['accuracy']:.4f} prec={estp_metric['precision']:.4f} "
                    f"rec={estp_metric['recall']:.4f} f1={estp_metric['f1']:.4f}\n\n")

        # t-SNE visualization
        if i == 0 and getattr(args, 'visualize', True):  # 只在第一次种子运行时生成可视化
            visualize_embeddings(final_model, estp_model, run_graph, dataset_name, device)
        

    if rep_sim_analysis:
        S = len(seeds)
        if S < 2:
            print("[rep_sim] 种子数 < 2，无法计算跨种子相似度")
        else:
            # 每个节点跨所有种子对的平均余弦相似度（精确闭式解）：
            # Σ_{i≠j} cos(ẑ_i, ẑ_j) = ||Σ_i ẑ_i||² − S （ẑ 已 L2 归一化），
            # 有序对共 S(S−1) 个，均值 = (||Σẑ||² − S) / (S(S−1))
            per_node = (rep_sum.pow(2).sum(dim=1) - S) / (S * (S - 1))
            print("====== 跨种子表征余弦相似度 (rep_sim_analysis) ======")
            print(f"seeds: {S}, nodes: {per_node.numel()}, repr: gnn_encoder (pretrained)")
            print(f"node-wise pairwise-cos mean (全图节点平均): {per_node.mean().item():.6f}")
            print(f"node-wise pairwise-cos std:  {per_node.std().item():.6f}")
            print(f"node-wise pairwise-cos min:  {per_node.min().item():.6f}")
            print(f"node-wise pairwise-cos max:  {per_node.max().item():.6f}")


    with open(result_path, 'a') as f_summary:
        f_summary.write("# summary (mean±std)\n")
        for i in ['accuracy', 'precision', 'recall', 'f1']:
            a_list = []
            b_list = []
            for j in range(len(final_metric_list)):
                a_list.append(final_metric_list[j][i])
                b_list.append(estp_metric_list[j][i])
            line_a = f"final_{i}: {np.mean(a_list):.4f}±{np.std(a_list):.4f}"
            line_b = f"early-stopping_{i}: {np.mean(b_list):.4f}±{np.std(b_list):.4f}"
            print(line_a)
            print(line_b)
            f_summary.write(line_a + "\n")
            f_summary.write(line_b + "\n")
    return final_metric_list, estp_metric_list



if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
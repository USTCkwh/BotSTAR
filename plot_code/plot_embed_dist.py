"""
对比 final epoch 两个编码器输出 a1/a2 前三个维度的均值曲线。

数据来源: main_transductive.py 的 visualize_embeddings 保存的 npz
(键 a1_final / a2_final, 全图节点, final epoch 模型, eval 模式)。

用法:
    python plot_embed_dist.py [数据集名|npz路径]
    # 数据集名: 如 Twibot-20 / MGTAB-FF, 取该数据集最新的含 a1_final 的 npz
    # npz路径: 直接指定文件
    # 不指定则自动取最新的含 a1_final 的 ours_*.npz
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wilcoxon

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'visualizations_data')


def load_data(target=None):
    if target is not None and os.path.isfile(target):
        path = target
    else:
        pattern = 'ours_*.npz' if target is None else f'ours_{target}_*.npz'
        cands = []
        for p in glob.glob(os.path.join(DATA_DIR, pattern)):
            try:
                with np.load(p) as d:
                    if 'a1_final' in d.files:
                        cands.append(p)
            except Exception:
                continue
        if not cands:
            sys.exit(f'visualizations_data/ 下没有包含 a1_final 的 {pattern}, 请先跑 main_transductive.py 生成数据')
        path = max(cands, key=os.path.getmtime)
    d = np.load(path)
    return path, d['a1_final'], d['a2_final'], str(d['dataset_name'])


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "MGTAB-FF"
    path, a1, a2, dataset = load_data(target)
    x1, x2 = a1[:, :1].mean(axis=1), a2[:, :1].mean(axis=1)

    std_frz = np.std(a1, axis=0)
    std_tun = np.std(a2, axis=0)
    stat, p_value = wilcoxon(std_tun, std_frz, alternative='greater')

    print(f"[STATS] dataset={dataset} nodes={a1.shape[0]} dim={a1.shape[1]}")
    print(f"[STATS] frz_std_mean={std_frz.mean():.6f} frz_std_median={np.median(std_frz):.6f}")
    print(f"[STATS] tun_std_mean={std_tun.mean():.6f} tun_std_median={np.median(std_tun):.6f}")
    print(f"[STATS] wilcoxon_stat={stat:.2f} p_value={p_value:.6e}")

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.linspace(-1, 2, 300)
    BW = 4.0
    kde1, kde2 = gaussian_kde(x1), gaussian_kde(x2)
    kde1.set_bandwidth(kde1.factor * BW)
    kde2.set_bandwidth(kde2.factor * BW)
    ax.fill_between(xs, kde1(xs), color='tab:blue', alpha=0.15, lw=0)
    ax.plot(xs, kde1(xs), color='tab:blue', lw=2,
            label='a1 (frozen pretrained encoder)')
    ax.fill_between(xs, kde2(xs), color='tab:red', alpha=0.15, lw=0)
    ax.plot(xs, kde2(xs), color='tab:red', lw=2,
            label='a2 (finetuned encoder)')

    ax.set_xlim(-1, 2)
    ax.set_ylabel('density')
    # 图例放在曲线峰值上方, 永不覆盖图线
    ymax = max(kde1(xs).max(), kde2(xs).max())
    ax.set_ylim(0, ymax * 1.2)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)

    out = os.path.join(BASE, dataset + '_embed_dist')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', bbox_inches='tight')
    print(f'data: {path}')
    print(f'figure saved: {out}.pdf / {out}.png')


if __name__ == '__main__':
    main()
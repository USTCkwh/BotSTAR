import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import glob
import os


def visualize_embeddings(data_path):
    """
    读取保存的编码数据并生成 t-SNE 可视化图
    
    Args:
        data_path: .npz 文件路径
    """
    # 读取数据
    data = np.load(data_path)
    final_embeddings = data['final_embeddings']
    estp_embeddings = data['estp_embeddings']
    labels = data['labels']
    dataset_name = str(data['dataset_name'])
    
    print(f"Dataset: {dataset_name}")
    print(f"Final embeddings shape: {final_embeddings.shape}")
    print(f"ESTP embeddings shape: {estp_embeddings.shape}")
    
    # t-SNE 降维
    print("Running t-SNE for final model...")
    tsne_final = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    final_2d = tsne_final.fit_transform(final_embeddings)
    
    print("Running t-SNE for estp model...")
    tsne_estp = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=1000)
    estp_2d = tsne_estp.fit_transform(estp_embeddings)
    
    # 从文件名读取模型名和数据集名
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    parts = base_name.split('_')
    model_name = parts[0]
    dataset_name_raw = '_'.join(parts[1:-2]) if len(parts) > 2 else parts[1] if len(parts) > 1 else 'unknown'
    
    # 简化数据集名：MGTAB-FF -> MGTAB, Twibot-20 保持不变
    if dataset_name_raw.startswith('MGTAB'):
        dataset_name = 'MGTAB'
    else:
        dataset_name = dataset_name_raw
    
    output_dir = 'visualizations'  # PDF 保存到 visualizations 目录
    
    # 绘制 final 模型可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'red']
    
    for i, color in enumerate(colors):
        idx = labels == i
        ax.scatter(final_2d[idx, 0], final_2d[idx, 1], c=color, alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    
    ax.axis('off')
    plt.tight_layout()
    
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    final_pdf_path = os.path.join(final_dir, f'{dataset_name}_{model_name}.pdf')
    plt.savefig(final_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved final model visualization: {final_pdf_path}")
    
    # 绘制 estp 模型可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, color in enumerate(colors):
        idx = labels == i
        ax.scatter(estp_2d[idx, 0], estp_2d[idx, 1], c=color, alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    
    ax.axis('off')
    plt.tight_layout()
    
    estp_dir = os.path.join(output_dir, 'estp')
    os.makedirs(estp_dir, exist_ok=True)
    estp_pdf_path = os.path.join(estp_dir, f'{dataset_name}_{model_name}.pdf')
    plt.savefig(estp_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved estp model visualization: {estp_pdf_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize embeddings from saved .npz files')
    parser.add_argument('--file', type=str, help='Specific .npz file to visualize')
    parser.add_argument('--latest', action='store_true', help='Visualize the latest .npz file')
    parser.add_argument('--all', action='store_true', help='Visualize all .npz files in visualizations_data/')
    parser.add_argument('--dataset', type=str, help='Visualize files for a specific dataset')
    
    args = parser.parse_args()
    
    if args.file:
        # 可视化指定文件
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        visualize_embeddings(args.file)
    
    elif args.latest:
        # 可视化最新的文件
        npz_files = glob.glob('visualizations_data/*.npz')
        if not npz_files:
            print("No .npz files found in visualizations_data/")
            return
        latest_file = max(npz_files, key=os.path.getctime)
        print(f"Latest file: {latest_file}")
        visualize_embeddings(latest_file)
    
    elif args.dataset:
        # 可视化指定数据集的所有文件
        npz_files = glob.glob(f'visualizations_data/{args.dataset}_*.npz')
        if not npz_files:
            print(f"No .npz files found for dataset: {args.dataset}")
            return
        for npz_file in npz_files:
            print(f"\nProcessing: {npz_file}")
            visualize_embeddings(npz_file)
    
    elif args.all:
        # 可视化所有文件
        npz_files = glob.glob('visualizations_data/*.npz')
        if not npz_files:
            print("No .npz files found in visualizations_data/")
            return
        for npz_file in npz_files:
            print(f"\nProcessing: {npz_file}")
            visualize_embeddings(npz_file)
    
    else:
        # 默认可视化最新的文件
        npz_files = glob.glob('visualizations_data/*.npz')
        if not npz_files:
            print("No .npz files found in visualizations_data/")
            print("Run main_transductive.py first to generate embedding data.")
            return
        latest_file = max(npz_files, key=os.path.getctime)
        print(f"Latest file: {latest_file}")
        visualize_embeddings(latest_file)


if __name__ == '__main__':
    main()

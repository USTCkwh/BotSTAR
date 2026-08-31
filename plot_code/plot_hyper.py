import matplotlib.pyplot as plt
import numpy as np

# ========== 原始表格数据（字符串形式） ==========
align_loss_rate_data = """
0.0	0.8563		0.8793		0.8563		0.8786
0.02	0.8580		0.8807		0.8563		0.8789
0.04	0.8563		0.8794		0.8563		0.8794
0.06	0.8588		0.8813		0.8588		0.8813
0.08	0.8563		0.8789		0.8563		0.8789
0.1	0.8571		0.8797		0.8563		0.8789
0.12	0.8571		0.8797		0.8563		0.8791
0.14	0.8555		0.8781		0.8571		0.8789
0.16	0.8571		0.8797		0.8571		0.8797
0.18	0.8563		0.8787		0.8563		0.8782
0.2	0.8555		0.8781		0.8571		0.8789
0.22	0.8588		0.8806		0.8588		0.8806
0.24	0.8732		0.8881		0.8724		0.8874
0.26	0.8732		0.8884		0.8732		0.8882
0.28	0.8715		0.8869		0.8715		0.8871
0.3	0.8724		0.8877		0.8715		0.8871
0.32	0.8740		0.8891		0.8724		0.8876
0.34	0.8724		0.8874		0.8732		0.8882
0.36	0.8732		0.8887		0.8749		0.8895
0.38	0.8749		0.8900		0.8740		0.8889
0.4	0.8715		0.8867		0.8757		0.8910
0.42	0.8740		0.8894		0.8715		0.8866
0.44	0.8715		0.8869		0.8749		0.8904
0.46	0.8740		0.8894		0.8715		0.8867
0.48	0.8749		0.8904		0.8749		0.8904
0.5	0.8766		0.8917		0.8732		0.8884
0.52	0.8740		0.8894		0.8740		0.8892
0.54	0.8749		0.8900		0.8749		0.8900
0.56	0.8732		0.8886		0.8740		0.8894
0.58	0.8732		0.8884		0.8732		0.8884
0.6	0.8740		0.8895		0.8740		0.8895
0.62	0.8740		0.8895		0.8757		0.8905
0.64	0.8732		0.8889		0.8732		0.8884
0.66	0.8724		0.8882		0.8740		0.8894
0.68	0.8707		0.8869		0.8732		0.8886
0.7	0.8724		0.8881		0.8740		0.8892
0.72	0.8715		0.8874		0.8740		0.8889
0.74	0.8690		0.8856		0.8715		0.8871
0.76	0.8724		0.8881		0.8740		0.8892
0.78	0.8681		0.8851		0.8707		0.8864
0.8	0.8673		0.8843		0.8707		0.8866
0.82	0.8715		0.8872		0.8715		0.8872
0.84	0.8681		0.8850		0.8715		0.8872
0.86	0.8681		0.8853		0.8715		0.8872
0.88	0.8673		0.8843		0.8673		0.8841
0.9	0.8631		0.8814		0.8664		0.8835
0.92	0.8648		0.8827		0.8673		0.8838
0.94	0.8673		0.8843		0.8690		0.8853
0.96	0.8664		0.8835		0.8673		0.8840
0.98	0.8648		0.8829		0.8605		0.8800
"""

edge_add_ratio_data = """
0.0	0.8749		0.8909		0.8664		0.8835
0.05	0.8783		0.8935		0.8783		0.8935
0.1	0.8757		0.8915		0.8791		0.8942
0.15	0.8757		0.8915		0.8774		0.8927
0.2	0.8757		0.8915		0.8757		0.8910
0.25	0.8749		0.8907		0.8774		0.8927
0.3	0.8757		0.8915		0.8707		0.8866
0.35	0.8766		0.8920		0.8774		0.8927
0.4	0.8808		0.8952		0.8757		0.8902
0.45	0.8800		0.8945		0.8757		0.8902
0.5	0.8808		0.8952		0.8757		0.8902
0.55	0.8791		0.8940		0.8757		0.8902
0.6	0.8800		0.8947		0.8757		0.8902
0.65	0.8791		0.8940		0.8724		0.8879
0.7	0.8791		0.8940		0.8774		0.8927
0.75	0.8800		0.8947		0.8757		0.8902
0.8	0.8791		0.8940		0.8774		0.8927
0.85	0.8800		0.8947		0.8774		0.8927
0.9	0.8808		0.8953		0.8774		0.8927
0.95	0.8808		0.8953		0.8757		0.8902
"""

adapt_align_loss_rate_data = """
adapt_align_loss_rate	Final Acc	Final F1	ESTP Acc	ESTP F1
0.0	0.8774		0.8925		0.8791		0.8937
0.02	0.8783		0.8933		0.8783		0.8933
0.04	0.8783		0.8932		0.8698		0.8859
0.06	0.8791		0.8940		0.8783		0.8930
0.08	0.8800		0.8947		0.8791		0.8942
0.1	0.8800		0.8947		0.8749		0.8894
0.12	0.8791		0.8938		0.8757		0.8902
0.14	0.8800		0.8947		0.8715		0.8872
0.16	0.8800		0.8945		0.8757		0.8902
0.18	0.8791		0.8940		0.8724		0.8879
0.2	0.8808		0.8952		0.8757		0.8902
0.22	0.8791		0.8938		0.8774		0.8927
0.24	0.8800		0.8945		0.8757		0.8902
0.26	0.8800		0.8945		0.8800		0.8945
0.28	0.8774		0.8925		0.8774		0.8927
0.3	0.8774		0.8925		0.8757		0.8902
0.32	0.8783		0.8933		0.8766		0.8919
0.34	0.8774		0.8925		0.8783		0.8933
0.36	0.8783		0.8933		0.8757		0.8912
0.38	0.8783		0.8933		0.8774		0.8925
0.4	0.8783		0.8933		0.8783		0.8933
0.42	0.8766		0.8920		0.8783		0.8933
0.44	0.8766		0.8920		0.8783		0.8933
0.46	0.8766		0.8920		0.8783		0.8933
0.48	0.8766		0.8920		0.8774		0.8925
0.5	0.8757		0.8912		0.8766		0.8917
0.52	0.8766		0.8920		0.8774		0.8925
0.54	0.8757		0.8912		0.8774		0.8925
0.56	0.8766		0.8920		0.8766		0.8917
0.58	0.8766		0.8920		0.8774		0.8925
"""

# ========== 超参数配置字典（键与数据变量对应） ==========
param_configs = {
    'align_loss_rate': {
        'data': align_loss_rate_data,
        'x_label': 'align_loss_rate',
        'title': 'Performance vs. align_loss_rate',
        'acc_legend': 'Accuracy',
        'f1_legend': 'F1'
    },
    'edge_add_ratio': {
        'data': edge_add_ratio_data,
        'x_label': 'edge_add_ratio',
        'title': 'Performance vs. edge_add_ratio',
        'acc_legend': 'Accuracy',
        'f1_legend': 'F1'
    },
    'adapt_align_loss_rate': {
        'data': adapt_align_loss_rate_data,
        'x_label': 'adapt_align_loss_rate',
        'title': 'Performance vs. adapt_align_loss_rate',
        'acc_legend': 'Accuracy',
        'f1_legend': 'F1'
    }
}

# ========== 数据解析函数 ==========
def parse_data(raw_data):
    """从原始字符串解析出 x, final_acc, final_f1, estp_acc, estp_f1"""
    lines = raw_data.strip().split('\n')
    data = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            row = [float(p) for p in parts]
            data.append(row)
        except ValueError:
            continue
    data = np.array(data)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError("数据格式不正确，应至少包含5列")
    x = data[:, 0]
    final_acc = data[:, 1]
    final_f1 = data[:, 2]
    estp_acc = data[:, 3]
    estp_f1 = data[:, 4]
    return x, final_acc, final_f1, estp_acc, estp_f1

def plot_performance(acc_data, f1_data, x_data, output_filename, text_cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 左轴：准确率（条形图）
    bars = ax.bar(x_data, acc_data, width=0.015, color='skyblue',
                  label=text_cfg['acc_legend'], alpha=0.7)
    # ax.set_xlabel(text_cfg['x_label'], fontsize=14)
    # ax.set_ylabel('Accuracy', fontsize=14, color='skyblue')
    ax.tick_params(axis='y', labelcolor='skyblue')
    
    # 自动设置准确率的Y轴范围（留出5%的边距）
    acc_min, acc_max = acc_data.min(), acc_data.max()
    margin = (acc_max - acc_min) * 0.05  # 5%边距
    ax.set_ylim(acc_min - margin, acc_max + margin)
    
    # 右轴：F1（折线图）
    ax2 = ax.twinx()
    line = ax2.plot(x_data, f1_data, color='red', marker='o',
                    markersize=4, linewidth=2, label=text_cfg['f1_legend'])
    # ax2.set_ylabel('F1 Score', fontsize=14, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 自动设置F1的Y轴范围（留出5%的边距）
    f1_min, f1_max = f1_data.min(), f1_data.max()
    margin2 = (f1_max - f1_min) * 0.05
    ax2.set_ylim(f1_min - margin2, f1_max + margin2)
    
    # 合并图例
    handles = [bars] + line
    labels = [text_cfg['acc_legend'], text_cfg['f1_legend']]
    ax.legend(handles, labels, loc='best')
    
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.set_title(text_cfg['title'], fontsize=16)  # 按需取消注释
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)

# ========== 主程序 ==========
if __name__ == '__main__':
    selected_param = 'adapt_align_loss_rate'  # 可修改为 'align_loss_rate' 或 'edge_add_ratio'
    
    if selected_param not in param_configs:
        raise ValueError(f"未知超参数: {selected_param}，可用选项: {list(param_configs.keys())}")
    
    cfg = param_configs[selected_param]
    raw_data = cfg['data']
    
    x, final_acc, final_f1, estp_acc, estp_f1 = parse_data(raw_data)
    
    base_name = selected_param.replace('_', '-')
    final_output = f'final_performance_{base_name}.pdf'
    estp_output = f'estp_performance_{base_name}.pdf'
    
    plot_performance(final_acc, final_f1, x, final_output, cfg)
    # plot_performance(estp_acc, estp_f1, x, estp_output, cfg)
    
    print(f"已生成图表: {final_output} 和 {estp_output}")
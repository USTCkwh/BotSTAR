import matplotlib.pyplot as plt
import numpy as np

data = {
    'Ours': [0.8478, 0.8557, 0.8650, 0.8656, 0.8662, 0.8698, 0.8732, 0.8766, 0.8766, 0.8808],
    'BotHP': [0.8454, 0.8546, 0.8617, 0.8631, 0.8645, 0.8673, 0.8670, 0.8670, 0.8695, 0.8715],
    'GraphMAE': [0.8467, 0.8557, 0.8608, 0.8631, 0.8650, 0.8659, 0.8676, 0.8712, 0.8690, 0.8664],
    'SEBot': [0.8360, 0.8538, 0.8538, 0.8588, 0.8597, 0.8555, 0.8588, 0.8698, 0.8614, 0.8629],
    'RGT': [0.8402, 0.8515, 0.8549, 0.8614, 0.8611, 0.8625, 0.8647, 0.8628, 0.8673, 0.8662],
    'BotDGT': [0.8356, 0.8408, 0.8487, 0.8588, 0.8574, 0.8560, 0.8628, 0.8631, 0.8591, 0.8681],
}

x = np.arange(10, 101, 10)

colors = {
    'Ours': '#d62728',
    'BotHP': '#1f77b4',
    'GraphMAE': '#9467bd',
    'SEBot': '#17becf',
    'RGT': '#ff7f0e',
    'BotDGT': '#2ca02c',
}

# markers = {
#     'BotHP': 'o',
#     'RGT': 's',
#     'BotDGT': '^',
#     'GraphMAE': 'D',
#     'SEBot': '8',
#     'Ours': '*',
# }

plt.figure(figsize=(9, 7))

for name, values in data.items():
    markersize = 8
    # if name == 'Ours':
    #     markersize = 12
    plt.plot(x, values, marker='o', markersize=markersize, linewidth=2, 
             label=name, color=colors[name])

plt.xlabel('Label Ratio (%)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
# plt.title('Label Efficiency Analysis', fontsize=14, fontweight='bold')
plt.xticks(x)
plt.ylim(0.83, 0.89)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('label_efficiency.pdf', dpi=300, bbox_inches='tight')

print("图表已保存到 label_efficiency.pdf")

for name, values in data.items():
    print(f"{name}: max={max(values):.4f}, final={values[-1]:.4f}")

plt.show()

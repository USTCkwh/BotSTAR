import matplotlib.pyplot as plt
import numpy as np

data = {
    1: 0.022582102566957474,
    2: 0.026822827756404877,
    3: 0.031922776252031326,
    4: 0.03290880471467972,
    5: 0.030248602852225304,
    6: 0.037008557468652725,
    7: 0.03776627779006958,
    8: 0.03872929513454437,
    9: 0.038489315658807755,
    10: 0.044966261833906174,
    11: 0.05920901149511337,
    12: 0.07073023170232773,
    13: 0.07831744849681854,
    14: 0.08181905746459961,
    15: 0.08461722731590271,
    16: 0.08336789906024933,
    17: 0.08294893801212311,
    18: 0.0802612230181694,
    19: 0.08576324582099915,
    20: 0.08421827107667923,
    21: 0.0810227245092392,
    22: 0.08042223751544952,
    23: 0.08126817643642426,
    24: 0.08091344684362411,
    25: 0.08063941448926926,
    26: 0.07939343899488449,
    27: 0.07779276371002197,
    28: 0.08009645342826843,
    29: 0.08032310009002686,
    30: 0.08099385350942612,
    31: 0.08048387616872787,
    32: 0.08217594772577286,
    33: 0.08145556598901749,
    34: 0.08151838928461075,
    35: 0.08218985795974731
}

base_value = data[1]
x = [(n - 1) * 10 for n in data.keys()]
y = [v / base_value * 100 for v in data.values()]

plot_n = 15
x = x[:plot_n]
y = y[:plot_n]

plt.figure(figsize=(8, 6))
index = np.arange(len(x))
plt.bar(index, y, color='red', alpha=0.7, width=0.5)
plt.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5)
plt.xticks(index, [str(x[i]) if i % 2 == 0 else '' for i in range(len(x))], rotation=45, ha='right')
plt.xlabel('Finetuning Epoch', fontsize=18)
plt.ylabel('Masked Feature Reconstruction Loss (%)', fontsize=18)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bothp_forgetting_degree.pdf', format='pdf', bbox_inches='tight')
plt.close()

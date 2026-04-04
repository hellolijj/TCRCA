import matplotlib.pyplot as plt

# 设置字体
plt.rcParams.update({'font.size': 10})


# 参数设置
params = {
    'Time Window': [2, 4, 6, 8, 10],
    'Scale S': [2, 4, 8, 16],
    'GNN Layers': [2, 3, 4],
    'Gamma': [0.05, 0.1, 0.2]
}

# TrainTicket 数据集指标
tt_metrics = {
    'Time Window': {
        'A@1': [0.722, 0.769, 0.782, 0.757, 0.765],
        'A@3': [0.891, 0.938, 0.952, 0.941, 0.946],
        'A@5': [0.971, 0.991, 0.995, 0.982, 0.986],
        'Avg@5': [0.743, 0.854, 0.878, 0.863, 0.873]
    },
    'Scale S': {
        'A@1': [0.758, 0.761, 0.782, 0.769],
        'A@3': [0.919, 0.921, 0.952, 0.938],
        'A@5': [0.983, 0.986, 0.995, 0.991],
        'Avg@5': [0.79, 0.802, 0.878, 0.861]
    },
    'GNN Layers': {
        'A@1': [0.758, 0.782, 0.744],
        'A@3': [0.925, 0.952, 0.912],
        'A@5': [0.991, 0.995, 0.961],
        'Avg@5': [0.79, 0.878, 0.855]
    },
    'Gamma': {
        'A@1': [0.771, 0.782, 0.764],
        'A@3': [0.943, 0.952, 0.941],
        'A@5': [0.991, 0.995, 0.982],
        'Avg@5': [0.856, 0.878, 0.862]
    }
}

# AIOps 数据集指标
aiops_metrics = {
    'Time Window': {
        'A@1': [0.768, 0.789, 0.829, 0.803, 0.816],
        'A@3': [0.928, 0.938, 0.994, 0.946, 0.968],
        'A@5': [0.973, 0.986, 1.0, 0.982, 0.986],
        'Avg@5': [0.832, 0.867, 0.941, 0.904, 0.926]
    },
    'Scale S': {
        'A@1': [0.803, 0.819, 0.829, 0.812],
        'A@3': [0.926, 0.961, 0.994, 0.973],
        'A@5': [0.983, 0.991, 1.0, 0.995],
        'Avg@5': [0.888, 0.914, 0.941, 0.927]
    },
    'GNN Layers': {
        'A@1': [0.803, 0.829, 0.801],
        'A@3': [0.963, 0.994, 0.961],
        'A@5': [0.995, 1.0, 0.973],
        'Avg@5': [0.9, 0.941, 0.912]
    },
    'Gamma': {
        'A@1': [0.812, 0.829, 0.813],
        'A@3': [0.978, 0.994, 0.981],
        'A@5': [0.995, 1.0, 0.986],
        'Avg@5': [0.921, 0.941, 0.927]
    }
}

# 指标颜色和样式
styles = {
    'A@1': ('o', 'blue'),
    'A@3': ('^', 'green'),
    'A@5': ('s', 'orange'),
    'Avg@5': ('D', 'red')
}

fig, axes = plt.subplots(2, 4, figsize=(18, 6))
titles = list(params.keys())

# 绘图
for i, param in enumerate(titles):
    for metric, (marker, color) in styles.items():
        axes[0, i].plot(params[param], tt_metrics[param][metric], marker=marker, label=metric, color=color)
        axes[1, i].plot(params[param], aiops_metrics[param][metric], marker=marker, label=metric, color=color)
    axes[0, i].set_title(f'{param} (TrainTicket)')
    axes[1, i].set_title(f'{param} (AIOps)')
    axes[0, i].set_ylim(0.7, 1.01)
    axes[1, i].set_ylim(0.75, 1.01)
    axes[0, i].grid(True, linestyle='--', alpha=0.5)
    axes[1, i].grid(True, linestyle='--', alpha=0.5)

    # 设置坐标轴标签
    axes[1, i].set_xlabel(param, fontsize=12)

for ax in axes[:, 0]:
    ax.set_ylabel("Score")

# # 图例
# axes[1, -1].legend(loc='lower right', fontsize=9)
# 修改图例位置为底部
fig.legend(loc='center', bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=9)

plt.tight_layout()
plt.show()

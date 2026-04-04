import matplotlib.pyplot as plt

# 设置全局字体、尺寸、中文/英文显示（SCI标准）
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.8,  # 边框粗细
    'grid.linewidth': 0.5,  # 网格线粗细
})

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

# 指标颜色和样式（SCI更清晰的配色+标记）
styles = {
    'A@1': ('o', '#1f77b4'),     # 蓝色
    'A@3': ('^', '#2ca02c'),     # 绿色
    'A@5': ('s', '#ff7f0e'),     # 橙色
    'Avg@5': ('D', '#d62728')    # 红色
}

# 调整画布尺寸，更适合SCI排版
fig, axes = plt.subplots(2, 4, figsize=(20, 7))
titles = list(params.keys())

# 循环绘图
for i, param in enumerate(titles):
    for metric, (marker, color) in styles.items():
        # 绘制 TrainTicket
        axes[0, i].plot(params[param], tt_metrics[param][metric],
                       marker=marker, color=color, linestyle='-', linewidth=1.5,
                       markersize=5, label=metric)
        # 绘制 AIOps
        axes[1, i].plot(params[param], aiops_metrics[param][metric],
                       marker=marker, color=color, linestyle='-', linewidth=1.5,
                       markersize=5)

    # 子图标题
    axes[0, i].set_title(f'{param} (TrainTicket)', fontsize=12, pad=8)
    axes[1, i].set_title(f'{param} (AIOps)', fontsize=12, pad=8)

    # Y轴范围
    axes[0, i].set_ylim(0.7, 1.02)
    axes[1, i].set_ylim(0.75, 1.02)

    # 网格
    axes[0, i].grid(True, linestyle='--', alpha=0.6)
    axes[1, i].grid(True, linestyle='--', alpha=0.6)

    # X轴标签
    axes[1, i].set_xlabel(param, fontsize=11)

# 统一设置 Y轴标签
axes[0, 0].set_ylabel("Score", fontsize=12)
axes[1, 0].set_ylabel("Score", fontsize=12)

# ===================== 核心修复：图例 =====================
# 放在整张图底部居中，不挤压子图
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)

# 调整子图间距，防止重叠（关键！）
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, wspace=0.25, hspace=0.3)

plt.show()
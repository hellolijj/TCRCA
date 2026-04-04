import networkx as nx
import matplotlib.pyplot as plt

# 构建一个有向图表示故障传播路径
G = nx.DiGraph()

# 节点及其异常程度（颜色深度）
nodes = {
    "API Gateway": 0.4,
    "Order Service": 0.7,
    "Inventory Service": 1.0,
    "Payment Service": 0.6,
    "External Bank API": 0.2,
}

# 添加节点
for node, score in nodes.items():
    G.add_node(node, score=score)

# 添加传播路径
edges = [
    ("Inventory Service", "Order Service"),
    ("Order Service", "Payment Service"),
    ("Payment Service", "API Gateway"),
    ("Payment Service", "External Bank API"),
]

G.add_edges_from(edges)

# 设置布局
pos = nx.spring_layout(G, seed=42)

# 绘图
plt.figure(figsize=(10, 6))
node_colors = [nodes[n] for n in G.nodes]
nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors, cmap=plt.cm.Reds, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{u.split()[0]}→{v.split()[0]}" for u, v in G.edges}, font_size=8)

plt.title("Case: Cascading Fault Propagation in Payment System")
plt.tight_layout()
plt.show()

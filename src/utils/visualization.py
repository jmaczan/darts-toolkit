import matplotlib.pyplot as plt
import networkx as nx


def visualize_architecture(architecture):
    G = nx.DiGraph()
    G.add_node("input")

    for i, node_ops in enumerate(architecture):
        node_name = f"node_{i}"
        G.add_node(node_name)

        for j, (input_idx, op_type) in enumerate(node_ops):
            if input_idx == 0:
                G.add_edge("input", node_name, operation=op_type.__name__)
            else:
                G.add_edge(f"node_{input_idx-1}", node_name, operation=op_type.__name__)

    G.add_node("output")
    G.add_edge(f"node_{len(architecture) - 1}", "output")

    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1500,
        font_size=10,
        arrows=True,
    )
    edge_labels = nx.get_edge_attributes(G, "operation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Architecture")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

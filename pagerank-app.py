import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from networkx.exception import PowerIterationFailedConvergence

st.set_page_config(layout="wide")
st.title("Interactive PageRank Demo")

st.markdown("""
Change the structure of the web graph and the damping factor (β), then see how PageRank scores and node sizes update in real time :)
""")

st.sidebar.header("Graph settings")
beta = st.sidebar.slider("Damping factor (β)", 0.01, 0.99, 0.85, 0.01)
max_iter = st.sidebar.slider("Max iterations", 10, 500, 100, step=10)
layout_type = st.sidebar.selectbox("Graph Layout", ["Spring", "Circular", "Shell"])

default_edges = "A B\nB C\nC A\nD C\nE C\nE D"
edges_input = st.sidebar.text_area("Enter edges (FROM TO, one per line):", default_edges, height=150)

edges = []
nodes = set()
for line in edges_input.strip().split("\n"):
    parts = line.strip().split()
    if len(parts) == 2:
        src, tgt = parts
        edges.append((src, tgt))
        nodes.update([src, tgt])
    elif len(parts) == 1:
        nodes.add(parts[0])

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

if len(G.nodes) == 0:
    st.error("Please enter at least one valid node or edge.")
    st.stop()

isolated = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
dangling = [n for n in G.nodes() if G.out_degree(n) == 0 and n not in isolated]

if dangling:
    st.warning(f"Dangling nodes (no out-links): {', '.join(dangling)}")
if isolated:
    st.info(f"Isolated nodes (no in- or out-links): {', '.join(isolated)}")

try:
    pagerank = nx.pagerank(G, alpha=beta, max_iter=max_iter)
except PowerIterationFailedConvergence:
    st.error(f"PageRank did not converge within {max_iter} iterations.")
    st.stop()

pr_values = np.array(list(pagerank.values()))
min_size, max_size = 500, 3000
size_range = np.ptp(pr_values)
sizes = min_size + (pr_values - pr_values.min()) / (size_range + 1e-9) * (max_size - min_size)

if layout_type == "Spring":
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G)))
elif layout_type == "Circular":
    pos = nx.circular_layout(G)
else:
    pos = nx.shell_layout(G)

fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=pr_values, cmap=cm.Set3, ax=ax)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, ax=ax)
labels = {node: f"{node}\n{pagerank[node]:.2f}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

ax.set_title(f"PageRank visualization (β = {beta})")
ax.axis("off")
st.pyplot(fig)

top_nodes = sorted(pagerank.items(), key=lambda x: -x[1])[:3]
st.markdown("### Top ranked pages")
for i, (node, score) in enumerate(top_nodes, 1):
    st.markdown(f"**{i}. Node {node}** — PageRank: `{score:.4f}`")


data = []
for node, pr in sorted(pagerank.items(), key=lambda x: -x[1]):
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    data.append({
        "Node": node,
        "PageRank": round(pr, 4),
        "In-Links": in_degree,
        "Out-Links": out_degree
    })

df = pd.DataFrame(data)
st.markdown("### All PageRank scores with link stats")
st.table(df)
st.download_button("Download as CSV", df.to_csv(index=False), "pagerank_scores.csv", "text/csv")

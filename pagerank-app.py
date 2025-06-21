import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from networkx.exception import PowerIterationFailedConvergence

st.set_page_config(layout="wide")
st.title("PageRank Demo")

st.markdown("""
Change the structure of the web graph and the damping factor (β) to see how PageRank scores update in real time :)
""")

st.sidebar.header("Graph Controls")
beta = st.sidebar.slider("Damping Factor (β)", 0.01, 0.99, 0.85, 0.01)
max_iter = st.sidebar.slider("Max Iterations", 10, 500, 100, step=10)

default_edges = "A B\nB C\nC A\nD C\nE C\nE D"
edges_input = st.sidebar.text_area("Enter edges (one per line, format: FROM TO):", default_edges, height=150)

edges = []
for line in edges_input.strip().split("\n"):
    try:
        src, tgt = line.strip().split()
        edges.append((src, tgt))
    except:
        continue

G = nx.DiGraph()
G.add_edges_from(edges)

if len(G.nodes) == 0:
    st.error("Please enter at least one valid edge.")
    st.stop()

try:
    pagerank = nx.pagerank(G, alpha=beta, max_iter=max_iter)
except PowerIterationFailedConvergence:
    st.error(f"PageRank did not converge within {max_iter} iterations.")
    st.stop()

pr_values = np.array(list(pagerank.values()))
min_size, max_size = 500, 3000
size_range = np.ptp(pr_values)
sizes = min_size + (pr_values - pr_values.min()) / (size_range + 1e-9) * (max_size - min_size)

pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=pr_values, cmap=cm.viridis, ax=ax)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, ax=ax)
labels = {node: f"{node}\n{pagerank[node]:.2f}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

ax.set_title(f"PageRank visualization (β = {beta})")
ax.axis("off")
st.pyplot(fig)

st.markdown("### Final PageRank scores")
st.table([{ "Node": k, "PageRank": round(v, 4) } for k, v in sorted(pagerank.items(), key=lambda x: -x[1])])

#
#   HEAD
#

# HEAD -> MODULES
import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# HEAD -> ADJACENCY MATRIX
A = np.array([
    [0, 7, 14, 12, 0, 0, 0],#7
    [7, 0, 0, 0, 0, 0, 0],#8
    [14, 0, 0, 0, 6, 7, 0],#9
    [12, 0, 0, 0, 0, 0, 7],#10
    [0, 0, 6, 0, 0, 0, 9],#11
    [0, 0, 7, 0, 0, 0, 4],#12
    [0, 0, 0, 7, 9, 4, 0]#13
]) * 10**6 # numbers on the graph are incorrect serial node ID
#A = np.array([
#    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#0
#    [2, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#1
#    [0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#2
#    [0, 5, 4, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],#3
#    [0, 0, 0, 12, 0, 6, 3, 0, 0, 0, 0, 0, 0, 0],#4
#    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],#5
#    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],#6
#    [0, 0, 0, 0, 0, 0, 0, 0, 7, 14, 12, 0, 0, 0],#7
#    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],#8
#    [0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 6, 7, 0],#9
#    [0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 7],#10
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 9],#11
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 4],#12
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 9, 4, 0]#13
#]) * 10**6
#A = np.array([
#    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#0
#    [2, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#1
#    [0, 1, 0, 4, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0],#2
#    [0, 5, 4, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],#3
#    [0, 0, 0, 12, 0, 6, 3, 0, 0, 0, 0, 0, 0, 0],#4
#    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],#5
#    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],#6
#    [0, 0, 20, 0, 0, 0, 0, 0, 7, 14, 12, 0, 0, 0],#7
#    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],#8
#    [0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 6, 7, 0],#9
#    [0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 7],#10
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 9],#11
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 4],#12
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 9, 4, 0]#13
#]) * 10**6

# HEAD -> CONSTANTS
n = len(A)
alpha = (np.e / np.pi / np.max(np.linalg.eigvals(A))).real
beta = 10

# HEAD -> SHOW
print(A)
print(alpha)


#
#   KATZ
#

# KATZ -> COMPUTE
C = np.linalg.inv(np.eye(n) - alpha * A.T) @ (beta * np.ones((n, 1)))

# KATZ -> SHOW
print(np.round(C, 0))
    

#
#   VOTING POWER
#

# VOTING POWER -> REDUCTION FN
def af(a: int, b: int) -> float:
    if a == b:
        return 1.0
    n = len(A)
    visited = [False] * n
    depth_of = {}
    depth_count = {}
    q = deque()
    q.append((a, 0))
    visited[a] = True
    depth_of[a] = 0
    depth_count[0] = 1
    while q:
        node, d = q.popleft()
        for nei in range(n):
            if A[node, nei] != 0 and not visited[nei]:
                visited[nei] = True
                depth = d + 1
                depth_of[nei] = depth
                depth_count[depth] = depth_count.get(depth, 0) + 1
                q.append((nei, depth))
    if b not in depth_of:
        return 0.0
    depth_b = depth_of[b]
    cumulative = sum(depth_count[d] for d in range(depth_b + 1))
    return 1.0 / cumulative

# VOTING POWER -> REDUCTION
R = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        R[i, j] = af(i, j)
print(R**-1)

# VOTING POWER -> CALCULATION
V = R * C.T

# VOTING POWER -> SHOW
print(np.round(V, 0))

# for decision now: get V_i, dot product with decisions S_c



#
#   GRAPH STUFF
#

G = nx.DiGraph()
for i in range(n):
    G.add_node(i)
for i in range(n):
    for j in range(n):
        if A[i, j] != 0:
            G.add_edge(i, j, weight=A[i, j]*10**-6)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=[
    f'#{int(255*(1 - C[i,0]/np.max(C))):02x}{int(255*(C[i,0]/np.max(C))):02x}00'
    for i in range(len(C))
], node_size=[1.5*beta*C[index] for index in range(len(C))])

# Draw edges
# Draw edges with curved lines for bidirectional visibility
drawn_pairs = set()
for u, v in G.edges():
    if (v, u) in G.edges() and (v, u) not in drawn_pairs:
        # Draw u->v curved one way, v->u curved opposite
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=G[u][v]['weight'],
            arrowstyle='-',
            arrowsize=10,
            connectionstyle='arc3,rad=0.0'
        )
        drawn_pairs.add((u, v))
        drawn_pairs.add((v, u))
    elif (v, u) not in G.edges():
        # Single edge or self-loop
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=G[u][v]['weight'],
            arrowstyle='-',
            arrowsize=10,
            connectionstyle='arc3,rad=0.0'
        )


# Draw labels
nx.draw_networkx_labels(G, pos, font_size=14, font_color='black')

plt.axis('off')
plt.show()

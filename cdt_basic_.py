import cdt
import networkx as nx

data, graph = cdt.data.load_dataset('sachs')
print(data.head())

glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(data)
print(skeleton)

print(nx.adjacency_matrix(skeleton).todense())

new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
print(nx.adjacency_matrix(new_skeleton).todense())

model = cdt.causality.graph.GES()
output_graph = model.predict(data, new_skeleton)
print(nx.adjacency_matrix(output_graph).todense())

from cdt.metrics import (precision_recall, SID, SHD)
scores = [metric(graph, output_graph) for metric in (precision_recall, SID, SHD)]
print(scores)

model2 = cdt.causality.graph.CAM()
output_graph_nc = model2.predict(data)
scores_nc = [metric(graph, output_graph_nc) for metric in (precision_recall, SID, SHD)]
print(scores_nc)

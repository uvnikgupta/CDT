import cdt

generator = cdt.data.AcyclicGraphGenerator('gp_add', noise_coeff=.2, nodes=20, parents_max=3)
data, graph = generator.generate()
data.head()

sam = cdt.causality.graph.SAM(nruns=12)
prediction = sam.predict(data)

from cdt.metrics import (precision_recall, SHD)
[metric(graph, prediction) for metric in (precision_recall, SHD)]

from scm.dynamic_scm import DynamicSCM
from dowhy import CausalModel
import dowhy.datasets
import networkx as nx

def get_gml(scm):
    G = nx.DiGraph(eval(str(scm.dag.adj)))
    nx.write_gml(G, path="scm.gml")

    with open("scm.gml") as f:
        dg = f.readlines()

    return ''.join([x.rstrip('\n') for x in dg])

def get_scm():
    input_nodes = [5,2,1]
    dSCM = DynamicSCM(min_parents=5, max_parents=6, 
                    parent_levels_probs=[1, 1], 
                    simple_operations={"+":1})
    scm = dSCM.create(input_nodes)
    return scm

if __name__ == "__main__":
    scm = get_scm()
    df = scm.sample(10000)
    gml = get_gml(scm)
    
    model = CausalModel(
        data=df,
        treatment="B1",
        outcome="C1",
        graph=get_gml(scm))
    
    identified_estimand = model.identify_effect()

# Estimate the target estimand using a statistical method.
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_matching")

# Refute the obtained estimate using multiple robustness checks.
# refute_results = model.refute_estimate(identified_estimand, estimate,
#                                        method_name="random_common_cause")


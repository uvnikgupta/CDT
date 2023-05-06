from scmodels import SCM
import random, math, copy


class DynamicSCM():

    def __init__(self):
        pass

    def get_data(self):
        return copy.deepcopy(self.__data)

    def get_distributions(self):
        dists = [
            f"LogLogistic(alpha={random.randint(5, 20)}, beta={round(random.uniform(1,3.5),1)})",
            f"Normal(mean={random.randint(0,10)}, std={round(random.uniform(1,20),2)})",
            f"LogNormal(mean={random.randint(0,10)}, std={round(random.uniform(1,20),2)})",
            f"Benini(alpha={round(random.uniform(.1, 1.0), 1)}, beta={round(random.uniform(.1, 1.0), 1)}, sigma={round(random.uniform(.1, 1.0), 1)})",
            f"Beta(alpha={round(random.uniform(.1, 1.0), 1)}, beta={round(random.uniform(.1, 1.0), 1)})",
            f"Exponential(rate={round(random.uniform(.5, 10.0), 1)})",
            f"FDistribution(d1={random.randint(2, 4)}, d2={random.randint(5, 8)})",
            f"Gamma(k={round(random.uniform(0.1, 4.0), 1)}, theta={round(random.uniform(2.0, 8.0), 1)})",
            f"GammaInverse(a={random.randint(1, 4)}, b={random.randint(2, 8)})",
            f"Bernoulli({round(random.random(), 1)})",
            f"Binomial(n={random.randint(10, 20)}, p={round(random.uniform(0.05, 1.00), 2)}, succ={random.randint(2, 10)})",
            f"BetaBinomial(n={random.randint(10, 100)}, alpha={round(random.uniform(.1, 1.0), 1)}, beta={round(random.uniform(2, 5),1)})",
            f"Die(sides={random.randint(4, 10)})",
            f"FiniteRV({{1: 0.33, 2: 0.34, 3: 0.33}})",
            f"Geometric(p={round(random.uniform(0.05, 1.00), 2)})",
            f"Poisson(lamda={round(random.uniform(0.05, 1.00), 1)})",
            f"FiniteRV({{{random.randint(5, 10)}: 0.5, {random.randint(0, 3)}: 0.16, {random.randint(15, 25)}: 0.17, {random.randint(30, 50)}: 0.17}})"
        ]
        return dists

    def add_noise(self, dist):
        noise_ops = [False, True]
        op = random.choices(noise_ops, weights=(5, 1), k=1)[0]
        if op:
            dist = f"( {dist} ) * N"
        return dist

    def add_complex_operation(self, dist):
        complex_ops = [False, "sqrt", "**2"]
        op = random.choices(complex_ops, weights=(100, 1, 1), k=1)[0]
        if op:
            if "*" in op:
                dist = f"( {dist} ){op}"
            else:
                dist = f"{op} ( {dist} )"
        return dist

    def create_simple_operation(self, dist, parent):
        simple_ops = ["+", "*", "()"]
        op_1 = random.choices(simple_ops, weights=(10, 10, 1), k=1)[0]
        if op_1 == "()":
            op_2 = random.choices(simple_ops, weights=(1, 1, 0), k=1)[0]
            dist = f"{op_1[0]} {dist} {op_2} {parent} {op_1[1]}"
        else:
            dist = f"{dist} {op_1} {parent}"
        return dist

    def get_distribution(self, parents):
        if not parents:
            dist = "N"
        else:
            dist = f"{parents[0]}"
            count = 0
            for p in parents[1:]:
                dist = self.create_simple_operation(dist, p)
                count += 1

                if count > 2:
                    dist = self.add_complex_operation(dist)
                    count = 0

                dist = self.add_noise(dist)

            dist = f"{dist} * N"
        return dist

    def get_parent_levels(self, levels_and_distributions):
        levels = list(range(len(levels_and_distributions)))
        level_probs = [round(math.exp(1.5 * l), 1) for l in levels]
        parent_levels = []
        for l in levels:
            possible_level = random.choices(levels, level_probs, k=1)[0]
            if possible_level not in parent_levels:
                parent_levels.append(possible_level)
        return parent_levels

    def get_parents(self, levels_and_distributions):
        parents = []
        parent_levels = self.get_parent_levels(levels_and_distributions)
        for level in parent_levels:
            possible_parents = levels_and_distributions[level][1]
            num_parents = random.randint(1, len(possible_parents))
            parents.extend(random.sample(possible_parents, num_parents))
        return parents

    def populate_level_distributions(self, level, level_data,
                                     levels_and_distributions):
        level_dists = []
        dist_names = []
        for n in range(level_data['num']):
            name = level_data['name'] + str(n + 1)
            dist = self.get_distribution(
                self.get_parents(levels_and_distributions))

            dist = f"{name} = {dist}, N ~ {random.sample(self.get_distributions(), 1)[0]}"
            level_dists.append(dist)
            dist_names.append(name)
        levels_and_distributions[level] = (level_dists, dist_names)

    def create_scm(self, level_name_numnodes):
        levels_and_distributions = {}
        for level, level_data in level_name_numnodes.items():
            self.populate_level_distributions(level, level_data,
                                              levels_and_distributions)

        scm_dists = []
        for n in range(len(levels_and_distributions)):
            scm_dists.extend(levels_and_distributions[n][0])

        self.__data = levels_and_distributions
        scm = SCM(scm_dists)
        return scm

    def create_names_and_num_nodes_dict(self, nodes_per_level, level_names):
        level_name_numnodes = {}
        for n, name in enumerate(level_names):
            level_name_numnodes[n] = {'name': name, 'num': nodes_per_level[n]}
        return level_name_numnodes

    def get_level_names(self, numnodes_per_level):
        level_names = []
        for n in range(len(numnodes_per_level)):
            level_names.append(chr(65 + n))
        return level_names

    def create_scm_from_nodes_list(self, nodes_list):
        level_names = self.get_level_names(nodes_list)
        level_name_numnodes = self.create_names_and_num_nodes_dict(
            nodes_list, level_names)

        scm = self.create_scm(level_name_numnodes)
        return scm

    def get_numnodes_per_level(self, input_nodes):
        nodes = input_nodes
        levels = int(math.pow(nodes, 1 / 3))
        numnodes_per_level = []
        for n in range(levels):
            ub = round(math.log(nodes)) - 0.6
            lb = ub - 0.7
            level_nodes = int(math.exp(round(random.uniform(lb, ub), 1)))
            if level_nodes == 0: break

            numnodes_per_level.append(level_nodes)
            nodes -= level_nodes

        return numnodes_per_level

    def create_scm_from_num_nodes(self, input_nodes):
        nodes_list = self.get_numnodes_per_level(input_nodes)
        scm = self.create_scm_from_nodes_list(nodes_list)
        return scm

    def create(self, input_nodes):
        if type(input_nodes) != list:
            scm = self.create_scm_from_num_nodes(input_nodes)
        else:
            scm = self.create_scm_from_nodes_list(input_nodes)
        return scm


if __name__ == "__main__":
    input_nodes = [2, 2, 2, 2, 2, 1]
    dSCM = DynamicSCM()
    scm = dSCM.create(input_nodes)
    scm.plot(node_size=250, savepath="scm.jpg")

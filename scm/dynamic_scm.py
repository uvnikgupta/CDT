from scmodels import SCM
import random, math, copy


class DynamicSCM():
    def __init__(self,
                 min_parents: int = 1,
                 max_parents: int = 10000,
                 parent_levels_probs: list[float] = [],
                 distributions_file: str = "distributions.txt",
                 simple_operations: dict[str, int] = {},
                 complex_operations: dict[str, int] = {}):
        self.__distributions_file = distributions_file
        self.__max_parents = max_parents
        self.__min_parents = min_parents
        self.__parent_levels_probs = parent_levels_probs
        self.__simple_operations = {
            "+": 1
        } if not simple_operations else simple_operations
        self.__complex_operations = {
            False: 1
        } if not complex_operations else complex_operations
        self.__distributions = []
        self.__data = None

    def __get_default_distributions(self):
        dists = [
            'f"LogLogistic(alpha={random.randint(5, 20)}, beta={round(random.uniform(1,3.5),1)})"',
            'f"Normal(mean={random.randint(0,10)}, std={round(random.uniform(1,20),2)})"',
            'f"LogNormal(mean={random.randint(0,10)}, std={round(random.uniform(1,20),2)})"',
            'f"Benini(alpha={round(random.uniform(.1, 1.0), 1)}, beta={round(random.uniform(.1, 1.0), 1)}, sigma={round(random.uniform(.1, 1.0), 1)})"',
            'f"Beta(alpha={round(random.uniform(.1, 1.0), 1)}, beta={round(random.uniform(.1, 1.0), 1)})"',
            'f"Exponential(rate={round(random.uniform(.5, 10.0), 1)})"',
            'f"FDistribution(d1={random.randint(2, 4)}, d2={random.randint(5, 8)})"',
            'f"Gamma(k={round(random.uniform(0.1, 4.0), 1)}, theta={round(random.uniform(2.0, 8.0), 1)})"',
            'f"GammaInverse(a={random.randint(1, 4)}, b={random.randint(2, 8)})"',
            'f"Bernoulli({round(random.random(), 1)})"',
            'f"Binomial(n={random.randint(10, 20)}, p={round(random.uniform(0.05, 1.00), 2)}, succ={random.randint(2, 10)})"',
            'f"BetaBinomial(n={random.randint(10, 100)}, alpha={round(random.uniform(.1, 1.0), 1)}, beta={round(random.uniform(2, 5),1)})"',
            'f"Die(sides={random.randint(4, 10)})"',
            'f"FiniteRV({{1: 0.33, 2: 0.34, 3: 0.33}})"',
            'f"Geometric(p={round(random.uniform(0.05, 1.00), 2)})"',
            'f"Poisson(lamda={round(random.uniform(0.05, 1.00), 1)})"',
            'f"FiniteRV({{{random.randint(5, 10)}: 0.5, {random.randint(0, 3)}: 0.16, {random.randint(15, 25)}: 0.17, {random.randint(30, 50)}: 0.17}})"'
        ]
        return dists

    def __read_distributions_from_file(self):
        try:
            with open(self.__distributions_file) as file:
                self.__distributions = file.readlines()
        except:
            print(
                f"WARNING: {self.__distributions_file} not found. Loading default set of distributions."
            )
            self.__distributions = self.__get_default_distributions()

    def __get_distributions(self):
        if not self.__distributions:
            self.__read_distributions_from_file()
        return self.__distributions

    def __add_noise(self, dist: str):
        noise_ops = [False, True]
        noise = random.choices(noise_ops, weights=(5, 1), k=1)[0]
        if noise:
            if "*" in self.__simple_operations:
                dist = f"( {dist} ) * N"
            else:
                dist = f"( {dist} ) + N"
        return dist

    def __add_complex_operation(self, dist: str):
        complex_ops = list(self.__complex_operations.keys())
        weights = list(self.__complex_operations.values())
        op = random.choices(complex_ops, weights=weights, k=1)[0]
        if op:
            if "*" in op:
                dist = f"( {dist} ){op}"
            else:
                dist = f"{op} ( {dist} )"
        return dist

    def __create_simple_operation(self, dist: str, parent: str):
        simple_ops = list(self.__simple_operations.keys())
        weights = list(self.__simple_operations.values())
        op_1 = random.choices(simple_ops, weights=weights, k=1)[0]
        if op_1 == "()":
            weights[[n for n, sop in enumerate(simple_ops)
                     if sop == '()'][0]] = 0
            op_2 = random.choices(simple_ops, weights=weights, k=1)[0]
            dist = f"{op_1[0]} {dist} {op_2} {parent} {op_1[1]}"
        else:
            dist = f"{dist} {op_1} {parent}"
        return dist

    def __get_node_formula(self, parents: list[str]):
        if not parents:
            dist = "N"
        else:
            dist = f"{parents[0]}"
            count = 0
            for p in parents[1:]:
                dist = self.__create_simple_operation(dist, p)
                count += 1

                if count > 2:
                    dist = self.__add_complex_operation(dist)
                    count = 0

                dist = self.__add_noise(dist)

            dist = f"{dist} * N"
        return dist

    def __check_and_fix_num_elements(self, list_to_check: list,
                                     desired_count: int):
        if len(list_to_check) > desired_count:
            list_to_check = list_to_check[:desired_count]
        elif len(list_to_check) < desired_count:
            diff = desired_count - len(list_to_check)
            list_to_check.extend(diff * [0])
        return list_to_check

    def __get_parent_levels(self, levels_and_distributions):
        '''
        Applies a probalistic method to select the parent levels 
        from which the parents for a new node needs to be selected. 
        It does so by checking if the user has provided any 
        desired probabilities. If yes, the specified probabilities 
        are used else an exponentially decreasing function is used
        to derive the probabilities such that the highest 
        probability is given to the immediate parent and the
        lowest probability to the greatest grand parent.
        '''
        parent_levels = []
        num_levels = len(levels_and_distributions)
        levels = list(reversed(range(num_levels)))

        if self.__parent_levels_probs:
            level_probs = self.__parent_levels_probs
            level_probs = self.__check_and_fix_num_elements(
                level_probs, num_levels)
        else:
            level_probs = [round(math.exp(1.5 * l), 1) for l in levels]

        if levels:
            parent_levels = random.choices(levels, level_probs, k=num_levels)
            parent_levels = list(set(parent_levels))
        return parent_levels

    def __get_min_max_parents(self, possible_parents):
        if self.__max_parents < len(possible_parents):
            max_parents = self.__max_parents
        else:
            max_parents = len(possible_parents)

        if self.__min_parents > max_parents:
            min_parents = max_parents
        else:
            min_parents = self.__min_parents
        return min_parents, max_parents

    def __get_parents(self, levels_and_distributions):
        '''
        Randomly chooses the parent nodes for a new child node that 
        is getting created using the following process:
        1. Get the parent levels from which to select the parents. 
        2. From each selected level 
            a. Get the min and max number of parents
            b. Generate a random number between min and max
            c. Randomnly select those many parents from all the 
               parents at this level
            d. Add the selected parents to the parent list
        Because the sampling is done at each level the final list 
        may now have more than the max number of parents. Hence 
        once again max number of parents is sampled from the final 
        parents list.
        '''
        parents = []
        parent_levels = self.__get_parent_levels(levels_and_distributions)
        for level in parent_levels:
            possible_parents = levels_and_distributions[level][1]
            min_parents, max_parents = self.__get_min_max_parents(
                possible_parents)
            num_parents = random.randint(min_parents, max_parents)
            parents.extend(random.sample(possible_parents, num_parents))

        _, max_parents = self.__get_min_max_parents(parents)
        parents = random.sample(parents, max_parents)
        return parents

    def __populate_level_distributions(self, level, level_data,
                                       levels_and_distributions):
        level_dists = []
        dist_names = []
        for n in range(level_data['num']):
            name = level_data['name'] + str(n + 1)
            dist = self.__get_node_formula(
                self.__get_parents(levels_and_distributions))
            sampled_dist = random.sample(self.__get_distributions(), 1)[0]
            dist = f"{name} = {dist}, N ~ {eval(sampled_dist)}"
            level_dists.append(dist)
            dist_names.append(name)
        levels_and_distributions[level] = (level_dists, dist_names)

    def __create_scm(self, level_name_numnodes):
        levels_and_distributions = {}
        for level, level_data in level_name_numnodes.items():
            self.__populate_level_distributions(level, level_data,
                                                levels_and_distributions)

        scm_dists = []
        for n in range(len(levels_and_distributions)):
            scm_dists.extend(levels_and_distributions[n][0])

        self.__data = levels_and_distributions
        scm = SCM(scm_dists)
        return scm

    def __create_names_and_num_nodes_dict(self, nodes_per_level: list[int],
                                          level_names: list[str]):
        level_name_numnodes = {}
        for n, name in enumerate(level_names):
            level_name_numnodes[n] = {'name': name, 'num': nodes_per_level[n]}
        return level_name_numnodes

    def __get_level_names(self, numnodes_per_level: list[int]):
        level_names = []
        for n in range(len(numnodes_per_level)):
            level_names.append(chr(65 + n))
        return level_names

    def __create_scm_from_nodes_list(self, nodes_list: list[int]):
        level_names = self.__get_level_names(nodes_list)
        level_name_numnodes = self.__create_names_and_num_nodes_dict(
            nodes_list, level_names)

        scm = self.__create_scm(level_name_numnodes)
        return scm

    def __get_numnodes_per_level(self, input_nodes: int):
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

    def __create_scm_from_num_nodes(self, input_nodes: int):
        nodes_list = self.__get_numnodes_per_level(input_nodes)
        scm = self.__create_scm_from_nodes_list(nodes_list)
        return scm

    def create(self, input_nodes):
        if type(input_nodes) != list:
            scm = self.__create_scm_from_num_nodes(input_nodes)
        else:
            scm = self.__create_scm_from_nodes_list(input_nodes)
        return scm

    def get_data(self):
        return copy.deepcopy(self.__data)


if __name__ == "__main__":
    input_nodes = [2, 2, 2, 2, 2, 1]
    dSCM = DynamicSCM(min_parents=2,
                      max_parents=2,
                      parent_levels_probs=[0.5, 0.5])
    scm = dSCM.create(input_nodes)
    scm.plot(node_size=250, savepath="scm.jpg")

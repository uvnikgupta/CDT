import random, math

class NodeFormula():
    def __init__(self, 
                 simple_operations: dict[str, int] = {},
                 complex_operations: dict[str, int] = {}) -> None:
        self.__simple_operations = simple_operations if simple_operations else {"+": 1}
        self.__complex_operations = complex_operations if complex_operations else {False: 1}
        self.__formula_list = []

    def __check_initialization(self):
        if not self.__formula_list:
            raise Exception("NodeFormula not initialized. \
                            Initialize it with a seed node before doing any operation") 

    def init(self, initial_node:str):
        self.__formula_list = [{"type" : "node",
                                "value" : initial_node}]
    
    def do_simple_operation(self, node:str):
        self.__check_initialization()

        simple_ops = list(self.__simple_operations.keys())
        weights = list(self.__simple_operations.values())
        multiplier = round(random.uniform(0.2, 4.0), 1)
        node = str(multiplier) + "*" + node

        op_1 = random.choices(simple_ops, weights=weights, k=1)[0]
        if op_1 == "()":
            weights[[n for n, sop in enumerate(simple_ops)
                     if sop == '()'][0]] = 0
            op_2 = random.choices(simple_ops, weights=weights, k=1)[0]
            prev_node = self.__formula_list[-1]["value"]
            self.__formula_list[-1]["value"] = f"{op_1[0]} {prev_node}"
            self.__formula_list.append({"type": "operation", "value": op_2})
            self.__formula_list.append({"type": "node", "value": f"{node} {op_1[1]}"})
        else:
            self.__formula_list.append({"type": "operation", "value": op_1})
            self.__formula_list.append({"type": "node", "value": node})

    def add_complex_operation(self):
        self.__check_initialization()

        complex_ops = list(self.__complex_operations.keys())
        weights = list(self.__complex_operations.values())
        op = random.choices(complex_ops, weights=weights, k=1)[0]
        if op:
            idx = [n for n, ele in enumerate(self.__formula_list) if ele["type"] == 'node']
            max_nodes_to_consider = math.ceil(math.log(len(idx)) + 1)
            num_nodes_in_operation = random.randint(1, max_nodes_to_consider)
            start_idx = len(self.__formula_list) - idx[-num_nodes_in_operation]
            start_node = self.__formula_list[-start_idx]["value"]
            if "*" in op:
                start_node = f"( {start_node}"
                end_node = start_node if start_idx == 1 else self.__formula_list[-1]["value"]
                end_node = f"{end_node} ){op}"
            else:
                start_node = f"{op}( {start_node}"
                end_node = start_node if start_idx == 1 else self.__formula_list[-1]["value"]
                end_node = f"{end_node} )"
            self.__formula_list[-start_idx]["value"] = start_node
            self.__formula_list[-1]["value"] = end_node

    def add_noise(self):
        if not self.__formula_list:
            self.__formula_list.append({"type": "node", "value": "N"})
        else:
            if "*" in self.__simple_operations:
                self.__formula_list.append({"type": "operation", "value": "*"})
                self.__formula_list.append({"type": "node", "value": "N"})
            else:
                self.__formula_list.append({"type": "operation", "value": "+"})
                self.__formula_list.append({"type": "node", "value": "N"})

    def __str__(self):
        formula_str = " ".join([element["value"] for element in self.__formula_list])
        return formula_str
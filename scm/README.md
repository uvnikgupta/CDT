# Dynamic Structural Causal Model (SCM)

Create a SCM dynamically by defining either the total number of nodes or a list of number of nodes at each level.<br><br>
The dynamic creation of the SCM can further be controlled using the following optional parameters:
1. min_parents --> Minimum number of parents any child node should have
2. max_parents --> Maximum number of parents any child node should have
3. parent_levels_probs --> List of probabilities to use when selecting the levels from where the parents for any child node should be sampled. The first item in the list is the probability of the immediate parent level getting selected, the second is the probability of the grand parent level getting seleted so on and so forth. Default is an exponentially decaying probability
4. distribution_type --> Type of distributions to be sampled from. Values 1=continuous, 2=discrete. If nothing is specified, mixed type distributions are used
5. distributions_file --> Full path of file containing a list of distributions in the sympy format. Default is <code>./distributions.txt</code>
6. simple_operations --> A dictionary with the operation as the key and the weight, in interger, of that operation in getting selected as the value. For eg <code>{"+":10, "*": 5}</code>. The default is only the additive operation <code>{"+":1}</code>
7. complex_operations --> Similar to simple_operations but meant for complex operations like <code>sqrt, \*\*2, etc</code>. This supports <code>False</code> also as one of the members to enable the usecase that the complex operation need be applied only some times. For eg <code>{False:10, "sqrt":2, "\*\*2":1}</code>. By default no complex operations are applied.

Example use of the tool is available in <code> <a href=https://github.com/uvnikgupta/CDT/blob/master/scm/test_dynamic_scm.ipynb>test_dynamic_scm.ipynb </a></code><br><br>

Contributions and requests for further features/enhancements are welcome.

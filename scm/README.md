# Dynamic Structural Causal Model (SCM)

Create a SCM dynamically by defining either the total number of nodes or a list of number of nodes at each level.<br>
The dynamic creation of the SCM can further be controlled using the following optional parameters:
1. min_parents --> Minimum number of parents any child node should have
2. max_parents --> Maximum number of parents any child node should have
3. parent_levels_probs --> List of probabilities to use when selecting the levels from where the parents for any child node should be sampled
4. distributions_file --> File containing a list of distributions in the sympy format

Example use of the tool is available in <code> test_dynamic_scm.ipynb </code><br><br>

Contributions and requests for further features/enhancements are welcome.
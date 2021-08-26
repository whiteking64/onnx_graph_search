import template_subgraphs


__all__ = ["subgraph_from_py_1"]

subgraph_from_py_1 = template_subgraphs.resblock_plain
del subgraph_from_py_1["relu_2"]
subgraph_from_py_1["add_1"]["output"] = ["output"]

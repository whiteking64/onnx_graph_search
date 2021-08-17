"""Python script for searching an identical subgraph in an onnx model
* TODO:
    - Do some refactor.
"""

import gc
import json

import matplotlib.pyplot as plt
import networkx as nx
import onnx
from google.protobuf.json_format import MessageToJson
from networkx.drawing.nx_pydot import graphviz_layout
from pprint import pprint


def jsonize_mnist_onnx(model_path):
    model = onnx.load(model_path)
    j = MessageToJson(model)
    with open("./resnet18.json", "w") as f:
        print(j, file=f)


def add_empty_label_list(dict_nodes):
    for k, v in dict_nodes.items():
        v["s_label"] = []
        dict_nodes[k] = v
    return dict_nodes


def save_graph(graph, save_name):
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(graph, pos, node_size=100, arrows=True, with_labels=False)
    plt.savefig(save_name, bbox_inches="tight")
    plt.clf()


def add_edges(graph, dict_nodes):
    for k, v in dict_nodes.items():
        for _k, _v in dict_nodes.items():
            if k != _k and len(set(v["output"]) & set(_v["input"])) > 0:
                graph.add_edge(k, _k)
    return graph


def search_basic(src_graph, query_graph):
    GM = nx.algorithms.isomorphism.DiGraphMatcher(src_graph, query_graph)
    return list(GM.subgraph_isomorphisms_iter())


def search_type(subgraph_map_list, src_nodes_dict, query_nodes_dict):
    subgraph_list_matched = []
    for i, matched in enumerate(subgraph_map_list):
        is_same_type = True
        for k, v in matched.items():
            if v == "src":
                continue
            if src_nodes_dict[k]["opType"] != query_nodes_dict[v]["opType"]:
                is_same_type = False
                break
            # print(f"\t{src_nodes_dict[k]['opType']}, {query_nodes_dict[v]['opType']}")
        if is_same_type:
            subgraph_list_matched.append(matched)
    return subgraph_list_matched


# model_path = "./resnet18-v2-7.onnx"
# jsonize_mnist_onnx(model_path)

with open("resnet18.json", "r") as f:
    data = json.load(f)

print(data.keys())
print(data["graph"].keys())
print(data["graph"]["name"])
print(len(data["graph"]["node"]))
node_list = data["graph"]["node"]

graph = nx.DiGraph()

network_nodes = {}
for i in range(len(node_list)):
    node_name = node_list[i]["name"]
    network_nodes[node_name] = {
        "input": node_list[i]["input"],
        "output": node_list[i]["output"],
        "opType": node_list[i]["opType"],
    }
    graph.add_node(node_name)
network_nodes = add_empty_label_list(network_nodes)
del node_list
gc.collect()

graph = add_edges(graph, network_nodes)
# save_graph(graph, "graph.png")

query_nodes = {
    "src": {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    },
    "conv_1": {
        "input": ["input"],
        "output": ["bn_1"],
        "opType": "Conv",
    },
    "bn_1": {
        "input": ["bn_1"],
        "output": ["relu_1"],
        "opType": "BatchNormalization",
    },
    "relu_1": {
        "input": ["relu_1"],
        "output": ["conv_2"],
        "opType": "Relu",
    },
    "conv_2": {
        "input": ["conv_2"],
        "output": ["add_in_1"],
        "opType": "Conv",
    },
    "conv_3": {
        "input": ["input"],
        "output": ["add_in_2"],
        "opType": "Conv",
    },
    "add_1": {
        "input": ["add_in_1", "add_in_2"],
        "output": ["output"],
        "opType": "Add",
    },
}
query_label_base = "resblock"

assert "src" in query_nodes
query_graph = nx.DiGraph()
for key in query_nodes.keys():
    query_graph.add_node(key)

query_graph = add_edges(query_graph, query_nodes)
# save_graph(query_graph, "query_graph.png")

subgraph_list = search_basic(graph, query_graph)
print(len(subgraph_list))

subgraph_list_matched = search_type(subgraph_list, network_nodes, query_nodes)
print(len(subgraph_list_matched))

for i, matched in enumerate(subgraph_list_matched):
    query_label = f"{query_label_base}_{i}"
    # append label to each matched keys
    for k in matched.keys():
        network_nodes[k]["s_label"].append(query_label)

pprint(network_nodes)

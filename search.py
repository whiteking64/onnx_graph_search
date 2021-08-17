"""Python script for searching an identical subgraph in an onnx model
* TODO:
    - The search function is incomplete, fix it.
    - Do some refactor.
"""


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

edges = []
for k, v in network_nodes.items():
    for _k, _v in network_nodes.items():
        if k != _k and len(set(v["output"]) & set(_v["input"])) > 0:
            edges.append([k, _k])
            graph.add_edge(k, _k)
print(len(edges))

pos = graphviz_layout(graph, prog="dot")
nx.draw(graph, pos, node_size=100, arrows=True, with_labels=False)
plt.savefig("graph.png", bbox_inches="tight")
plt.clf()

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
query_label = "resblock"

assert "src" in query_nodes
query_graph = nx.DiGraph()
for key in query_nodes.keys():
    query_graph.add_node(key)

for k, v in query_nodes.items():
    for _k, _v in query_nodes.items():
        if k != _k and len(set(v["output"]) & set(_v["input"])) > 0:
            query_graph.add_edge(k, _k)

pos = graphviz_layout(query_graph, prog="dot")
nx.draw(query_graph, pos, node_size=100, arrows=True, with_labels=False)
plt.savefig("query_graph.png", bbox_inches="tight")
plt.clf()

GM = nx.algorithms.isomorphism.DiGraphMatcher(graph, query_graph)
subgraph_list = list(GM.subgraph_isomorphisms_iter())
print(len(subgraph_list))

subgraph_list_matched = []
for i, matched in enumerate(subgraph_list):
    is_same_type = True
    for k, v in matched.items():
        if v == "src":
            continue
        if network_nodes[k]["opType"] != query_nodes[v]["opType"]:
            is_same_type = False
            break
        # print(f"\t{network_nodes[k]['opType']}, {query_nodes[v]['opType']}")
    if is_same_type:
        subgraph_list_matched.append(matched)
        # append label to each matched keys
        for k in matched.keys():
            network_nodes[k]["s_label"].append(query_label)
print(len(subgraph_list_matched))
pprint(network_nodes)

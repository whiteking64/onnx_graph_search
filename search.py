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


def jsonize_mnist_onnx(model_path):
    model = onnx.load(model_path)
    j = MessageToJson(model)
    with open("./resnet18.json", "w") as f:
        print(j, file=f)


def get_type_list(network_dict, key):
    assert key in ["input", "output"]
    return [network_dict[name]["opType"] for name in network_dict[key]]


class Node:
    def __init__(self, node_name):
        self.node_name = node_name
        self.neighbors = []

    def add_neighbor(self, neighbor):
        assert len(self.neighbors) <= 2
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
        assert len(self.neighbors) <= 2

    def __str__(self):
        print_str = self.node_name
        for neighbor in self.neighbors:
            print_str += f"\n\t-> {neighbor}"
        return print_str


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_name):
        self.nodes[node_name] = Node(node_name)

    def get_node(self, node_name):
        return self.nodes.get(node_name, None)

    def add_edge(self, src, dst):
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)

        self.nodes[src].add_neighbor(dst)

    def __iter__(self):
        return iter(self.nodes.values())


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


edges = []
for k, v in network_nodes.items():
    for _k, _v in network_nodes.items():
        if k != _k and len(set(v["output"]) & set(_v["input"])) > 0:
            edges.append([k, _k])
            graph.add_edge(k, _k)
print(len(edges))

nx.draw(graph, node_size=100, arrows=True, with_labels=False)
plt.savefig("graph.png", bbox_inches="tight")

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
    "dst": {
        "input": ["output"],
        "output": ["output"],
        "opType": "None",
    },
}

query_graph = nx.DiGraph()
for key in query_nodes.keys():
    query_graph.add_node(key)

for k, v in query_nodes.items():
    for _k, _v in query_nodes.items():
        if k != _k and len(set(v["output"]) & set(_v["input"])) > 0:
            query_graph.add_edge(k, _k)

nx.draw(query_graph, node_size=100, arrows=True, with_labels=False)
plt.savefig("query_graph.png", bbox_inches="tight")

GM = nx.algorithms.isomorphism.DiGraphMatcher(graph, query_graph)
print(GM.subgraph_is_isomorphic())

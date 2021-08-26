"""Python script for searching an identical subgraph in an onnx mode
"""

import argparse
import gc
import importlib
import json
import logging
from typing import DefaultDict

import matplotlib.pyplot as plt
import networkx as nx
import onnx
from google.protobuf.json_format import MessageToJson
from networkx.drawing.nx_pydot import graphviz_layout
from pprint import pprint

import template_subgraphs


template_subgraph_names = sorted(name for name in template_subgraphs.__all__)

parser = argparse.ArgumentParser(description="ONNX model subgraph search")
parser.add_argument(
    "-m",
    "--model",
    metavar="MODEL",
    required=True,
    type=str,
    help="source model to be searched",
)
parser.add_argument(
    "-j",
    "--jsons",
    metavar="SUBGRAPHS",
    nargs="*",
    type=str,
    help="json files of query subgraph definitions",
)
parser.add_argument(
    "-t",
    "--templates",
    metavar="TEMPLATE",
    nargs="*",
    type=str,
    default=["resblock_plain"],
    choices=template_subgraph_names,
    help="template subgraph architecture: "
    + " | ".join(template_subgraph_names)
    + " (default: [resblock_plain])",
)
parser.add_argument(
    "-p",
    "--pys",
    metavar="PYTHON_FILES",
    nargs="*",
    type=str,
    help="python files of query subgraph definitions",
)
parser.add_argument(
    "-v", "--verbose", dest="verbose", action="store_true", help="verbose mode"
)
parser.add_argument(
    "--save_src_graph_path",
    metavar="SAVE_SRC_GRAPH_PATH",
    type=str,
    help="path to save the source graph with matched subgraph label(s)",
)


def setup_logger(name, logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a INFO log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)

    # create file handler which logs even DEBUG messages
    fh = None
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s"
        )
        fh.setFormatter(fh_formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    if logfile:
        logger.addHandler(fh)
    return logger


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
        if is_same_type:
            subgraph_list_matched.append(matched)
    return subgraph_list_matched


def search_subgraph(graph, subgraph, src_nodes_dict, **kwargs):
    verbose = kwargs.get("verbose", False)
    assert subgraph.get("src", {}) == {
        "input": ["input"],
        "output": ["input"],
        "opType": "None",
    }, 'The subgraph must contain an input node named "src"'
    query_graph = nx.DiGraph()
    for key in subgraph.keys():
        query_graph.add_node(key)

    query_graph = add_edges(query_graph, subgraph)
    # save_graph(query_graph, "query_graph.png")

    subgraph_list = search_basic(graph, query_graph)
    if verbose:
        logger.info(f"\tStructure match: {len(subgraph_list)}")

    subgraph_list_matched = search_type(subgraph_list, src_nodes_dict, subgraph)
    if verbose:
        logger.info(f"\tType match: {len(subgraph_list_matched)}")

    return subgraph_list_matched


def prepare_source_graph(node_list):
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

    return graph, network_nodes


def main():
    args = parser.parse_args()
    verbose = args.verbose
    if verbose:
        logger.info(vars(args))
    if not args.jsons and not args.templates and not args.pys:
        raise ValueError(
            "One of json files, template subgraph variables or python files must be specified!"
        )

    # model_path = "./resnet18-v2-7.onnx"
    # jsonize_mnist_onnx(model_path)
    with open(args.model, "r") as f:
        data = json.load(f)
    # Prepare source graph
    node_list = data["graph"]["node"]
    graph, network_nodes = prepare_source_graph(node_list)
    # save_graph(graph, "graph.png")

    # Collect query subgraphs
    # NOTE: currently labels are given automatically and incrementally
    query_dict = {}
    label_counter = 0
    if args.templates:
        templates = set(args.templates)
        for template_name in templates:
            label = f"label_{label_counter}"
            query_dict[label] = template_subgraphs.__dict__[template_name]
            label_counter += 1
    if args.jsons:
        for json_path in args.jsons:
            with open(json_path, "r") as f:
                tmp_data = json.load(f)
            assert isinstance(tmp_data, dict), "subgraph definition incorrect."
            label = f"label_{label_counter}"
            query_dict[label] = tmp_data
            label_counter += 1
    if args.pys:
        for py_file in set(args.pys):
            try:
                spec = importlib.util.spec_from_file_location("module.name", py_file)
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)
            except:
                raise ValueError("Python files load error.")
            for subgraph_name in foo.__all__:
                label = f"label_{label_counter}"
                query_dict[label] = foo.__dict__[subgraph_name]
                label_counter += 1

    subgraph_list_matched_dict = {}
    for label, subgraph in query_dict.items():
        if verbose:
            logger.info(f"Searching query label: {label} ...")
        subgraph_list_matched = search_subgraph(
            graph, subgraph, network_nodes, verbose=verbose
        )
        subgraph_list_matched_dict[label] = subgraph_list_matched

    for label, subgraph_list in subgraph_list_matched_dict.items():
        # append label to each matched keys
        for i, matched in enumerate(subgraph_list):
            for k in matched.keys():
                network_nodes[k]["s_label"].append(f"{label}_{i}")

    # pprint(network_nodes)
    if args.save_src_graph_path:
        with open(args.save_src_graph_path, "w") as f:
            json.dump(network_nodes, f)


if __name__ == "__main__":
    logger = setup_logger(__name__)
    main()

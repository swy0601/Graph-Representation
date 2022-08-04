import os.path
import json
from embedding_handle import handleJavaCode, codeEmbedding, one_hot_node_type, ControlEdgeHandle
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

import re
import numpy as np
import glob
import pandas as pd

data_number = 0
data_flow_number = 0
control_flow_number = 0


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.json")]


def write_pkl(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)


def ConvertToGraph(json_content):
    """
       Convert a specific json data from JavaParse to a basic Graph
    """
    graph_raw = json_content
    Vertice = []
    Vertice_info = []
    Edge_info = []
    Edge_index = [[], []]
    Edge_index2 = [[], []]

    def createGraph(graph, name, last):
        index = -1
        isExternal = True
        for key in graph.keys():
            if key == "!":
                isExternal = False
                Vertice.append(graph[key])
                Vertice_info.append([])
                index = len(Vertice) - 1

                if len(Vertice) != 1:
                    Edge_info.append(name)
                    Edge_index[0].append(last)
                    Edge_index[1].append(index)
                break

        if isExternal == False:
            for key in graph.keys():
                if isinstance(graph[key], list):
                    if len(graph[key]) == 0:
                        Vertice_info[index].append({key: graph[key]})
                    else:
                        list_temp = []
                        for item in graph[key]:
                            if item["!"] != "com.github.javaparser.ast.expr.NameExpr" \
                                    and item["!"] != 'com.github.javaparser.ast.Modifier' \
                                    and item["!"] != 'com.github.javaparser.ast.body.Parameter':
                                list_temp.append(len(Vertice_info))
                            createGraph(item, key, index)
                        for te in range(len(list_temp) - 1):
                            Edge_index2[0].append(list_temp[te])
                            Edge_index2[1].append(list_temp[te + 1])

                elif isinstance(graph[key], dict):
                    createGraph(graph[key], key, index)

                elif isinstance(graph[key], str) and key != "!":
                    Vertice_info[index].append({key: graph[key]})
        else:
            Vertice_info[index].append({name: graph})

    createGraph(graph_raw, "graph", 0)
    return {"node_type": Vertice, "node_list": Vertice_info, "edge_type": Edge_info, "edge_list": Edge_index,
            "control_list": Edge_index2}


def json_parse_to_graph(N_PATHS_AST, R_PATHS_AST, U_PATHS_AST):
    """
       Convert json file to Graph Representation
    """
    n_dataset_files = get_directory_files(N_PATHS_AST)
    r_dataset_files = get_directory_files(R_PATHS_AST)
    u_dataset_files = get_directory_files(U_PATHS_AST)

    graph_list = []
    target_list = []
    code_filename_list = []

    for json_file in r_dataset_files:
        with open(os.path.join(R_PATHS_AST, json_file)) as f:
            content = json.load(f)
            graph = ConvertToGraph(content)
            graph_list.append(graph)
            target_list.append(0)
            code_filename_list.append(os.path.join(R_PATHS_AST, json_file.replace(".json", ".java")))

    for json_file in n_dataset_files:
        with open(os.path.join(N_PATHS_AST, json_file)) as f:
            content = json.load(f)
            graph = ConvertToGraph(content)
            graph_list.append(graph)
            target_list.append(1)
            code_filename_list.append(os.path.join(N_PATHS_AST, json_file.replace(".json", ".java")))

    for json_file in u_dataset_files:
        with open(os.path.join(U_PATHS_AST, json_file)) as f:
            content = json.load(f)
            graph = ConvertToGraph(content)
            graph_list.append(graph)
            target_list.append(2)
            code_filename_list.append(os.path.join(U_PATHS_AST, json_file.replace(".json", ".java")))

    return graph_list, target_list, code_filename_list


def graph_to_input(graph, javeCode, target, tokenizer, model):
    """
       Convert Graph to Vector Data for train and test, adding extra Extra in this process
    """
    node_type = graph["node_type"]  # node type
    node_list = graph["node_list"]  # node data, contain the range
    edge_list = graph["edge_list"]  # index of edge

    auto_control_list = graph["control_list"]
    node_embedding_list = []
    node_one_hot_list = []

    print("==================", javeCode, "=============")
    global data_number
    global data_flow_number
    global control_flow_number
    data_number = data_number + 1
    print(data_number)

    other_node = []

    code_list_list = []
    code_range_list = []
    node_declaration_list = []
    node_assign_list = []
    for i in range(len(node_list)):
        code_range = node_list[i][0]["range"]
        nl_list, code_list = handleJavaCode(javeCode, code_range)
        node_embedding = codeEmbedding(nl_list, code_list, tokenizer, model)
        node_type_one_hot = one_hot_node_type(node_type[i])
        node_one_hot_list.append(node_type_one_hot)
        node_embedding_list.append(node_embedding)

        code_list_list.append([code_list, node_type[i]])
        code_range_list.append([code_range["beginLine"], code_range["endLine"]])

        if 'com.github.javaparser.ast.expr.AssignExpr' in node_type[i] \
                or 'com.github.javaparser.ast.expr.MethodCallExpr' in node_type[i] \
                or 'com.github.javaparser.ast.expr.BinaryExpr' in node_type[i] \
                or 'com.github.javaparser.ast.expr.UnaryExpr' in node_type[i]:

            node_assign_list.append(
                [i,
                 (code_range["beginLine"] - 1, code_range["endLine"] - 1,
                  ),
                 re.split(' |\.|\)|\(|\[|\]|\=', "".join(code_list)),
                 code_list])
        elif 'com.github.javaparser.ast.expr.VariableDeclarationExpr' in node_type[i] \
                or 'com.github.javaparser.ast.body.Parameter' in node_type[i]:
            node_declaration_list.append(
                [i,
                 (code_range["beginLine"] - 1, code_range["endLine"] - 1,
                  ),
                 re.split(' |\.|\)|\(|\[|\]|\=', "".join(code_list)),
                 code_list])
        else:
            other_node.append([i,
                               (code_range["beginLine"] - 1, code_range["endLine"] - 1,
                                ),
                               node_type[i],
                               re.split(' |\.|\)|\(|\[|\]|\=', "".join(code_list)),
                               code_list])

    # count the control edge number
    control_edge_list = ControlEdgeHandle(javeCode, auto_control_list)
    for control_edge in control_edge_list:
        edge_list[0].append(control_edge[0])
        edge_list[1].append(control_edge[1])

    # count the data edge number
    data_edge_list = DataEdgeHandle(node_declaration_list, node_assign_list)
    for data_edge in data_edge_list:
        edge_list[0].append(data_edge[0])
        edge_list[1].append(data_edge[1])
    #
    control_flow_number = control_flow_number + len(control_edge_list)
    data_flow_number = data_flow_number + len(data_edge_list)
    print(data_flow_number)
    print(control_flow_number)
    # print(code_list_list)
    return node_type, node_list, node_embedding_list, edge_list, target, node_one_hot_list,


def DataEdgeHandle(declaration_list, assign_list):
    """
       Handle Extra Data Edge, three ways tested in Ablation Study
    """
    data_flow_edge_list = []

    # TYPE 1
    for decl in declaration_list:
        data_flow = []
        flag = False
        for assign in assign_list:
            if decl[2][1] in assign[2]:
                flag = True
                data_flow.append(assign[0])
        if flag:
            data_flow.insert(0, decl[0])
            for j in range(len(data_flow) - 1):
                data_flow_edge_list.append([data_flow[j], data_flow[j + 1]])

    # TYPE 2
    # for decl in declaration_list:
    #     data_flow = []
    #     flag = False
    #     for assign in assign_list:
    #         if decl[2][1] in assign[2]:
    #             flag = True
    #             data_flow.append(assign[0])
    #     if flag:
    #         data_flow.insert(0, decl[0])
    #         for j in range(len(data_flow) - 1):
    #             data_flow_edge_list.append([data_flow[0], data_flow[j + 1]])

    # TYPE 3
    # for decl in declaration_list:
    #     data_flow = []
    #     flag = False
    #     for assign in assign_list:
    #         if decl[2][1] in assign[2]:
    #             flag = True
    #             data_flow.append(assign[0])
    #     if flag:
    #         data_flow.insert(0, decl[0])
    #         for j in range(len(data_flow) - 1):
    #             data_flow_edge_list.append([data_flow[j], data_flow[j + 1]])
    #             if [data_flow[0], data_flow[j + 1]] not in data_flow_edge_list:
    #                 data_flow_edge_list.append([data_flow[0], data_flow[j + 1]])

    return data_flow_edge_list


if __name__ == "__main__":
    N_PATHS_AST = "..\Data\\Neutral"
    R_PATHS_AST = "..\Data\\Readable"
    U_PATHS_AST = "..\Data\\Unreadable"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    graph_list, target_list, code_filename_list = json_parse_to_graph(N_PATHS_AST, R_PATHS_AST, U_PATHS_AST)

    graph_input = []
    target_input = []
    file_input = []
    for i in range(len(graph_list)):
        node_type, node_list, node_embedding_list, edge_list, target, node_one_hot_list = \
            graph_to_input(graph_list[i], code_filename_list[i], target_list[i], tokenizer, model)
        nodes_info = []

        for j in range(len(node_embedding_list)):
            node_embedding = np.array(node_embedding_list[j])
            node_embedding = np.mean(node_embedding_list[j], axis=0)
            a = node_one_hot_list[j]
            node_info = np.concatenate((node_embedding.tolist(), node_one_hot_list[j]), axis=0)
            nodes_info.append(node_info)
        x = torch.tensor(nodes_info)

        x_zero = torch.zeros(1000, 836).float()
        x_zero[:x.size(0), :] = x

        y = torch.tensor([target]).float()
        edge_index = torch.tensor(edge_list)
        graph_data = Data(x=x_zero, edge_index=edge_index, y=target)
        graph_input.append(graph_data)
        target_input.append(target)
        file_input.append(code_filename_list[i].split("\\")[-1])
    index_list = [i for i in range(0, len(graph_input))]

    pkl_data = {"file": file_input, "input": graph_input, "target": target_input}
    cpg_dataset = pd.DataFrame(pkl_data)

    # please change the name ("input_XXXXXX.pkl") if necessary
    # the "matrix" is not necessary here, it's for future studying
    write_pkl(cpg_dataset[["input", "target", "matrix"]], "", f"input_XXXXXX.pkl")
    print("Build pkl Successfully")

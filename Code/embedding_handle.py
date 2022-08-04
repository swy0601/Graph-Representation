import torch
import numpy as np


def handleJavaCode(filename, code_range):
    """
    Extract his code and his comments for each node,
    1. filename is the original file;
    2. code-range is the range of this node
    """
    with open(filename, 'r') as f:
        file = f.read()
        file_list = file.replace("\t", " ").split("\n")
        range_file_list = []

        beginLine = code_range["beginLine"] - 2
        beginColumn = code_range["beginColumn"]
        endLine = code_range["endLine"] - 2
        endColumn = code_range["endColumn"]

        if beginLine < 0 or endLine < 0:
            return [], []
        if beginLine == endLine:
            for i in range(0, len(file_list)):
                if i == beginLine:
                    range_file_list.append(file_list[i][beginColumn - 1:endColumn])
        else:
            # print(len(file_list))
            for i in range(0, len(file_list)):
                if i == beginLine:
                    range_file_list.append(file_list[i][beginColumn - 1:])
                elif i == endLine:
                    range_file_list.append(file_list[i][0: endColumn])
                elif i > beginLine and i < endLine:
                    range_file_list.append(file_list[i])
            # print("kkk")

        nl_list = []
        code_list = []

        for str in range_file_list:
            if str.find("//") != -1:
                nl_list.append(str)
            else:
                code_list.append(str)

        return nl_list, code_list


def codeEmbedding(nl_list, code_list, tokenizer, model):
    """
    CodeEmbedding the extracted data
    """
    # print("begin to embedding")

    code = ""
    nl = ""
    for str in code_list:
        code = code + str

    for str in nl_list:
        nl = nl + str

    code_tokens = tokenizer.tokenize(code)
    nl_tokens = tokenizer.tokenize(nl)
    token_list = []
    token_embeddings = []
    tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    token_list = cutToken(tokens, token_list)
    for token in token_list:
        token_id = tokenizer.convert_tokens_to_ids(token)
        context_embeddings = model(torch.tensor(token_id)[None, :])[0]
        token_embeddings.append(context_embeddings)

    torch_tensor = torch.cat(token_embeddings, dim=1)

    return torch_tensor.tolist()[0]


def cutToken(tokens, token_list):
    """
    Cut tokens which are too long
    """
    if len(tokens) > 500:
        token_list.append(tokens[0:500])
        tokens = tokens[500: len(tokens)]
        cutToken(tokens, token_list)
    else:
        token_list.append(tokens)
    return token_list


def one_hot_node_type(node_type):
    """
     Handle 68 kinds of nodes with One-Hot
    """
    node_type = node_type.replace("com.github.javaparser.ast.", "")

    hot_dict = {'ArrayCreationLevel': 0, 'CompilationUnit': 1, 'Modifier': 2, 'body.ClassOrInterfaceDeclaration': 3,
                'body.ConstructorDeclaration': 4, 'body.MethodDeclaration': 5, 'body.Parameter': 6,
                'body.VariableDeclarator': 7, 'comments.BlockComment': 8, 'comments.JavadocComment': 9,
                'comments.LineComment': 10, 'expr.ArrayAccessExpr': 11, 'expr.ArrayCreationExpr': 12,
                'expr.ArrayInitializerExpr': 13, 'expr.AssignExpr': 14, 'expr.BinaryExpr': 15,
                'expr.BooleanLiteralExpr': 16, 'expr.CastExpr': 17, 'expr.CharLiteralExpr': 18, 'expr.ClassExpr': 19,
                'expr.ConditionalExpr': 20, 'expr.DoubleLiteralExpr': 21, 'expr.EnclosedExpr': 22,
                'expr.FieldAccessExpr': 23, 'expr.InstanceOfExpr': 24, 'expr.IntegerLiteralExpr': 25,
                'expr.LongLiteralExpr': 26, 'expr.MarkerAnnotationExpr': 27, 'expr.MemberValuePair': 28,
                'expr.MethodCallExpr': 29, 'expr.Name': 30, 'expr.NameExpr': 31, 'expr.NormalAnnotationExpr': 32,
                'expr.NullLiteralExpr': 33, 'expr.ObjectCreationExpr': 34, 'expr.SimpleName': 35,
                'expr.SingleMemberAnnotationExpr': 36, 'expr.StringLiteralExpr': 37, 'expr.SuperExpr': 38,
                'expr.ThisExpr': 39, 'expr.UnaryExpr': 40, 'expr.VariableDeclarationExpr': 41,
                'stmt.AssertStmt': 42,
                'stmt.BlockStmt': 43, 'stmt.BreakStmt': 44, 'stmt.CatchClause': 45, 'stmt.ContinueStmt': 46,
                'stmt.DoStmt': 47, 'stmt.EmptyStmt': 48, 'stmt.ExplicitConstructorInvocationStmt': 49,
                'stmt.ExpressionStmt': 50, 'stmt.ForEachStmt': 51, 'stmt.ForStmt': 52, 'stmt.IfStmt': 53,
                'stmt.LabeledStmt': 54, 'stmt.LocalClassDeclarationStmt': 55, 'stmt.ReturnStmt': 56,
                'stmt.SwitchEntry': 57,
                'stmt.SwitchStmt': 58, 'stmt.ThrowStmt': 59, 'stmt.TryStmt': 60,
                'stmt.WhileStmt': 61,
                'type.ArrayType': 62,
                'type.ClassOrInterfaceType': 63, 'type.PrimitiveType': 64, 'type.TypeParameter': 65,
                'type.VoidType': 66,
                'type.WildcardType': 67}

    index = hot_dict[node_type]
    all_zero = np.zeros(len(hot_dict.keys()), dtype=int)
    node_type_one_hot = all_zero.copy()
    node_type_one_hot[index] = 1
    # print(node_type_one_hot)
    return list(node_type_one_hot)


def ControlEdgeHandle(filename, auto_edge_list):
    """
     Handle Extra Control Edges, partial added manually
    """
    our_edge_for_flow_un = []
    our_edge_for_flow_re = []
    our_edge_for_flow_nu = []
    print(filename)

    if filename == "Scalabrino0.java":
        our_edge_for_flow_un = [[]]
    elif filename == "Scalabrino1.java":
        our_edge_for_flow_re = [[10, 18], [18, 12]]

    # In order to keep the code clean,
    # More content of manually adding control flow has been removed

    extra_edge_list = []
    for i in range(len(auto_edge_list[0])):
        extra_edge_list.append([auto_edge_list[0][i], auto_edge_list[1][i]])

    for i in our_edge_for_flow_re:
        if i not in extra_edge_list:
            extra_edge_list.append(i)
    for i in our_edge_for_flow_nu:
        if i not in extra_edge_list:
            extra_edge_list.append(i)
    for i in our_edge_for_flow_un:
        if i not in extra_edge_list:
            extra_edge_list.append(i)

    return extra_edge_list

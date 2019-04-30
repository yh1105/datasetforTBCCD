import pycparser

def _pad_nobatch(children):
    child_len = max([len(c) for n in children for c in n])
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]
    return children

def _traverse_tree_noid(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)
        for item in current_node.children():
            queue.append(item[1])
            child_json = {
                "node": _name(item[1]),
                "children": []
            }
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes


def _traverse_tree_withid(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)
        global alltokenlist
        alltokenlist = []
        vistast(current_node)
        if alltokenlist != []:
            ss = alltokenlist[0]
            sss = ss.replace(' ', '')
            if sss.startswith('"') or sss.startswith("'"):
                pass
            else:
                child_json = {
                    "node": sss,
                    "children": []
                }

                current_node_json['children'].append(child_json)
        for item in current_node.children():
            if _name(item[1]).startswith('"') or _name(item[1]).startswith("'"):
                k = 1
            else:
                queue.append(item[1])
                child_json = {
                    "node": _name(item[1]),
                    "children": []
                }
                current_node_json['children'].append(child_json)
                queue_json.append(child_json)
    return root_json, num_nodes


def _name(node):
    return type(node).__name__


def vistast(node):
    nvlist = [(n, getattr(node, n)) for n in node.attr_names]
    if nvlist != []:
        if hasattr(node, 'op'):
            word = node.op
        elif hasattr(node, 'declname'):
            word = node.declname
            if word is None:
                word = node.__class__.__name__
        elif isinstance(node, pycparser.c_ast.IdentifierType):
            word = node.names[0]
        elif isinstance(node, pycparser.c_ast.Constant):
            word = node.value
        elif isinstance(node, pycparser.c_ast.ID):
            word = node.name
        else:
            word = nvlist[0][1]
            word = str(word)
        alltokenlist.append(word)


def _traverse_tree_noast(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)
        global alltokenlist
        alltokenlist = []
        vistast(current_node)
        if alltokenlist != []:
            ss = alltokenlist[0]
            sss = ss.replace(' ', '')
            if sss.startswith('"') or sss.startswith("'"):
                k = 1
            else:
                child_json = {
                    "node": sss,
                    "children": []
                }
                current_node_json['children'].append(child_json)
        for item in current_node.children():
            if _name(item[1]).startswith('"') or _name(item[1]).startswith("'"):
                pass
            else:
                queue.append(item[1])
                child_json = {
                    "node": "AstNode",
                    "children": []
                }
                current_node_json['children'].append(child_json)
                queue_json.append(child_json)
    return root_json, num_nodes


def getData_nofinetune(l, dictt, embeddings):
    nodes11 = []
    children11 = []
    nodes22 = []
    children22 = []
    label = l[2]
    queue1 = [(dictt[l[0]], -1)]
    while queue1:
        node1, parent_ind1 = queue1.pop(0)
        node_ind1 = len(nodes11)
        queue1.extend([(child, node_ind1) for child in node1['children']])
        children11.append([])
        if parent_ind1 > -1:
            children11[parent_ind1].append(node_ind1)
        nodes11.append(embeddings[node1['node']])
    queue2 = [(dictt[l[1]], -1)]
    while queue2:
        node2, parent_ind2 = queue2.pop(0)
        node_ind2 = len(nodes22)
        queue2.extend([(child, node_ind2) for child in node2['children']])
        children22.append([])
        if parent_ind2 > -1:
            children22[parent_ind2].append(node_ind2)
        nodes22.append(embeddings[node2['node']])
    children111 = []
    children222 = []
    children111.append(children11)
    children222.append(children22)
    children1 = _pad_nobatch(children111)
    children2 = _pad_nobatch(children222)
    return [nodes11], children1, [nodes22], children2, label


def getData_finetune(l, dictt, embeddings):
    nodes11 = []
    children11 = []
    nodes22 = []
    children22 = []
    label = l[2]
    queue1 = [(dictt[l[0]], -1)]
    while queue1:
        node1, parent_ind1 = queue1.pop(0)
        node_ind1 = len(nodes11)
        queue1.extend([(child, node_ind1) for child in node1['children']])
        children11.append([])
        if parent_ind1 > -1:
            children11[parent_ind1].append(node_ind1)
        nodes11.append(embeddings[node1['node']])
    queue2 = [(dictt[l[1]], -1)]
    while queue2:
        node2, parent_ind2 = queue2.pop(0)
        node_ind2 = len(nodes22)
        queue2.extend([(child, node_ind2) for child in node2['children']])
        children22.append([])
        if parent_ind2 > -1:
            children22[parent_ind2].append(node_ind2)
        nodes22.append(embeddings[node2['node']])
    children111 = []
    children222 = []
    batch_labels = []
    children111.append(children11)
    children222.append(children22)
    children1 = _pad_nobatch(children111)
    children2 = _pad_nobatch(children222)
    batch_labels.append(label)
    return nodes11, children1, nodes22, children2, batch_labels

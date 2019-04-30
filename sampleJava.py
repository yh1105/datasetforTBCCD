import javalang
from javalang.ast import Node
def _name(node):
    return type(node).__name__


def dfsSearch_withid(children):
    if not isinstance(children, (str, Node, list, tuple)):
        return
    if isinstance(children, (str, Node)):
        if str(children) == '':
            return
        if str(children).startswith('"'):
            return
        if str(children).startswith("'"):
            return
        if str(children).startswith("/*"):
            return
        # ss = str(children)
        global num_nodes
        num_nodes += 1
        listt1.append(children)
        return
    for child in children:
        if isinstance(child, (str, Node, list, tuple)):
            dfsSearch_withid(child)


def _traverse_treewithid(root):
    global num_nodes
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        global listt1
        listt1 = []
        dfsSearch_withid(current_node.children)
        children = listt1
        for child in children:
            child_json = {
                "node": str(child),
                "children": []
            }
            current_node_json['children'].append(child_json)
            if isinstance(child, (Node)):
                queue_json.append(child_json)
                queue.append(child)
    return root_json, num_nodes

def _pad_nobatch(children):
    child_len = max([len(c) for n in children for c in n])
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]
    return children






def dfsSearch_noid(children):
    if not isinstance(children, (Node, list, tuple)):
        return
    if isinstance(children, Node):
        global num_nodes
        num_nodes+=1
        listt1.append(children)
        return
    for child in children:
        if isinstance(child, (Node, list, tuple)):
            dfsSearch_noid(child)
def _traverse_tree_noid(root):
    global num_nodes
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        global listt1
        listt1=[]
        dfsSearch_noid(current_node.children)
        children = listt1
        for child in children:
            child_json = {
                "node": str(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)
            queue.append(child)
    return root_json, num_nodes
def _traverse_tree_noast(root):
    global num_nodes
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)

        global listt1
        listt1 = []
        dfsSearch_withid(current_node.children)
        children = listt1
        for child in children:
            if isinstance(child, (Node)):
                child_json = {
                    "node": "AstNode",
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)
                queue.append(child)
            else:
                child_json = {
                    "node": str(child),
                    "children": []
                }

                current_node_json['children'].append(child_json)
    return root_json, num_nodes
def getData_nofinetune(l,dictt,embeddings):
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
    return [nodes11],children1,[nodes22],children2,label


def getData_finetune(l,dictt,embeddings):
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
    return nodes11,children1,nodes22,children2,batch_labels
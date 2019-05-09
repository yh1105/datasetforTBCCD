import pycparser
def _traverse_tree(root):
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
            # if sss.startswith('"') or sss.startswith("'"):
            #     k=1
            # else:
            child_json = {
                "node": sss,
                "children": []
            }
            current_node_json['children'].append(child_json)
        for item in current_node.children():
            # if _name(item[1]).startswith('"') or _name(item[1]).startswith("'"):
            #     k=1
            # else:
            queue.append(item[1])
            child_json = {
                "node": "AstNode",
                "children": []
            }
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)
    return root_json, num_nodes

def _name(node):
    return type(node).__name__
def vistast(node):
    nvlist=[(n,getattr(node,n)) for n in node.attr_names]
    # print("nvlists:")
    # print(nvlist)
    # print("nvliste:")
    if nvlist!=[]:
        if hasattr(node,'op'):
            word=node.op
            #print("word_op:" + word)
        elif hasattr(node,'declname'):
            word=node.declname
            #print("word_declname:" + word)
            if word is None:
                word=node.__class__.__name__
        elif isinstance(node,pycparser.c_ast.IdentifierType):
            word=node.names[0]
            #print("word_IdentifierType:" + word)
        elif isinstance(node,pycparser.c_ast.Constant):
            word=node.value
            #print("word_Constant:" + word)
        elif isinstance(node,pycparser.c_ast.ID):
            word=node.name
            #print("word_ID:" + word)
        else:
            word=nvlist[0][1]
            word=str(word)
            #print("word_word:" + word)
        alltokenlist.append(word)
numnodes=0
def dfsDict(root):
    global listtfinal
    listtfinal.append(str(root['node']))
    global numnodes
    numnodes+=1
    if len(root['children']):
        pass
    else:
        return
    for dictt in root['children']:
        dfsDict(dictt)
f=open("flistPOJ.txt",'r')
ff=open("sentencePOJnoast.txt",'w')
line=f.readline().rstrip("\t")
li=line.split("\t")
# print(len(li))
# print(str(li[7500]))
for l in li:
    tree = pycparser.parse_file(l)
    listtfinal = []
    numnodes = 0
    sample,num_nodes=_traverse_tree(tree)
    dfsDict(sample)
    for lis in listtfinal:
        ff.write(lis)
        ff.write(" ")
    ff.write("\n")
    #print(listtfinal)
ff.close()

from collections import deque
import numpy as np
from data.reorder.tree import TreeNode
from data.reorder.kolo import KOLO

INTERVAL_NODE_START_ID = 1000

def getOrderedHierarchy(confusion):
    CM = confusion['matrix']
    names = confusion['names']
    hierarchy = confusion['hierarchy']

    SM = getSimilarityMatrix(CM)
    tree = hierarchyToTree(hierarchy)
    name2id = {name: id for id, name in enumerate(names)}
    setIds(tree, name2id)
    setLeaves(tree)

    kolo = KOLO(tree, SM)
    opt_tree, _ = kolo.getOrdered()
    opt_hierarchy = treeToHierarchy(opt_tree)
    return opt_hierarchy

def treeToHierarchy(tree):
    hierarchy = []
    def dfs(node):
        if len(node.children)==0: return node.name
        item = {}
        item['name'] = node.name
        item['children'] = [dfs(c) for c in node.children]
        return item
    hierarchy = [dfs(c) for c in tree.children]
    return hierarchy

def hierarchyToTree(hierarchy):
    def buildTree(item):
        if isinstance(item, str): return TreeNode(name=item)
        node = TreeNode(name=item['name'])
        node.children = [buildTree(child) for child in item['children']]
        return node
    tree = buildTree({'name': 'root', 'children': hierarchy})
    return tree

def setIds(tree, name2id):
    """Set id of leaves according to 'name2id'.
    """
    global INTERVAL_NODE_START_ID
    start_id = INTERVAL_NODE_START_ID
    que = deque([tree])
    while len(que) > 0:
        p = que.popleft()
        if p.isLeaf():
            p.id = name2id[p.name]
        else:
            p.id = start_id
            start_id += 1
        for c in p.children:
            que.append(c)

def setLeaves(node):
    if node.isLeaf():
        node.leaves = [node]
        return
    node.leaves = []
    for c in node.children:
        setLeaves(c)
        node.leaves.extend(c.leaves)
 
def getSimilarityMatrix(CM):
    """return similarity / distance matrix of confusion matrix CM, currently distance
    """
    lam = 1
    eps = 0.1
    CM = np.array(CM)
    n = len(CM)
    for i in range(n): CM[i][i] = 0
    SM = [[100 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            row_i, row_j = CM[i], CM[j]
            row_ij, row_ji = row_i[j], row_j[i]
            item_1 = lam / (row_ij + row_ji + eps)

            row_i[j] = row_j[i] = 0
            item_2 = np.linalg.norm(row_i - row_j)
            row_i[j], row_j[i] = row_ij, row_ji

            SM[i][j] = item_1 + item_2
            SM[j][i] = SM[i][j]
    return SM




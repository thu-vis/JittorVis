from data.reorder.tree import TreeNode
import itertools
import math
import copy

MAX = 1e9
DUMMY_NODE_START_ID = 2000

def getCombinations(lst, siz):
    out_combos = []
    list_set = set(lst)
    for combination in itertools.combinations(lst, siz):
        left = set(combination)
        right = list_set - left
        parts = [tuple(left), tuple(right)]
        if list(reversed(parts)) not in out_combos:
            out_combos.append(parts)
    return out_combos


def getPermutations(lst, siz):
    def validate(parts, siz):
        for p in parts:
            for i in range(siz - 1):
                if len(p) > i + 1:
                    if p[i] > p[i + 1]:
                        return False
        return True
    outs = []
    for c in itertools.permutations(lst):
        parts = [c[:siz], c[siz:]]
        if validate(parts, siz):
            if list(reversed(parts)) not in outs:
                outs.append(parts)
    return outs

class KOLO:
    """Class for solving k-ary tree reordering.

        tree(TreeNode) 

        similarityMatrix(2D list)
    """
    def __init__(self, tree, matrix) -> None:
        self.tree = tree
        self.S = matrix
        self.M = {} 
        self.I = {}
        self.dummy_node_idx = DUMMY_NODE_START_ID
        self.dummy_nodes = []
        self.mem_M = {}
        self.mem_I = {}

    def getOrdered(self):
        """return ordered tree 
        """
        import time
        start = time.clock()
        self.M, self.I = self.opt(self.tree)
        end = time.clock()
        print('time for ordering', end - start)
        min_score = MAX
        min_k = -1
        for k in self.M.keys():
            if self.M[k] < min_score:
                min_score = self.M[k]
                min_k = k
        
        # similar to original order
        ord_ids = self.I[min_k]
        ori_ids = self.tree.preOrder()
        diff = [abs(ord_ids[i] - ori_ids[i]) for i in range(len(ord_ids))]
        diff_rev = [abs(ord_ids[::-1][i] - ori_ids[i]) for i in range(len(ord_ids))]
        if diff_rev < diff: ord_ids = ord_ids[::-1]

        opt_tree = copy.deepcopy(self.tree)
        def reorderTree(node):
            nonlocal ord_ids
            if len(node.children)==0:
                return
            node.children.sort(key=lambda c: ord_ids.index(c.leaves[0].id))
            for c in node.children: reorderTree(c)
        reorderTree(opt_tree)
        return opt_tree, ord_ids

    def opt(self, v:TreeNode):
        if v.getLeavesId() in self.mem_M:
            return self.mem_M[v.getLeavesId()], self.mem_I[v.getLeavesId()]

        if v.isLeaf():
            M = {}
            M[v.id, v.id] = 0
            self.mem_M[tuple([v.id])] = M
            I = {}
            I[v.id, v.id] = [v.id]
            self.mem_I[tuple([v.id])] = I 
            # print(tuple([v.id]))
            return M, I

        v_children = v.children
        children_number = len(v_children)
        v_left, v_right = None, None

        possible_permutations = getCombinations(list(range(children_number)), math.ceil(children_number / 2))

        M_list = []
        I_list = []

        for left_indexes, right_indexes in possible_permutations:
            tmp_M = {}
            M_list.append(tmp_M)

            tmp_I = {}
            I_list.append(tmp_I)

            left_nodes = [v_children[idx] for idx in left_indexes]
            right_nodes = [v_children[idx] for idx in right_indexes]
        
            if len(left_nodes) >= 2:
                dummy_left = TreeNode(self.dummy_node_idx)
                self.dummy_nodes.append(dummy_left)
                self.dummy_node_idx += 1
                dummy_left.children = left_nodes
                dummy_left.leaves = []
                for n in left_nodes: dummy_left.leaves.extend(n.leaves)
                v_left = dummy_left
            else:
                assert len(left_nodes)==1
                v_left = left_nodes[0]

            if len(right_nodes) >= 2:
                dummy_right = TreeNode(self.dummy_node_idx)
                self.dummy_nodes.append(dummy_right)
                self.dummy_node_idx += 1
                dummy_right.children = right_nodes
                dummy_right.leaves = []
                for n in right_nodes: dummy_right.leaves.extend(n.leaves)
                v_right = dummy_right
            else:
                assert len(right_nodes)==1
                v_right = right_nodes[0]

            L_M, L_I = self.opt(v_left)
            R_M, R_I = self.opt(v_right)

            L = v_left.leaves
            R = v_right.leaves

            tmp = {}
            tmp_h = {}
            for i in L:
                i_idx = -1
                for idx, c in enumerate(v_left.children):
                    if i in c.leaves:
                        i_idx = idx
                        break
                for l in R:
                    tmp_il = MAX
                    MAXh = None
                    if i_idx == -1:
                        h = i
                        if L_M[i.id, h.id] + self.S[h.id][l.id] < tmp_il:
                            tmp_il = L_M[i.id, h.id] + self.S[h.id][l.id]
                            MAXh = h
                    else:
                        for h_idx in range(len(v_left.children)):
                            if h_idx == i_idx: continue
                            for h in v_left.children[h_idx].leaves:
                                if L_M[i.id, h.id] + self.S[h.id][l.id] < tmp_il:
                                    tmp_il = L_M[i.id, h.id] + self.S[h.id][l.id]
                                    MAXh = h
                    tmp[i.id, l.id] = tmp_il   
                    tmp_h[i.id, l.id] = MAXh.id

            for i in L:
                for j in R:
                    j_idx = -1
                    for idx, c in enumerate(v_right.children):
                        if j in c.leaves:
                            j_idx = idx
                            break
                    tmp_ij = MAX
                    MAXl = None
                    if j_idx == -1:
                        l = j
                        if tmp[i.id, l.id] + R_M[l.id, j.id] < tmp_ij:
                            tmp_ij = tmp[i.id, l.id] + R_M[l.id, j.id]
                            MAXl = l
                    else:
                        for l_idx in range(len(v_right.children)):
                            if l_idx == j_idx: continue
                            for l in v_right.children[l_idx].leaves:
                                if tmp[i.id, l.id] + R_M[l.id, j.id] < tmp_ij:
                                    tmp_ij = tmp[i.id, l.id] + R_M[l.id, j.id]
                                    MAXl = l
                    tmp_M[i.id, j.id] = tmp_M[j.id, i.id] = tmp_ij
                    tmp_I[i.id, j.id] = [] 
                    tmp_I[i.id, j.id].extend(L_I[i.id, tmp_h[i.id, MAXl.id]])
                    tmp_I[i.id, j.id].extend(R_I[MAXl.id, j.id])
                    tmp_I[j.id, i.id] = []
                    tmp_I[j.id, i.id].extend(R_I[j.id, MAXl.id])
                    tmp_I[j.id, i.id].extend(L_I[tmp_h[i.id, MAXl.id], i.id])
        
        M = {}
        I = {}
        for i, tmp_M in enumerate(M_list):
            tmp_I = I_list[i]
            for k in tmp_M.keys():
                if k not in M:
                    M[k] = tmp_M[k]
                    I[k] = tmp_I[k]
                else:
                    if tmp_M[k] < M[k]:
                        M[k] = tmp_M[k]
                        I[k] = tmp_I[k]

        # print(tuple(sorted(v_left.getLeavesId() + v_right.getLeavesId())))
        self.mem_M[tuple(sorted(v_left.getLeavesId() + v_right.getLeavesId()))] = M
        self.mem_I[tuple(sorted(v_left.getLeavesId() + v_right.getLeavesId()))] = I
        return M, I



def test():

    def setLeaves(node):
        if len(node.leaves) > 0:
            return
        if node.isLeaf():
            node.leaves = [node]
            return
        node.leaves = []
        for c in node.children:
            setLeaves(c)
            node.leaves.extend(c.leaves)
    
    test_m = [[100 for _ in range(11)] for _ in range(11)]
    for i in range(10):
        test_m[i][i+1] = 10
        test_m[i+1][i] = 10

    for i in range(10):
        for j in range(i, 10):
            test_m[j][i] = test_m[i][j]

    tree = TreeNode(0, name='root')
    n1 = TreeNode(1)
    n2 = TreeNode(2)
    n3 = TreeNode(3)
    n4 = TreeNode(4)
    n5 = TreeNode(5)
    n6 = TreeNode(6)
    n7 = TreeNode(7)
    n8 = TreeNode(8)
    n9 = TreeNode(9)
    n10 = TreeNode(10)
    tree.children = [n3, n2, n1]
    n1.children = [n4, n6, n5]
    n2.children = [n10, n8, n7, n9]
    # n4.children = [n7, n8]
    # n5.children = [n9, n10]
    
    setLeaves(tree)
    kolo = KOLO(tree, test_m)
    opt,opt2 = kolo.getOrdered()
    print(opt2)
import pickle
import numpy as np
from datetime import datetime
from re import sub, search
import os
from subprocess import Popen, PIPE
import heapq
from scripts.karger import get_best_partition
from scripts import config
import math


def normalize(arr):
    if sum(arr) == 0:
        return np.zeros_like(arr)
    return arr / sum(arr)

def create_feature_map_image(data_id, shape, var_node_id):
    #print('create_feature_map_image', data_id, shape, var_node_id)
    from PIL import Image
    data = pickle.load(open(config.DATA_PATH, 'rb'))['node_data']
    img_arr = [float(x) for x in data[var_node_id]['attrs']['data'].split(',')]
    shape = [int(x) for x in shape[1:-2].split(',')]
    while 1 in shape:
        sum_axis = shape.index(1)
        del shape[sum_axis]

    img_arr = np.array(img_arr).reshape(shape)
    img_arr = np.minimum(np.maximum(img_arr, -1), 1)

    if len(shape) > 2 and shape[-1] == shape[-2] and shape[-1] >= 10:
        while len(shape) > 2:
            sum_axis = 0
            img_arr = img_arr.sum(axis=sum_axis) / shape[0]
            del shape[sum_axis]
    else:
        while len(shape) >= 2:
            min_shape = min(shape)
            max_shape = max(shape)
            if min_shape == max_shape:
                break
            sum_axis = shape.index(min_shape)
            img_arr = img_arr.sum(axis=sum_axis) / min_shape
            del shape[sum_axis]
    if len(shape) == 1:
        width = math.ceil(math.sqrt(shape[0]))
        img_arr = np.concatenate((img_arr, np.zeros(width * width - shape[0])))
        if width < 32:
            ratio = 128 // width
            img_arr = np.concatenate([[i for j in range(ratio)] for i in img_arr])
            img_arr = np.concatenate([np.concatenate([img_arr[i: i + width * ratio] for j in range(ratio)]) for i in range(0, len(img_arr), ratio * width)])
            width = width * ratio
        shape = [width, width]
        #img_arr = np.array([np.array(img_arr, copy=True) for i in range(4)])
    #img_arr = img_arr / 2 + 0.5
    shape.append(3)
    img_arr = transform_to_color(img_arr).reshape(shape)
    #print('reshape to', shape)

    img = Image.fromarray(img_arr).convert('RGB')
    img_path = os.path.join('data', 'feature_maps', 'feature_map_{}.png'.format(str(data_id)))
    img.save(img_path)
    return img_path

def transform_to_color(img_arr):
    color1 = np.array([103, 0, 31])
    color2 = np.array([5, 48, 97])
    white = np.array([255, 255, 255])
    return np.array([((-x) * color1 + (1 + x) * white) if x < 0 else (x * color2 + (1 - x) * white) for x in img_arr.reshape(-1)], dtype=np.uint8)


def get_network_and_op_groups_from_data(data):
    print('get_network_and_op_groups_from_data')
    node_data = data['node_data']
    nodes = []
    edges = []
    root_nodes = []
    # the node tree, include var node
    # edges with stack
    level = 0
    extern_node_name_id_to_index = {}
    node_id_to_index = {}

    execute_op_info = data['execute_op_info']
    schedule = []
    op_groups = []
    gpu_id_to_index = {}
    gpu_ids = []
    # for op_id in execute_op_info:
    #     gpu_id = np.random.randint(0, 5)
    for gpu_id in range(3):
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    gpu_ids = sorted(gpu_ids)
    for gpu_id in gpu_ids:
        gpu_id_to_index[gpu_id] = len(schedule)
        schedule.append({
            'gpu_id': gpu_id,
            'op_groups': []
        })

    print('add all leaf node from leaf data')
    for node_id in node_data:
        node = node_data[node_id]
        if node['attrs']['is_var'] == '1':
            continue
        node_id_to_index[node_id] = len(nodes)
        nodes.append({
            'index': len(nodes),
            'id': node_id,
            'title': node['attrs']['name'],
            'level': level,
            'children': [],
            'parent': -1,
            'stacks': node['stacks'],
            'next': list(set(node['outputs'])),
            'pre': [],
            'type': 'leaf_op_node' if node['attrs']['is_var'] == '0' else 'leaf_var_node',
            'code_path': '',
            'line_num': -1,
            'is_var': node['attrs']['is_var'] == '1',
            'attrs': node['attrs'],
            'gpu_distribution': np.zeros(len(schedule)),
            'time_cost': 0,
            'op_groups': []
        })

    print('add all edges connect all leaf nodes')
    for node in nodes:
        #if len(node['next']) > 1:
        #    print("node['next']", len(node['next']))
        #    print(node)
        for i, var_node_id in enumerate(node['next']):
            var_node = node_data[var_node_id]
            node['attrs']['shape'] = var_node['attrs']['shape']
            node['attrs']['var_node_id'] = var_node_id
            node['attrs']['has_feature'] = 'data' in var_node['attrs']
            for j, node_id in enumerate(list(set(var_node['outputs']))):
                edges.append({
                    'index': len(edges),
                    'start_stack': [node['index']],
                    'end_stack': [node_id_to_index[node_id]],
                    'start_index': 0,
                    'end_index': 0,
                    'start': node['index'],
                    'end': node_id_to_index[node_id]
                })
        node['next'] = []

    for edge_index, edge in enumerate(edges):
        for node_index in edge['start_stack']:
            nodes[node_index]['next'].append(edge_index)
        for node_index in edge['end_stack']:
            nodes[node_index]['pre'].append(edge_index)

    print('process the tree of nodes')
    for i in range(len(nodes)):
        child_index = i
        name_id = []
        for stack in nodes[i]['stacks']:
            if stack['name'] == 'None':
                stack['name'] = 'model'
            name_id.append(stack['name'])
            stack['name_id'] = '$'.join(name_id)
        for j in range(len(nodes[i]['stacks']) - 1, -1, -1):
            extern_node = nodes[i]['stacks'][j]
            if extern_node['name_id'] not in extern_node_name_id_to_index:
                extern_node_index = len(nodes)
                extern_node_name_id_to_index[extern_node['name_id']] = extern_node_index
                nodes.append({
                    'index': extern_node_index,
                    'id': -1,
                    'title': extern_node['name'],
                    'level': nodes[child_index]['level'] + 1,
                    'children': [],
                    'parent': -1,
                    'stacks': [],
                    'next': [],
                    'pre': [],
                    'type': extern_node['type'],
                    'code_path': extern_node['file_path'],
                    'line_num': extern_node['lineno'],
                    'is_var': False,
                    'attrs': {},
                    'gpu_distribution': np.zeros(len(schedule)),
                    'time_cost': 0,
                    'op_groups': []
                })
            else:
                extern_node_index = extern_node_name_id_to_index[extern_node['name_id']]

            nodes[child_index]['parent'] = extern_node_index
            if child_index not in nodes[extern_node_index]['children']:
                nodes[extern_node_index]['children'].append(child_index)
            nodes[extern_node_index]['level'] = max(nodes[extern_node_index]['level'], nodes[child_index]['level'] + 1)
            child_index = extern_node_index

    for node in nodes:
        if node['is_var'] and node['parent'] == -1:
            node['parent'] = -2
            pre = node['pre']
            next = node['next']
            for pre_id in pre:
                pre_node_stack = [edges[pre_id]['start_stack'][0]]
                while nodes[pre_node_stack[-1]]['parent'] != -1:
                    pre_node_stack.append(nodes[pre_node_stack[-1]]['parent'])
                for next_id in next:
                    next_node_stack = [edges[next_id]['end_stack'][0]]
                    while nodes[next_node_stack[-1]]['parent'] != -1:
                        next_node_stack.append(nodes[next_node_stack[-1]]['parent'])

                    index = -1
                    while pre_node_stack[index] == next_node_stack[index]:
                        index -= 1
                    if index != -1:
                        node['parent'] = pre_node_stack[index + 1]
                        nodes[node['parent']]['children'].append(node['index'])
                        if nodes[node['parent']]['level'] < 1:
                            nodes[node['parent']]['level'] = 1
                        break
                if node['parent'] != -2:
                    break
            if node['parent'] == -2:
                node['parent'] = -1

    for node in nodes:
        node['next'] = []
        node['pre'] = []


    print('process the stack of all edges')
    for edge_index, edge in enumerate(edges):
        while nodes[edge['start_stack'][-1]]['parent'] != -1:
            edge['start_stack'].append(nodes[edge['start_stack'][-1]]['parent'])

        while nodes[edge['end_stack'][-1]]['parent'] != -1:
            edge['end_stack'].append(nodes[edge['end_stack'][-1]]['parent'])
        index = -1
        while edge['start_stack'][index] == edge['end_stack'][index]:
            index -= 1
        if index != -1:
            edge['start_stack'] = edge['start_stack'][:index + 1]
            edge['end_stack'] = edge['end_stack'][:index + 1]
        edge['start_stack'].reverse()
        edge['end_stack'].reverse()
        edge['start'] = edge['start_stack'][0]
        edge['end'] = edge['end_stack'][0]

        for node_index in edge['start_stack']:
            nodes[node_index]['next'].append(edge_index)
        for node_index in edge['end_stack']:
            nodes[node_index]['pre'].append(edge_index)


    for node in nodes:
        del node['stacks']
        if node['parent'] == -1:
            root_nodes.append(node['index'])

    # add virtual extern_nodes for node whose number of children is greater than 30
    # nodes, edges = split_oversize_extern_nodes(nodes, edges, [x for x in root_nodes])

    # compute the level of nodes
    for node in nodes:
        node['next'] = []
        node['pre'] = []
        if node['level'] != 0:
            node['level'] = 1

    max_node_level = 0
    for node in nodes:
        if node['level'] == 0:
            temp_node = node
            while temp_node['parent'] != -1:
                nodes[temp_node['parent']]['level'] = max(nodes[temp_node['parent']]['level'], temp_node['level'] + 1)
                temp_node = nodes[temp_node['parent']]
            max_node_level = max(max_node_level, temp_node['level'])


    for edge in edges:
        edge['start_stack'] = edge['start_stack'][-1:]
        edge['end_stack'] = edge['end_stack'][-1:]


    # process the stack of all edges
    for edge_index, edge in enumerate(edges):
        while nodes[edge['start_stack'][-1]]['parent'] != -1:
            edge['start_stack'].append(nodes[edge['start_stack'][-1]]['parent'])

        while nodes[edge['end_stack'][-1]]['parent'] != -1:
            edge['end_stack'].append(nodes[edge['end_stack'][-1]]['parent'])
        index = -1
        while edge['start_stack'][index] == edge['end_stack'][index]:
            index -= 1
        if index != -1:
            edge['start_stack'] = edge['start_stack'][:index + 1]
            edge['end_stack'] = edge['end_stack'][:index + 1]
        edge['start_stack'].reverse()
        edge['end_stack'].reverse()
        edge['start'] = edge['start_stack'][0]
        edge['end'] = edge['end_stack'][0]

        for node_index in edge['start_stack']:
            nodes[node_index]['next'].append(edge_index)
        for node_index in edge['end_stack']:
            nodes[node_index]['pre'].append(edge_index)


    # direction of nodes and edge
    # direction = False mean it is reverse node or direction
    # reversed = True mean it has been reversed for layout


    print('process reverse nodes and edges')
    # 确定反向节点和边
    for node in nodes:
        node['direction'] = True

    for root_node in root_nodes:
        if nodes[root_node]['title'] == 'grad':
            queue = [root_node]
            while len(queue) > 0:
                nodes[queue[0]]['direction'] = False
                queue.extend(nodes[queue[0]]['children'])
                del queue[0]

    # 去除重复边

    remove_duplicate_edges(nodes, edges)

    # 反转 终点为反向节点的边
    for edge in edges:
        start_node = nodes[edge['start']]
        end_node = nodes[edge['end']]
        edge['direction'] = start_node['direction'] and end_node['direction']
        if not end_node['direction']:
            reverse_edge_direction(edge, nodes)
        else:
            edge['reversed'] = False

    max_var_node_group_size = get_max_var_node_group_size(nodes, edges)

    process_data_of_extern_node(root_nodes, nodes, edges)

    network = {
        'nodes': nodes,
        'edges': edges,
        'root': root_nodes,
        'gpu_ids': gpu_ids,
        'max_var_node_group_size': max_var_node_group_size,
        'max_node_level': max_node_level
    }

    # compute dependency among leaf_nodes
    node_dependency = compute_dependency_of_leaf_nodes(nodes, edges)

    for op_id in execute_op_info:
        op = execute_op_info[op_id]
        gpu_id = np.random.randint(0, 3)
        op_index = len(op_groups)
        schedule_index = gpu_id_to_index[gpu_id]
        _nodes = []
        for node_id in op['fused_ops']:
            if node_id in node_id_to_index:
                _nodes.append(node_id_to_index[node_id])
        op_group = {
            'index': op_index,
            'id': op_id,
            'nodes': _nodes,
            'file_path': op['file_path'],
            'jit_key': op['jit_key'],
            'time_cost': 1,
            'gpu_id': gpu_id,
            'schedule_index': schedule_index,
            'attrs': op['attrs']
        }
        for node_index in op_group['nodes']:
            nodes[node_index]['op_groups'].append(op_index)
            nodes[node_index]['time_cost'] += 1 / len(op_group['nodes'])
            nodes[node_index]['gpu_distribution'][gpu_id_to_index[gpu_id]] += 1
        compute_dependency_of_op_group(op_group, schedule, schedule_index, network, op_groups, node_dependency)
        schedule[schedule_index]['op_groups'].append(op_index)
        op_groups.append(op_group)

    for node in nodes:
        node['gpu_distribution'] = normalize(node['gpu_distribution'])

    for node in nodes:
        if node['parent'] != -1:
            nodes[node['parent']]['gpu_distribution'] += node['gpu_distribution'] * node['time_cost']
            nodes[node['parent']]['time_cost'] += node['time_cost']
            nodes[node['parent']]['op_groups'] += node['op_groups']

    for node in nodes:
        node['gpu_distribution'] = normalize(node['gpu_distribution']).tolist()
        node['op_groups'] = list(set(node['op_groups']))

    # try keep op_groups with same jit_key having same jit_type
    op_group_label = {}
    count = 0
    jit_key_items = []

    for gpu_group in schedule:
        for i, op_group_index in enumerate(gpu_group['op_groups']):
            op_group = op_groups[op_group_index]
            jit_key = op_group['jit_key']
            if jit_key not in op_group_label:
                op_group_label[jit_key] = count
                jit_key_items.append({
                    'jit_key': jit_key,
                    'neighbors': []
                })
                count += 1
            if i > 0:
                neighbor_jit_key = op_groups[gpu_group['op_groups'][i - 1]]['jit_key']
                if neighbor_jit_key not in jit_key_items[op_group_label[jit_key]]['neighbors']:
                    jit_key_items[op_group_label[jit_key]]['neighbors'].append(neighbor_jit_key)
                if jit_key not in jit_key_items[op_group_label[neighbor_jit_key]]['neighbors']:
                    jit_key_items[op_group_label[neighbor_jit_key]]['neighbors'].append(jit_key)

    op_group_type = {}
    for jit_key_item in jit_key_items:
        jit_key = jit_key_item['jit_key']
        if jit_key not in op_group_type:
            waiting_list = [jit_key]
            i = 0
            while i < len(waiting_list):
                jit_key = waiting_list[i]
                jit_key_item = jit_key_items[op_group_label[jit_key]]
                neighbors = jit_key_item['neighbors']
                neighbor_types = []
                for neighbor_jit_key in neighbors:
                    if neighbor_jit_key in op_group_type:
                        neighbor_types.append(op_group_type[neighbor_jit_key])
                    elif neighbor_jit_key not in waiting_list:
                        waiting_list.append(neighbor_jit_key)
                for count in range(len(jit_key_items)):
                    if count not in neighbor_types:
                        op_group_type[jit_key] = count
                        break
                i += 1

    for op_group in op_groups:
        jit_key = op_group['jit_key']
        op_group['jit_type'] = op_group_type[jit_key]

    max_jit_type = 10
    # because the number of using color, jit_type should not be too large
    for gpu_group in schedule:
        for i, op_group_index in enumerate(gpu_group['op_groups']):
            op_group = op_groups[op_group_index]
            if op_group['jit_type'] >= max_jit_type:
                neighbor_types = []
                if i > 0:
                    neighbor_jit_type = op_groups[gpu_group['op_groups'][i - 1]]['jit_type']
                    neighbor_types.append(neighbor_jit_type)
                if i < len(gpu_group['op_groups']) - 1:
                    neighbor_jit_type = op_groups[gpu_group['op_groups'][i + 1]]['jit_type']
                    if neighbor_jit_type < max_jit_type:
                        neighbor_types.append(neighbor_jit_type)
                for count in range(max_jit_type):
                    if count not in neighbor_types:
                        op_group['jit_type'] = count
                        break


    return network, op_groups, schedule

def process_data_of_extern_node(extern_node_indexs, nodes, edges):
    for node_index in extern_node_indexs:
        extern_node = nodes[node_index]
        if 'has_feature' in extern_node['attrs']:
            continue
        process_data_of_extern_node(extern_node['children'], nodes, edges)

        tail_node_indexes = []
        for child_index in extern_node['children']:
            child = nodes[child_index]
            if not child['attrs']['has_feature']:
                continue
            if len(child['next']) == 0:
                tail_node_indexes.append([child_index, 1])
                print('no next')
                continue
            in_edge = False
            out_edge = False
            for edge_index in child['next']:
                edge = edges[edge_index]
                for next_node_index in edge['end_stack']:
                    if next_node_index in extern_node['children']:
                        in_edge = True
                        break
                if in_edge:
                    break
                else:
                    out_edge = True
            if not in_edge and out_edge:
                tail_node_indexes.append([child_index, 0])

        tail_node_indexes = sorted(tail_node_indexes, key=lambda x:x[1], reverse=False)
        has_feature = 0
        var_node_id = []
        shape = []
        for tail_node_index in tail_node_indexes:
            if nodes[tail_node_index[0]]['attrs']['has_feature']:
                has_feature += 1
                var_node_id.append(nodes[tail_node_index[0]]['attrs']['var_node_id'])
                shape.append(nodes[tail_node_index[0]]['attrs']['shape'])
        if has_feature > 1:
            print('two tail')
        extern_node['attrs']['has_feature'] = has_feature > 0
        extern_node['attrs']['var_node_id'] = var_node_id[0] if has_feature > 0 else -1
        extern_node['attrs']['shape'] = shape[0] if has_feature > 0 else ''

def reverse_edge_direction(edge, nodes):
    edge['reversed'] = True
    for end_node_index in edge['end_stack']:
        nodes[end_node_index]['pre'].remove(edge['index'])
        nodes[end_node_index]['next'].append(edge['index'])

    for start_node_index in edge['start_stack']:
        nodes[start_node_index]['pre'].append(edge['index'])
        nodes[start_node_index]['next'].remove(edge['index'])

    temp = edge['start']
    edge['start'] = edge['end']
    edge['end'] = temp
    temp = edge['start_index']
    edge['start_index'] = edge['end_index']
    edge['end_index'] = temp
    temp = edge['start_stack']
    edge['start_stack'] = edge['end_stack']
    edge['end_stack'] = temp


def remove_duplicate_edges(nodes, edges):
    print('removing duplicate edges from', len(edges))
    edge_dic = {}
    duplicate_count = 0
    edge_i = 0
    while edge_i < len(edges):
        edge = edges[edge_i]
        edge_key = str(edge['start']) + '-' + str(edge['end'])
        if edge_key in edge_dic:
            duplicate_count += 1
            keep_edge = edges[edge_dic[edge_key]]
            target_key = 'end_stack'
            node_key = 'pre'
            if edge['start_stack'][-1] != keep_edge['start_stack'][-1]:
                target_key = 'start_stack'
                node_key = 'next'

            for i in range(len(edge[target_key])):
                if edge[target_key][i] != keep_edge[target_key][i]:
                    edge[target_key] = edge[target_key][i:]
                    break
                else:
                    nodes[edge[target_key][i]][node_key].remove(edge['index'])

            edge[target_key.replace('_stack', '')] = edge[target_key][0]
        else:
            edge_dic[edge_key] = edge['index']
            edge_i += 1

    print('remove', duplicate_count, 'duplicate edges from', len(edges))
    return duplicate_count


def split_oversize_extern_nodes(nodes, edges, root_nodes):
    # 遍历节点，当某个节点的子节点数量超过限制
    # 计算连通域，连通域分为三类，过大>30，适当>5,过小<=5
    # 最小割算法分割过大的连通域

    #compute_link_relation
    all_link_relation = []
    for node in nodes:
        links = {}
        for edge_index in node['pre']:
            edge = edges[edge_index]
            for node_index in edge['start_stack']:
                links[node_index] = True
        for edge_index in node['next']:
            edge = edges[edge_index]
            for node_index in edge['end_stack']:
                links[node_index] = True
        all_link_relation.append(links)

    max_threshold = 50
    min_threshold = 9
    virtual_node_count = 0
    extern_nodes_to_check = root_nodes
    while len(extern_nodes_to_check) > 0:
        node_index = extern_nodes_to_check.pop()
        node = nodes[node_index]
        children = node['children']
        if len(children) <= max_threshold:
            for child in children:
                if nodes[child]['level'] > 0:
                    extern_nodes_to_check.append(child)
            continue
        # 节点的子节点数量超过限制
        cluster_ids = [-1] * len(children)
        cluster_count = 0

        for i, child_index in enumerate(children):
            cluster_id = -1
            change_dic = {}

            for j in range(i):
                neighbor_index = children[j]
                if neighbor_index in all_link_relation[child_index]:
                    if cluster_id == -1:
                        cluster_id = cluster_ids[j]
                        cluster_ids[i] = cluster_id
                    else:
                        change_dic[cluster_ids[j]] = cluster_id
            if cluster_id == -1:
                cluster_ids[i] = cluster_count
                cluster_count += 1

            for j in range(i):
                if cluster_ids[j] in change_dic:
                    cluster_ids[j] = change_dic[cluster_ids[j]]

        cluster_set = list(set(cluster_ids))
        cluster_node_indexes = [[] for cluster_id in cluster_set]
        for i, cluster_id in enumerate(cluster_ids):
            cluster_node_indexes[cluster_set.index(cluster_id)].append(children[i])
        cluster_node_indexes = sorted(cluster_node_indexes, key=lambda x:len(x))
        node_groups_to_add = []
        current_node_indexes = []
        for node_indexes in cluster_node_indexes:
            if len(node_indexes) <= min_threshold:
                current_node_indexes.extend(node_indexes)
                if len(current_node_indexes) > min_threshold:
                    node_groups_to_add.append(current_node_indexes)
                    current_node_indexes = []
            elif len(node_indexes) <= max_threshold:
                node_groups_to_add.append(node_indexes)
            else:
                graph = {}
                for i in node_indexes:
                    graph[str(i)] = []
                    for j in node_indexes:
                        if i == j:
                            continue
                        if j in all_link_relation[i]:
                            graph[str(i)].append(str(j))
                groups, del_edge_num = get_best_partition(graph, num=100)
                cluster_node_indexes.append([int(x) for x in groups[0]])
                cluster_node_indexes.append([int(x) for x in groups[1]])
        if len(current_node_indexes) > 0:
            node_groups_to_add.append(current_node_indexes)

        new_children = []
        for node_indexes in node_groups_to_add:
            # update all_link_relation for new children
            link_relation = {}
            for i in node_indexes:
                nodes[i]['parent'] = len(nodes)
                for key in all_link_relation[i]:
                    if key not in node_indexes:
                        link_relation[key] = True
            all_link_relation.append(link_relation)
            new_children.append(len(nodes))

            nodes.append({
                'index': len(nodes),
                'id': -1,
                'title': 'group-' + str(virtual_node_count),
                'level': 1,
                'children': node_indexes,
                'parent': node['index'],
                'stacks': [],
                'next': [],
                'pre': [],
                'type': 'virtual_node',
                'code_path': '',
                'line_num': -1,
                'is_var': False,
                'attrs': {},
                'gpu_distribution': np.zeros_like(node['gpu_distribution']),
                'time_cost': 0,
                'op_groups': []
            })
            print('add virtual node', virtual_node_count)
            virtual_node_count += 1
        node['children'] = new_children


        if len(new_children) > max_threshold:
            extern_nodes_to_check.append(node_index)
        else:
            for child in new_children:
                extern_nodes_to_check.append(child)


    return nodes, edges


def compute_dependency_of_leaf_nodes(all_nodes, all_edges):
    leaf_node_num = 0
    for node in all_nodes:
        if node['level'] > 0:
            break
        leaf_node_num += 1

    node_dependency = [[] for i in range(leaf_node_num)]
    node_reverse_dependency = [[] for i in range(leaf_node_num)]

    for edge in all_edges:
        node_dependency[edge['end_stack'][-1]].append(edge['start_stack'][-1])
        node_reverse_dependency[edge['start_stack'][-1]].append(edge['end_stack'][-1])

    heads = []
    for i, dependency in enumerate(node_dependency):
        if len(dependency) == 0:
            heads.append(i)

    head_i = 0
    while len(heads) < leaf_node_num:
        head = heads[head_i]
        dependency = node_reverse_dependency[head]
        for depend_index in dependency:
            if depend_index not in heads:
                heads.append(depend_index)
                node_dependency[depend_index].extend(node_dependency[head])
                node_dependency[depend_index] = list(set(node_dependency[depend_index]))
                heads.append(depend_index)
        head_i += 1

    return node_dependency


def compute_dependency_of_op_group(op_group, schedule, schedule_index, network, op_groups, node_dependency):
    current_end_time = 0
    current_depend_op_group_index = []
    if len(schedule[schedule_index]['op_groups']) > 0:
        current_end_time = op_groups[schedule[schedule_index]['op_groups'][-1]]['end_time']
    for gpu_index, gpu_schedule in enumerate(schedule):
        for index in range(len(gpu_schedule['op_groups'])):
            if is_depend_on(op_group, op_groups[gpu_schedule['op_groups'][index]], node_dependency):
                if op_groups[gpu_schedule['op_groups'][index]]['end_time'] > current_end_time:
                    current_depend_op_group_index.append(gpu_schedule['op_groups'][index])
                    current_end_time = op_groups[gpu_schedule['op_groups'][index]]['end_time']
    if current_end_time > 0:
        current_end_time += 0.01
    op_group['start_time'] = current_end_time
    op_group['end_time'] = current_end_time + op_group['time_cost']
    op_group['depend_op_index'] = current_depend_op_group_index[-1:]


def is_depend_on(op_group1, op_group2, node_dependency):
    for node_index1 in op_group1['nodes']:
        for depend_index in node_dependency[node_index1]:
            if depend_index in op_group2['nodes']:
                return True
    return False


def has_intersection(arr1, arr2):
    for x in arr1:
        for y in arr2:
            if x == y:
                return True
    return False


def get_exploring_height_and_level(all_nodes, exploring_nodes):
    # exploring_height and exploring_level
    for i in range(len(exploring_nodes) - 1, -1, -1):
        exploring_node = exploring_nodes[i]
        node = all_nodes[exploring_node['index']]
        exploring_level = 1
        exploring_childs = []
        for child_index in exploring_node['children']:
            child_node = all_nodes[child_index]
            if child_node['exploring']:
                exploring_level = max(child_node['exploring_level'] + 1, exploring_level)
                exploring_childs.append(child_index)
        exploring_node['exploring_level'] = exploring_level
        exploring_node['exploring_childs'] = exploring_childs
        node['exploring_level'] = exploring_level
        node['exploring_childs'] = exploring_childs

    for exploring_node in exploring_nodes:
        node = all_nodes[exploring_node['index']]
        if exploring_node['parent'] == -1:
            exploring_node['exploring_height'] = 0
        else:
            exploring_node['exploring_height'] = all_nodes[exploring_node['parent']]['exploring_height'] + 1
        node['exploring_height'] = exploring_node['exploring_height']
    return exploring_nodes


def get_hull_of_exploring_nodes(all_nodes, all_edges, exploring_nodes, nodes, update_layout):
    hulls = []
    margin = 5
    # compute x-y-range and exploring_height and exploring_level and descendants
    for i in range(len(exploring_nodes) - 1, -1, -1):
        exploring_node = exploring_nodes[i]
        node = all_nodes[exploring_node['index']]
        exploring_level = 1
        descendants = {}
        exploring_childs = []
        for child_index in exploring_node['children']:
            child_node = all_nodes[child_index]
            if child_node['exploring']:
                exploring_level = max(child_node['exploring_level'] + 1, exploring_level)
                for key in child_node['descendants']:
                    descendants[key] = True
                exploring_childs.append(child_index)
            else:
                descendants[child_index] = True
        exploring_node['exploring_level'] = exploring_level
        exploring_node['descendants'] = descendants
        exploring_node['exploring_childs'] = exploring_childs
        node['exploring_level'] = exploring_level
        node['descendants'] = descendants
        node['exploring_childs'] = exploring_childs

    for exploring_node in exploring_nodes:
        node = all_nodes[exploring_node['index']]
        if exploring_node['exploring_level'] > 3:
            exploring_node['exploring_height'] = -1
        elif exploring_node['parent'] == -1:
            exploring_node['exploring_height'] = 0
        else:
            exploring_node['exploring_height'] = all_nodes[exploring_node['parent']]['exploring_height'] + 1
        node['exploring_height'] = exploring_node['exploring_height']

    new_exploring_nodes = []
    for exploring_node in exploring_nodes:
        if exploring_node['exploring_height'] > -1:
            new_exploring_nodes.append(exploring_node)
    exploring_nodes = new_exploring_nodes

    if len(exploring_nodes) > 0:
        # construct the grid
        for i, node in enumerate(nodes):
            node['state'] = {}
            node['node_index'] = node['index']
            node['exploring_node_indexes'] = -1
            for exploring_node in exploring_nodes:
                if node['index'] in exploring_node['descendants']:
                    node['state'][exploring_node['index']] = 'inside'
                    node['exploring_node_indexes'] = exploring_node['index']
                else:
                    node['state'][exploring_node['index']] = 'outside'
            node['index'] = i
        grid = construct_grid(nodes, exploring_nodes, update_layout=update_layout)
        for i, node in enumerate(nodes):
            node['chosen'] = {}

        for i in range(len(exploring_nodes) - 1, -1, -1):
            exploring_node = exploring_nodes[i]
            # debug_plot(nodes, exploring_node['index'])

            hull_points, del_btn_pos = get_hull_of_exploring_node(grid, nodes, margin, exploring_node)
            hulls.append({
                'node_index': exploring_node['index'],
                 'hull_points': hull_points,
                'exploring_level': exploring_node['exploring_level'],
                'exploring_height': exploring_node['exploring_height'],
                'del_btn_pos': del_btn_pos
            })

        new_nodes = []
        for node in nodes:
            if 'node_index' in node:
                new_nodes.append(node)
        nodes = new_nodes
    return hulls, nodes


def construct_grid(items, exploring_nodes, update_layout=False):
    # construct the grid
    ys = [item['y'] for item in items]
    ys = list(set(ys))
    ys = sorted(ys)

    grid = {
        'columns': [],
        'x_to_index': {}
    }
    link_relation = []
    for i, item in enumerate(items):
        if item['x'] not in grid['x_to_index']:
            grid['x_to_index'][item['x']] = len(grid['columns'])
            grid['columns'].append({
                'items': [],
                'x': item['x'],
                'left': item['x'],
                'right': item['x'],
            })
        column_index = grid['x_to_index'][item['x']]
        grid['columns'][column_index]['left'] = min(grid['columns'][column_index]['left'], item['x'] - item['w'] / 2)
        grid['columns'][column_index]['right'] = max(grid['columns'][column_index]['right'], item['x'] + item['w'] / 2)
        grid['columns'][column_index]['items'].append(item)

    # sort the grid
    grid['columns'] = sorted(grid['columns'], key=lambda column: column['x'])

    for column in grid['columns']:
        column['items'] = sorted(column['items'], key=lambda item: item['y'])


    if update_layout:
        # 确定exploring节点之间的父子节点关系
        node_index_to_forest_index = {}
        exploring_forest = []
        for height in range(3):
            tree = []
            for i, exploring_node in enumerate(exploring_nodes):
                if exploring_node['exploring_height'] == height:
                    node_index_to_forest_index[exploring_node['index']] = [height, len(tree)]
                    parent = exploring_node['parent']
                    if parent in node_index_to_forest_index:
                        parent = node_index_to_forest_index[parent][1]
                        exploring_forest[height - 1][parent]['children'].append(len(tree))
                    tree.append({
                        'index': exploring_node['index'],
                        'parent': parent,
                        'children': [],
                        'height': height
                    })
            exploring_forest.append(tree)

        # 统计一下各个exploring node下所有节点的数量和平均y
        info = {}
        for item in items:
            exploring_node_index = item['exploring_node_indexes']
            if exploring_node_index == -1 or exploring_node_index in node_index_to_forest_index:
                if exploring_node_index not in info:
                    info[exploring_node_index] = {
                        'num': 0,
                        'sum_of_y': 0
                    }
                info[exploring_node_index]['num'] += 1
                info[exploring_node_index]['sum_of_y'] += item['y']

        for height in range(1, -1, -1):
            for ex_node in exploring_forest[height]:
                if len(ex_node['children']) > 0:
                    info[ex_node['index']] = {
                        'num': 0,
                        'sum_of_y': 0
                    }
                    for child_forest_index in ex_node['children']:
                        child_index = exploring_forest[height + 1][child_forest_index]['index']
                        info[ex_node['index']]['num'] += info[child_index]['num']
                        info[ex_node['index']]['sum_of_y'] += info[child_index]['sum_of_y']

        # 自顶向下计算叶子节点的顺序
        exploring_forest[0].append({
            'index': -1,
            'parent': -1,
            'children': [],
            'height': 0
        })
        exploring_node_indexes = sorted(exploring_forest[0], key=lambda node:(info[node['index']]['sum_of_y'] / info[node['index']]['num']))
        i = 0
        while i < len(exploring_node_indexes):
            curr_node = exploring_node_indexes[i]
            if len(curr_node['children']) > 0:
                height = curr_node['height']
                sub_exploring_node_indexes = [exploring_forest[height + 1][j] for j in curr_node['children']]
                sub_exploring_node_indexes = sorted(sub_exploring_node_indexes, key=lambda node:(info[node['index']]['sum_of_y'] / info[node['index']]['num']))
                exploring_node_indexes = exploring_node_indexes[:i] + sub_exploring_node_indexes + exploring_node_indexes[i + 1:]
            else:
                i += 1
        exploring_node_indexes = [node['index'] for node in exploring_node_indexes]

        top = float('inf')
        bottom = float('-inf')
        for i, column in enumerate(grid['columns']):
            top = min(top, column['items'][0]['y'] - column['items'][0]['h'] / 2)
            bottom = max(bottom, column['items'][-1]['y'] + column['items'][-1]['h'] / 2)

        for column in grid['columns']:
            col_items = column['items']
            target_ys = []
            for i, item in enumerate(col_items):
                target_ys.append(item['y'])

            item_num = len(col_items)
            for i in range(item_num):
                for j in range(1, item_num - i):
                    curr_index = exploring_node_indexes.index(col_items[j]['exploring_node_indexes'])
                    pre_index = exploring_node_indexes.index(col_items[j - 1]['exploring_node_indexes'])
                    if curr_index < pre_index:
                        temp = col_items[j]
                        col_items[j] = col_items[j - 1]
                        col_items[j - 1] = temp
            # 全部挤上面
            start_y = top
            curr_ys = []
            curr_sum = 0
            target_sum = 0

            for i, item in enumerate(col_items):
                curr_ys.append(start_y + item['h'] / 2)
                start_y += item['h']
                target_sum += target_ys[i]
                curr_sum += curr_ys[i]

            start = 0
            end = len(curr_ys)
            left_top = 0
            left_bottom = bottom - start_y
            while start < end:
                avg_delta = (target_sum - curr_sum) / (end - start)
                if avg_delta >= left_bottom:
                    avg_delta = left_bottom
                if avg_delta < -left_top:
                    avg_delta = -left_top

                num = end - start
                while start < end:
                    start_delta = target_ys[start] - curr_ys[start]
                    if start_delta < -left_top:
                        start_delta = -left_top
                    if start_delta <= avg_delta:
                        curr_ys[start] += start_delta
                        left_top = -start_delta
                        start += 1
                    else:
                        break
                while start < end:
                    end_delta = target_ys[end - 1] - curr_ys[end - 1]
                    if end_delta > left_bottom:
                        end_delta = left_bottom
                    if end_delta >= avg_delta:
                        curr_ys[end - 1] += end_delta
                        left_bottom = end_delta
                        end -= 1
                    else:
                        break
                if end - start == num:
                    for i in range(start, end):
                        curr_ys[i] += avg_delta
                    break

            # curr_ys 就是最后的y值了，然后每个item按照这个y为目标调整y,top,bottom,node_top, node_bottom
            for i, item in enumerate(col_items):
                delta_y = curr_ys[i] - item['y']
                keys = ['y', 'node_top', 'node_bottom']
                for key in keys:
                    item[key] += delta_y


        a = 0
        # 根据各个exploring_nodes，从小到大，父节点先于子节点，因为节点间只有包含或者分离两种关系，所以后面的理论上不会影响前面的排序


    # update the left and right of column
    for i in range(1, len(grid['columns'])):
        column = grid['columns'][i]
        center = (column['left'] + grid['columns'][i - 1]['right']) / 2
        column['left'] = center
        grid['columns'][i - 1]['right'] = center


    # update the left and right of item and add null items to grid
    # for each column
    top = float('inf')
    bottom = float('-inf')
    for i, column in enumerate(grid['columns']):
        for j, item in enumerate(column['items']):
            item['left'] = column['left']
            item['right'] = column['right']
            item['top'] = item['y'] - item['h'] * 0.5
            item['bottom'] = item['y'] + item['h'] * 0.5
        top = min(top, column['items'][0]['top'])
        bottom = max(bottom, column['items'][-1]['bottom'])

    for i, column in enumerate(grid['columns']):
        new_items = []
        if column['items'][0]['top'] <= top:
            column['items'][0]['top'] = top
        else:
            new_items.append({
                'index': len(items),
                'left': column['left'],
                'right': column['right'],
                'top': top,
                'bottom': column['items'][0]['top'],
                'state': {}
            })
            new_items[-1]['x'] = (new_items[-1]['left'] + new_items[-1]['right']) / 2
            new_items[-1]['y'] = (new_items[-1]['top'] + new_items[-1]['bottom']) / 2
            new_items[-1]['w'] = new_items[-1]['right'] - new_items[-1]['left']
            new_items[-1]['h'] = new_items[-1]['bottom'] - new_items[-1]['top']
            for exploring_node in exploring_nodes:
                new_items[-1]['state'][exploring_node['index']] = 'other'
            items.append(new_items[-1])
        new_items.append(column['items'][0])

        for j in range(1, len(column['items'])):
            curr_item = column['items'][j]
            pre_item = new_items[-1]
            if pre_item['bottom'] > curr_item['top']:
                center = (pre_item['y'] + pre_item['h'] / 2 + curr_item['y'] - curr_item['h'] / 2) / 2
                pre_item['bottom'] = center
                curr_item['top'] = center
            elif pre_item['bottom'] < curr_item['top']:
                new_items.append({
                    'index': len(items),
                    'left': column['left'],
                    'right': column['right'],
                    'top': pre_item['bottom'],
                    'bottom': curr_item['top'],
                    'state': {}
                })
                new_items[-1]['x'] = (new_items[-1]['left'] + new_items[-1]['right']) / 2
                new_items[-1]['y'] = (new_items[-1]['top'] + new_items[-1]['bottom']) / 2
                new_items[-1]['w'] = new_items[-1]['right'] - new_items[-1]['left']
                new_items[-1]['h'] = new_items[-1]['bottom'] - new_items[-1]['top']
                for exploring_node in exploring_nodes:
                    new_items[-1]['state'][exploring_node['index']] = 'other'
                items.append(new_items[-1])
            new_items.append(curr_item)

        if new_items[-1]['bottom'] >= bottom:
            new_items[-1]['bottom'] = bottom
        else:
            new_items.append({
                'index': len(items),
                'left': column['left'],
                'right': column['right'],
                'top': new_items[-1]['bottom'],
                'bottom': bottom,
                'state': {}
            })
            new_items[-1]['x'] = (new_items[-1]['left'] + new_items[-1]['right']) / 2
            new_items[-1]['y'] = (new_items[-1]['top'] + new_items[-1]['bottom']) / 2
            new_items[-1]['w'] = new_items[-1]['right'] - new_items[-1]['left']
            new_items[-1]['h'] = new_items[-1]['bottom'] - new_items[-1]['top']
            for exploring_node in exploring_nodes:
                new_items[-1]['state'][exploring_node['index']] = 'other'
            items.append(new_items[-1])
        column['items'] = new_items

    # add link relation of items
    for i in range(len(items)):
        link_relation.append([])

    # in column
    for i, column in enumerate(grid['columns']):
        for j in range(1, len(column['items'])):
            curr_item = column['items'][j]
            pre_item = column['items'][j - 1]
            link_relation[curr_item['index']].append(pre_item['index'])
            link_relation[pre_item['index']].append(curr_item['index'])

    # between column
    for i in range(1, len(grid['columns'])):
        pre_column = grid['columns'][i - 1]['items']
        curr_column = grid['columns'][i]['items']
        pre_index = 0
        curr_index = 0
        while pre_index < len(pre_column) and curr_index < len(curr_column):
            pre_item = pre_column[pre_index]
            curr_item = curr_column[curr_index]
            if if_intersaction_of_two_range(pre_item['top'], pre_item['bottom'], curr_item['top'], curr_item['bottom']):
                link_relation[curr_item['index']].append(pre_item['index'])
                link_relation[pre_item['index']].append(curr_item['index'])

            if pre_index == len(pre_column) - 1:
                curr_index += 1
            elif curr_index == len(curr_column) - 1:
                pre_index += 1
            elif pre_item['bottom'] > curr_item['bottom']:
                curr_index += 1
            else:
                pre_index += 1
    grid['link_relation'] = link_relation
    return grid


def get_hull_of_exploring_node(grid, nodes, _margin, exploring_node):
    margin = _margin * (exploring_node['exploring_level'] - 0.3)
    exploring_node_index = exploring_node['index']
    exploring_childs = exploring_node['exploring_childs']
    descendants = exploring_node['descendants']

    link_relation = grid['link_relation']

    # add all nodes except outside nodes and its neighbor
    for i in range(len(link_relation)):
        nodes[i]['chosen'][exploring_node_index] = nodes[i]['state'][exploring_node_index] == 'inside'
        if nodes[i]['state'][exploring_node_index] == 'other':
            child_chosen = False
            for child_index in exploring_childs:
                if nodes[i]['chosen'][child_index]:
                    child_chosen = True
                    break
            nodes[i]['chosen'][exploring_node_index] = child_chosen
    # for i in range(len(link_relation)):
    #     nodes[i]['chosen'][exploring_node_index] = nodes[i]['state'][exploring_node_index] != 'outside'
    # for i in range(len(link_relation)):
    #     if nodes[i]['state'][exploring_node_index] == 'outside':
    #         for j, index in enumerate(link_relation[i]):
    #             if nodes[index]['state'][exploring_node_index] == 'other':
    #                 child_chosen = False
    #                 for child_index in exploring_childs:
    #                     if nodes[index]['chosen'][child_index]:
    #                         child_chosen = True
    #                         break
    #                 nodes[index]['chosen'][exploring_node_index] = child_chosen


    # 多连通域处理
    # debug_plot(nodes, exploring_node_index)
    merge_multi_component(nodes, link_relation, exploring_node_index)
    # debug_plot(nodes, exploring_node_index)

    # merge 同列相邻的块
    for i, column in enumerate(grid['columns']):
        column['groups'] = []
        j = 0
        while j < len(column['items']):
            if column['items'][j]['chosen'][exploring_node_index]:
                group = {
                    'top': column['items'][j]['top']
                }
                while j < len(column['items']) and column['items'][j]['chosen'][exploring_node_index]:
                    group['bottom'] = column['items'][j]['bottom']
                    j += 1
                column['groups'].append(group)
            j += 1

    # 按列扫描，建立包络的连接关系
    hull_edges = []
    hull_nodes = []
    pre_groups = []
    start_index = 0
    for i, column in enumerate(grid['columns']):
        j = 0
        if len(pre_groups) == 0:
            for group_index, group in enumerate(column['groups']):
                left_top_node = {
                    'index': len(hull_nodes),
                    'x': column['left'],
                    'y': group['top'],
                    'type': 'lt',
                    'g_id': str(i) + '_' + str(group_index)
                }
                hull_nodes.append(left_top_node)
                group['left_top_node_index'] = left_top_node['index']
                hull_edges.append([])
                left_bottom_node = {
                    'index': len(hull_nodes),
                    'x': column['left'],
                    'y': group['bottom'],
                    'type': 'lb',
                    'g_id': str(i) + '_' + str(group_index)
                }
                hull_nodes.append(left_bottom_node)
                group['left_bottom_node_index'] = left_bottom_node['index']
                hull_edges.append([])
                right_top_node = {
                    'index': len(hull_nodes),
                    'x': column['right'],
                    'y': group['top'],
                    'type': 'rt',
                    'g_id': str(i) + '_' + str(group_index)
                }
                hull_nodes.append(right_top_node)
                hull_edges.append([])
                right_bottom_node = {
                    'index': len(hull_nodes),
                    'x': column['right'],
                    'y': group['bottom'],
                    'type': 'rb',
                    'g_id': str(i) + '_' + str(group_index)
                }
                hull_nodes.append(right_bottom_node)
                hull_edges.append([])

                hull_edges[left_top_node['index']].append(left_bottom_node['index'])
                hull_edges[left_bottom_node['index']].append(left_top_node['index'])

                hull_edges[left_top_node['index']].append(right_top_node['index'])
                hull_edges[right_top_node['index']].append(left_top_node['index'])

                hull_edges[left_bottom_node['index']].append(right_bottom_node['index'])
                hull_edges[right_bottom_node['index']].append(left_bottom_node['index'])
                group['right_top_node_index'] = right_top_node['index']
                group['right_bottom_node_index'] = right_bottom_node['index']
        else:
            pre_j = 0
            while pre_j < len(pre_groups):
                pre_group = pre_groups[pre_j]
                pre_top = pre_group['top']
                pre_bottom = pre_group['bottom']
                group_index1 = j
                group_index2 = 0
                flag = False
                if j >= len(column['groups']):
                    # 当前group在pre_group下方，不相交，换到下一个pre_group
                    hull_edges[pre_group['right_top_node_index']].append(pre_group['right_bottom_node_index'])
                    hull_edges[pre_group['right_bottom_node_index']].append(pre_group['right_top_node_index'])
                    pre_j += 1
                    continue
                while j < len(column['groups']):
                    group = column['groups'][j]
                    if 'right_top_node_index' not in group:
                        left_top_node = {
                            'index': len(hull_nodes),
                            'x': column['left'],
                            'y': group['top'],
                            'type': 'lt',
                            'g_id': str(i) + '_' + str(j)
                        }
                        hull_nodes.append(left_top_node)
                        group['left_top_node_index'] = left_top_node['index']
                        hull_edges.append([])

                        right_top_node = {
                            'index': len(hull_nodes),
                            'x': column['right'],
                            'y': group['top'],
                            'type': 'rt',
                            'g_id': str(i) + '_' + str(j)
                        }
                        hull_nodes.append(right_top_node)
                        group['right_top_node_index'] = right_top_node['index']
                        hull_edges.append([])
                        hull_edges[left_top_node['index']].append(right_top_node['index'])
                        hull_edges[right_top_node['index']].append(left_top_node['index'])
                    else:
                        right_top_node = hull_nodes[group['right_top_node_index']]
                        left_top_node = hull_nodes[group['left_top_node_index']]

                    if pre_top < group['bottom']:
                        if pre_bottom <= group['top']:
                            # 当前group在pre_group下方，不相交，换到下一个pre_group
                            hull_edges[pre_group['right_top_node_index']].append(pre_group['right_bottom_node_index'])
                            hull_edges[pre_group['right_bottom_node_index']].append(pre_group['right_top_node_index'])
                            flag = True
                            break
                        else:
                            # 当前group是第一个和pre_group相交的group
                            if pre_j == 0 or pre_groups[pre_j - 1]['bottom'] < left_top_node['y']:
                                hull_edges[left_top_node['index']].append(pre_group['right_top_node_index'])
                                hull_edges[pre_group['right_top_node_index']].append(left_top_node['index'])
                            # else:
                            #     hull_edges[pre_groups[pre_j - 1]['right_bottom_node_index']].append(pre_group['right_top_node_index'])
                            #     hull_edges[pre_group['right_top_node_index']].append(pre_groups[pre_j - 1]['right_bottom_node_index'])

                            break
                    else:
                        # 当前group在pre_group上方，不相交

                        if 'left_bottom_node_index' not in group:
                            left_bottom_node = {
                                'index': len(hull_nodes),
                                'x': column['left'],
                                'y': group['bottom'],
                                'type': 'lb',
                                'g_id': str(i) + '_' + str(j)
                            }
                            hull_nodes.append(left_bottom_node)
                            group['left_bottom_node_index'] = left_bottom_node['index']
                            hull_edges.append([])

                            right_bottom_node = {
                                'index': len(hull_nodes),
                                'x': column['right'],
                                'y': group['bottom'],
                                'type': 'rb',
                                'g_id': str(i) + '_' + str(j)
                            }
                            hull_nodes.append(right_bottom_node)
                            group['right_bottom_node_index'] = right_bottom_node['index']
                            hull_edges.append([])
                            hull_edges[left_bottom_node['index']].append(right_bottom_node['index'])
                            hull_edges[right_bottom_node['index']].append(left_bottom_node['index'])
                        else:
                            left_bottom_node = hull_nodes[group['left_bottom_node_index']]
                            right_bottom_node = hull_nodes[group['right_bottom_node_index']]

                        hull_edges[left_top_node['index']].append(left_bottom_node['index'])
                        hull_edges[left_bottom_node['index']].append(left_top_node['index'])


                    j += 1
                    group_index1 = j
                if flag:
                    pre_j += 1
                    continue
                while j < len(column['groups']):
                    group = column['groups'][j]
                    if group['top'] >= pre_bottom:
                        break
                    j += 1
                    group_index2 = j

                for k in range(group_index1 + 1, group_index2):
                    group1 = column['groups'][k - 1]
                    group = column['groups'][k]

                    if 'left_top_node_index' not in group:
                        left_top_node = {
                            'index': len(hull_nodes),
                            'x': column['left'],
                            'y': group['top'],
                            'type': 'lt',
                            'g_id': str(i) + '_' + str(k)
                        }
                        hull_nodes.append(left_top_node)
                        group['left_top_node_index'] = left_top_node['index']
                        hull_edges.append([])

                        right_top_node = {
                            'index': len(hull_nodes),
                            'x': column['right'],
                            'y': group['top'],
                            'type': 'rt',
                            'g_id': str(i) + '_' + str(k)
                        }
                        hull_nodes.append(right_top_node)
                        group['right_top_node_index'] = right_top_node['index']
                        hull_edges.append([])
                        hull_edges[left_top_node['index']].append(right_top_node['index'])
                        hull_edges[right_top_node['index']].append(left_top_node['index'])
                    else:
                        left_top_node = hull_nodes[group['left_top_node_index']]
                        right_top_node = hull_nodes[group['right_top_node_index']]

                    if 'left_bottom_node_index' not in group1:
                        left_bottom_node = {
                            'index': len(hull_nodes),
                            'x': column['left'],
                            'y': group1['bottom'],
                            'type': 'lb',
                            'g_id': str(i) + '_' + str(k - 1)
                        }
                        hull_nodes.append(left_bottom_node)
                        group1['left_bottom_node_index'] = left_bottom_node['index']
                        hull_edges.append([])

                        right_bottom_node = {
                            'index': len(hull_nodes),
                            'x': column['right'],
                            'y': group1['bottom'],
                            'type': 'rb',
                            'g_id': str(i) + '_' + str(k - 1)
                        }
                        hull_nodes.append(right_bottom_node)
                        group1['right_bottom_node_index'] = right_bottom_node['index']
                        hull_edges.append([])

                        hull_edges[left_bottom_node['index']].append(right_bottom_node['index'])
                        hull_edges[right_bottom_node['index']].append(left_bottom_node['index'])
                    else:
                        left_bottom_node = hull_nodes[group1['left_bottom_node_index']]
                        right_bottom_node = hull_nodes[group1['right_bottom_node_index']]


                    hull_edges[left_top_node['index']].append(left_bottom_node['index'])
                    hull_edges[left_bottom_node['index']].append(left_top_node['index'])

                group = column['groups'][j - 1]
                if 'left_bottom_node_index' not in group:
                    left_bottom_node = {
                        'index': len(hull_nodes),
                        'x': column['left'],
                        'y': group['bottom'],
                        'type': 'lb',
                        'g_id': str(i) + '_' + str(j - 1)
                    }
                    hull_nodes.append(left_bottom_node)
                    group['left_bottom_node_index'] = left_bottom_node['index']
                    hull_edges.append([])

                    right_bottom_node = {
                        'index': len(hull_nodes),
                        'x': column['right'],
                        'y': group['bottom'],
                        'type': 'rb',
                        'g_id': str(i) + '_' + str(j - 1)
                    }
                    hull_nodes.append(right_bottom_node)
                    group['right_bottom_node_index'] = right_bottom_node['index']
                    hull_edges.append([])

                    hull_edges[left_bottom_node['index']].append(right_bottom_node['index'])
                    hull_edges[right_bottom_node['index']].append(left_bottom_node['index'])
                else:
                    left_bottom_node = hull_nodes[group['left_bottom_node_index']]
                    right_bottom_node = hull_nodes[group['right_bottom_node_index']]


                if pre_j == len(pre_groups) - 1 or pre_groups[pre_j + 1]['top'] > left_bottom_node['y']:
                    hull_edges[pre_group['right_bottom_node_index']].append(left_bottom_node['index'])
                    hull_edges[left_bottom_node['index']].append(pre_group['right_bottom_node_index'])
                else:
                    hull_edges[pre_groups[pre_j + 1]['right_top_node_index']].append(pre_group['right_bottom_node_index'])
                    hull_edges[pre_group['right_bottom_node_index']].append(pre_groups[pre_j + 1]['right_top_node_index'])


                pre_j += 1
                j -= 1
                if pre_j >= len(pre_groups) or (j > -1 and column['groups'][j]['bottom'] < pre_groups[pre_j]['top']):
                    j += 1

            while j < len(column['groups']):
                group = column['groups'][j]

                left_top_node = {
                    'index': len(hull_nodes),
                    'x': column['left'],
                    'y': group['top'],
                    'type': 'lt',
                    'g_id': str(i) + '_' + str(j)
                }
                hull_nodes.append(left_top_node)
                group['left_top_node_index'] = left_top_node['index']
                hull_edges.append([])

                right_top_node = {
                    'index': len(hull_nodes),
                    'x': column['right'],
                    'y': group['top'],
                    'type': 'rt',
                    'g_id': str(i) + '_' + str(j)
                }
                hull_nodes.append(right_top_node)
                group['right_top_node_index'] = right_top_node['index']
                hull_edges.append([])

                left_bottom_node = {
                    'index': len(hull_nodes),
                    'x': column['left'],
                    'y': group['bottom'],
                    'type': 'lb',
                    'g_id': str(i) + '_' + str(j)
                }
                hull_nodes.append(left_bottom_node)
                group['left_bottom_node_index'] = left_bottom_node['index']
                hull_edges.append([])

                right_bottom_node = {
                    'index': len(hull_nodes),
                    'x': column['right'],
                    'y': group['bottom'],
                    'type': 'rb',
                    'g_id': str(i) + '_' + str(j)
                }
                hull_nodes.append(right_bottom_node)
                group['right_bottom_node_index'] = right_bottom_node['index']
                hull_edges.append([])

                hull_edges[left_top_node['index']].append(right_top_node['index'])
                hull_edges[right_top_node['index']].append(left_top_node['index'])

                hull_edges[left_bottom_node['index']].append(right_bottom_node['index'])
                hull_edges[right_bottom_node['index']].append(left_bottom_node['index'])

                hull_edges[left_top_node['index']].append(left_bottom_node['index'])
                hull_edges[left_bottom_node['index']].append(left_top_node['index'])
                j += 1


        if i == len(grid['columns']) - 1:
            for group in column['groups']:
                hull_edges[group['right_top_node_index']].append(group['right_bottom_node_index'])
                hull_edges[group['right_bottom_node_index']].append(group['right_top_node_index'])

        pre_groups = column['groups']

        # plot_group(grid, hull_nodes[start_index:])
        start_index = len(hull_nodes)

    hulls = []
    for i, hull_node in enumerate(hull_nodes):
        if len(hull_edges[i]) == 0:
            continue
        hull = []
        hull.append(hull_node)
        pre_index = hull[-1]['index']
        curr_index = hull_edges[pre_index].pop()
        hull.append(hull_nodes[curr_index])
        hull_edges[curr_index].remove(pre_index)

        while len(hull_edges[curr_index]) > 0:
            next_index = hull_edges[curr_index].pop()
            pre_index = curr_index
            curr_index = next_index
            hull.append(hull_nodes[curr_index])
            hull_edges[curr_index].remove(pre_index)

        hull[0] = {
            'index': hull[0]['index'],
            'type': hull[0]['type'],
            'g_id': hull[0]['g_id'],
            'x': hull[0]['x'],
            'y': hull[0]['y']
        }
        hull.reverse()
        hulls.append(hull)

    del_btn_pos = {
        'x': float('-inf'),
        'y': float('inf')
    }
    del_btn_node_index = -1
    res = []
    for hull_i, hull in enumerate(hulls):
        new_hull = [hull[0]]
        for i in range(1, len(hull)):
            if hull[i]['x'] != new_hull[-1]['x'] or hull[i]['y'] != new_hull[-1]['y']:
                new_hull.append(hull[i])

        # TODO 优化折角
        hull = new_hull
        new_edges = []
        new_hull = [new_hull[0]]

        for i in range(1, len(hull) - 1):
            pre_node = hull[i - 1]
            curr_node = hull[i]
            next_node = hull[i + 1]
            if pre_node['x'] == curr_node['x'] and next_node['x'] == curr_node['x']:
                continue
            if pre_node['y'] == curr_node['y'] and next_node['y'] == curr_node['y']:
                continue
            direction = get_out_direction_of_edge(pre_node, curr_node)
            new_hull.append(curr_node)
            new_edges.append({
                'start': len(new_hull) - 2,
                'end': len(new_hull) - 1,
                'direction': direction
            })
            new_hull[-1]['pre_edge'] = len(new_edges) - 1
            new_hull[-2]['next_edge'] = len(new_edges) - 1

        direction = get_out_direction_of_edge(hull[-2], hull[-1])
        new_edges.append({
            'start': len(new_hull) - 1,
            'end': 0,
            'direction': direction
        })
        new_hull[0]['pre_edge'] = len(new_edges) - 1
        new_hull[-1]['next_edge'] = len(new_edges) - 1

        for node in new_hull:
            node['processed'] = 0

        # 尽可能缩
        if hull_i == 0:
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=False, bundle_state='inside', margin=margin)
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=True, bundle_state='outside', must_minus=True, margin=margin)
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=False, bundle_state='inside', must_minus=True, margin=margin)
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=True, bundle_state='outside', must_minus=True, margin=margin)
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=False, bundle_state='inside', must_minus=True, margin=margin)
            # hull_plot(new_hull)
        else:
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=False, bundle_state='inside', margin=margin)
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=True, bundle_state='outside', must_minus=True, margin=margin)
            # hull_plot(new_hull)
            new_hull, new_edges = angle_process(new_hull, new_edges, grid, exploring_node,
                                                expand=False, bundle_state='inside', must_minus=True, margin=margin)
            # hull_plot(new_hull)

        if new_hull is None:
            continue


        new_hull.append({
            'x': new_hull[0]['x'],
            'y': new_hull[0]['y']
        })

        for i, node in enumerate(new_hull):
            node['index'] = i

        if hull_i == 0:
            size = len(new_hull)
            deltas = [{
                'x': 0,
                'y': 0
            } for i in range(size)]
            for i in range(size - 1):
                node = new_hull[i]
                direction = new_edges[node['next_edge']]['direction']
                if direction == 'top':
                    deltas[i]['y'] = -1
                    deltas[i + 1]['y'] = -1
                elif direction == 'bottom':
                    deltas[i]['y'] = 1
                    deltas[i + 1]['y'] = 1
                elif direction == 'left':
                    deltas[i]['x'] = -1
                    deltas[i + 1]['x'] = -1
                elif direction == 'right':
                    deltas[i]['x'] = 1
                    deltas[i + 1]['x'] = 1
                else:
                    print('should not be here')

            if deltas[0]['x'] == 0:
                deltas[0]['x'] = deltas[-1]['x']
            else:
                deltas[-1]['x'] = deltas[0]['x']

            if deltas[0]['y'] == 0:
                deltas[0]['y'] = deltas[-1]['y']
            else:
                deltas[-1]['y'] = deltas[0]['y']
            for i in range(size):
                new_hull[i]['x'] += deltas[i]['x'] * margin
                new_hull[i]['y'] += deltas[i]['y'] * margin
            for i in range(size):
                if new_hull[i]['x'] > del_btn_pos['x']\
                        or (new_hull[i]['x'] == del_btn_pos['x'] and new_hull[i]['y'] < del_btn_pos['y']):
                    del_btn_pos['x'] = new_hull[i]['x']
                    del_btn_pos['y'] = new_hull[i]['y']
                    del_btn_node_index = i
            size -= 1

            # del_pre_index = (del_btn_node_index + size - 1) % size
            # del_next_index = (del_btn_node_index + 1) % size
            # delta_margin = exploring_node['exploring_level'] * _margin * 2
            # new_hull[del_pre_index]['x'] += deltas[del_pre_index]['x'] * margin
            # new_hull[del_pre_index]['x'] += deltas[del_pre_index]['x'] * delta_margin
            # new_hull[del_btn_node_index]['x'] += deltas[del_btn_node_index]['x'] * margin
            # new_hull[del_btn_node_index]['y'] += deltas[del_btn_node_index]['y'] * margin
            # new_hull[del_btn_node_index]['x'] += deltas[del_btn_node_index]['x'] * delta_margin
            # new_hull[del_btn_node_index]['y'] += deltas[del_btn_node_index]['y'] * delta_margin
            # new_hull[del_next_index]['y'] += deltas[del_next_index]['y'] * margin
            # new_hull[del_next_index]['y'] += deltas[del_next_index]['y'] * delta_margin
            # if del_btn_node_index == 0 or del_pre_index == 0:
            #     new_hull[size]['x'] += deltas[size]['x'] * margin
            #     new_hull[size]['x'] += deltas[size]['x'] * delta_margin
            # if del_btn_node_index == 0 or del_next_index == 0:
            #     new_hull[size]['y'] += deltas[size]['y'] * margin
            #     new_hull[size]['y'] += deltas[size]['y'] * delta_margin


            del_btn_pos['x'] = new_hull[del_btn_node_index]['x']
            del_btn_pos['y'] = new_hull[del_btn_node_index]['y']
        # hull_plot(new_hull)
        if hull_i > 0:
            new_hull.reverse()
        res.append(new_hull)

    return res, del_btn_pos


def if_intersaction_of_two_rect(top1, bottom1, left1, right1, top2, bottom2, left2, right2):
    if if_intersaction_of_two_range(top1, bottom1, top2, bottom2) \
            and if_intersaction_of_two_range(left1, right1, left2, right2):
        return True
    return False


def intersaction_of_two_rect(top1, bottom1, left1, right1, top2, bottom2, left2, right2):
    return {
        'top': max(top1, top2),
        'left': max(left1, left2),
        'bottom': min(bottom1, bottom2),
        'right': min(right1, right2)
    }


def if_intersaction_of_two_range(min1, max1, min2, max2):
    if min1 >= max2 or min2 >= max1:
        return False
    return True


def merge_multi_component(items, link_relation, exploring_node_index):
    # get clusters
    cluster_ids = [-1] * len(items)
    cluster_count = 0

    for i, item in enumerate(items):
        if not item['chosen'][exploring_node_index]:
            continue
        cluster_id = -1
        change_dic = {}

        for neighbor_index in link_relation[i]:
            if neighbor_index < i and cluster_ids[neighbor_index] != -1:
                if cluster_id == -1:
                    cluster_id = cluster_ids[neighbor_index]
                    cluster_ids[i] = cluster_id
                else:
                    change_dic[cluster_ids[neighbor_index]] = cluster_id
        if cluster_id == -1:
            cluster_ids[i] = cluster_count
            cluster_count += 1

        for j in range(i):
            if cluster_ids[j] in change_dic:
                cluster_ids[j] = change_dic[cluster_ids[j]]

    cluster_set = list(set(cluster_ids))
    if -1 in cluster_set:
        cluster_set.remove(-1)
    cluster_count = len(cluster_set)
    if cluster_count == 1:
        return

    # debug_plot(items, exploring_node_index)
    cluster_match = {}
    for i, id in enumerate(cluster_set):
        cluster_match[id] = i

    cluster_nodes = [{
        'index': -1,
        'height': 0
    } for i in range(cluster_count)]
    cluster_node_count = cluster_count

    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id != -1:
            cluster_ids[i] = cluster_match[cluster_id]
        elif items[i]['state'][exploring_node_index] == 'other':
            cluster_ids[i] = cluster_node_count
            cluster_nodes.append({
                'index': i,
                'height': items[i]['h']
            })
            cluster_node_count += 1

    cluster_node_link_relation = [[] for i in range(cluster_node_count)]

    for i, neighbor_indexes in enumerate(link_relation):
        if cluster_ids[i] != -1:
            start = cluster_ids[i]
            for j in neighbor_indexes:
                if cluster_ids[j] != -1:
                    end = cluster_ids[j]
                    if start != end:
                        if start not in cluster_node_link_relation[end]:
                            cluster_node_link_relation[end].append(start)
                        if end not in cluster_node_link_relation[start]:
                            cluster_node_link_relation[start].append(end)

    index_to_add = find_nodes_to_link(cluster_nodes, cluster_node_link_relation, cluster_count)

    for i in index_to_add:
        items[i]['chosen'][exploring_node_index] = True
        # debug_plot(items, exploring_node_index)
    return


def debug_plot(items, exploring_noed_index):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpathes

    margin = 4
    fig, ax = plt.subplots()
    # 长方形
    for item in items:
        rect = mpathes.Rectangle(np.array([item['left'], -item['bottom']]), (item['right'] - item['left']), (item['bottom'] - item['top']), color='black')
        ax.add_patch(rect)
    for item in items:
        color = 'white'
        try:
            if item['state'][exploring_noed_index] == 'inside':
                color = 'green'
            elif item['state'][exploring_noed_index] == 'outside':
                color = 'red'
            elif item['chosen'][exploring_noed_index]:
                color = 'blue'
            else:
                color = 'yellow'
        except:
            print('no state or no chosen')
        rect = mpathes.Rectangle(np.array([item['left'] + margin, -item['bottom'] + margin]), (item['right'] - item['left'] - 2 * margin), (item['bottom'] - item['top'] - 2 * margin), color=color)
        ax.add_patch(rect)
    for item in items:
        plt.text((item['left'] + item['right']) / 2, -item['bottom'], str(item['index']), fontsize=6)

    plt.axis('equal')
    plt.grid()
    plt.show()


def hull_plot(hull):
    if hull is None:
        return
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpathes
    from matplotlib.path import Path as path

    fig, ax = plt.subplots()
    # 长方形
    items = []
    codes = [path.MOVETO]
    for node in hull:
        items.append((node['x'], -node['y']))
        codes.append(path.LINETO)
    items.append(items[0])
    codes[-1] = path.CLOSEPOLY

    polygon = path(items, codes)
    patch = mpathes.PathPatch(polygon, facecolor='green', lw=2)
    ax.add_patch(patch)

    for item in hull:
        plt.text(item['x'], -item['y'], str(item['index']), fontsize=6)


    plt.axis('equal')
    plt.grid()
    plt.show()


def hull_plot1(nodes, edges, start_node):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpathes
    from matplotlib.path import Path as path

    fig, ax = plt.subplots()
    # 长方形
    items = [(start_node['x'], -start_node['y'])]
    codes = [path.MOVETO]

    start_index = start_node['index']

    while True:
        curr_edge_index = start_node['next_edge']
        curr_edge = edges[curr_edge_index]
        end_node_index = curr_edge['end']
        end_node = nodes[end_node_index]

        if end_node['index'] == start_index:
            break
        codes.append(path.LINETO)
        items.append((end_node['x'], -end_node['y']))
        start_node = end_node

    items.append(items[0])
    codes.append(path.CLOSEPOLY)

    polygon = path(items, codes)
    patch = mpathes.PathPatch(polygon, facecolor='green', lw=2)
    ax.add_patch(patch)

    plt.axis('equal')
    plt.grid()
    plt.show()


def plot_group(grid, hull):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpathes

    margin = 4
    fig, ax = plt.subplots()

    for column in grid['columns']:
        left = column['left']
        right = column['right']
        for group in column['groups']:
            top = group['top']
            bottom = group['bottom']
            rect = mpathes.Rectangle(np.array([left + margin, -bottom + margin]),
                                     (right - left - 2 * margin),
                                     (bottom - top - 2 * margin), color='green')
            ax.add_patch(rect)

    for i in range(len(hull)):
        item = hull[i]
        x = item['x']
        y = -item['y']

        plt.text(x, y, str(item['index']), fontsize=6)


    plt.axis('equal')
    plt.grid()
    plt.show()


def get_out_direction_of_edge(node, next_node):
    if node['x'] == next_node['x']:
        if 'l' in node['type'] and 'l' in next_node['type']:
            if node['g_id'] == next_node['g_id'] \
                    or ('t' in node['type'] and 't' in next_node['type']) \
                    or ('b' in node['type'] and 'b' in next_node['type']):
                return 'left'
            else:
                return 'right'
        elif 'r' in node['type'] and 'r' in next_node['type']:
            if node['g_id'] == next_node['g_id'] \
                    or ('t' in node['type'] and 't' in next_node['type']) \
                    or ('b' in node['type'] and 'b' in next_node['type']):
                return 'right'
            else:
                return 'left'
        else:
            left_y = node['y']
            right_y = next_node['y']
            if 'r' in node['type']:
                left_y = next_node['y']
                right_y = node['y']
            if 't' in node['type'] and 't' in next_node['type']:
                if left_y < right_y:
                    return 'left'
                else:
                    return 'right'
            else:
                if left_y < right_y:
                    return 'right'
                else:
                    return 'left'
    elif node['y'] == next_node['y']:
        if 't' in node['type'] and 't' in next_node['type']:
            return 'top'
        elif 'b' in node['type'] and 'b' in next_node['type']:
            return 'bottom'
        else:
            top_x = node['x']
            bottom_x = next_node['x']
            if 'b' in node['type']:
                top_x = next_node['x']
                bottom_x = node['x']
            if 'l' in node['type'] and 'l' in next_node['type']:
                if top_x < bottom_x:
                    return 'top'
                else:
                    return 'bottom'
            else:
                if top_x < bottom_x:
                    return 'bottom'
                else:
                    return 'top'
    else:
        print('should not be here')


def find_nodes_to_link(cluster_nodes, link_relation, num):
    if num <= 1:
        return []
    res = []
    path_matrix = []
    for i in range(num - 1):
        path_matrix.append(my_dijkstra_shortest_path(cluster_nodes, link_relation, i, [j for j in range(i + 1, num)]))

    min_path = []
    min_path_size = float('inf')
    start = -1
    end = -1
    for i, row in enumerate(path_matrix):
        for j, path in enumerate(row):
            if path['value'] < min_path_size:
                min_path_size = path['value']
                min_path = path['points']
                start = i
                end = i + j + 1

    cluster_to_merge = [start, end]
    for i in min_path:
        if i < num:
            if i not in cluster_to_merge:
                cluster_to_merge.append(i)
        else:
            res.append(cluster_nodes[i]['index'])
        neighbors = link_relation[i]
        for neighbor in neighbors:
            if neighbor < num and neighbor not in cluster_to_merge:
                cluster_to_merge.append(neighbor)

    if len(cluster_to_merge) == num:
        return res
    new_num = num - len(cluster_to_merge) + 1

    del_i = cluster_to_merge[1:]

    for i in min_path:
        if i > num:
            del_i.append(i)

    new_cluster_nodes = []
    old_index_to_new_index = {}
    new_index_to_old_index = {}
    for i, node in enumerate(cluster_nodes):
        if i not in del_i:
            old_index_to_new_index[i] = len(new_cluster_nodes)
            new_index_to_old_index[len(new_cluster_nodes)] = i
            new_cluster_nodes.append(node)
    for i in del_i:
        old_index_to_new_index[i] = old_index_to_new_index[start]

    new_link_relation = [[] for i in range(len(new_cluster_nodes))]

    for i in range(len(new_cluster_nodes)):
        neighbor = link_relation[new_index_to_old_index[i]]
        for j in neighbor:
            k = old_index_to_new_index[j]
            if i != k:
                if i not in new_link_relation[k]:
                    new_link_relation[k].append(i)
                if k not in new_link_relation[i]:
                    new_link_relation[i].append(k)

    res = res + find_nodes_to_link(new_cluster_nodes, new_link_relation, new_num)
    return res


def my_dijkstra_shortest_path(cluster_nodes, link_relation, start, ends):
    distances = [float('inf')] * len(link_relation)
    sum_ys = [float('inf')] * len(link_relation)
    pres = [-1 for i in range(len(link_relation))]
    ends_num = len(ends)
    used = [False] * len(link_relation)
    distances[start] = 0
    sum_ys[start] = 0

    queue = []
    heapq.heappush(queue, (0, start))

    while len(queue) > 0 and ends_num > 0:
        _, index = heapq.heappop(queue)
        if used[index] == True:
            continue
        used[index] = True
        neighbors = link_relation[index]
        for neighbor in neighbors:
            if used[neighbor]:
               continue
            elif (distances[neighbor] > distances[index] + 1) \
                    or (distances[neighbor] == distances[index] + 1 and sum_ys[neighbor] > sum_ys[index] + cluster_nodes[index]['height']):
                distances[neighbor] = distances[index] + 1
                sum_ys[neighbor] = sum_ys[index] + cluster_nodes[index]['height']
                pres[neighbor] = index
                heapq.heappush(queue, (100000 * distances[neighbor] + sum_ys[neighbor], neighbor))

        if index in ends:
            ends_num -= 1
    paths =[]
    for i in ends:
        path = []
        curr = i
        while pres[curr] != start:
            curr = pres[curr]
            path.append(curr)
        paths.append({
            'points': path,
            'value': 100000 * distances[i] + sum_ys[i]
        })

    return paths


def search_value(grid, curr_value, curr_min, curr_max, exploring_node_index, axis='x',
                 direction=1, bundle_state='inside', margin=0, start_node=None, nodes=None, edges=None):
    delta_margin = 1
    if direction == 1:
        delta_margin = -1
    # delta_margin = 0
    if start_node is not None:
        start_index = start_node['index']
        curr_edge_index = start_node['next_edge']
        curr_edge = edges[curr_edge_index]
        end_node_index = curr_edge['end']
        end_node = nodes[end_node_index]

        pre_edge_index = start_node['pre_edge']
        pre_edge = edges[pre_edge_index]
        start_pre_node_index = pre_edge['start']
        start_pre_node = nodes[start_pre_node_index]

        next_edge_index = end_node['next_edge']
        next_edge = edges[next_edge_index]
        end_next_node_index = next_edge['end']
        end_next_node = nodes[end_next_node_index]

        forbiden_index = [start_index, end_node['index'], start_pre_node['index'], end_next_node['index']]
    if axis == 'x' and direction == 1:
        min_left = float('inf')
        for column in grid['columns']:
            if column['x'] > curr_value:
                for item in column['items']:
                    if item['state'][exploring_node_index] == bundle_state \
                            and if_intersaction_of_two_range(curr_min, curr_max, item['node_top'], item['node_bottom']):
                        min_left = min(min_left, item['node_left'])
                        if min_left == curr_value:
                            return curr_value
        if start_node is not None:
            while True:
                if start_node['index'] not in forbiden_index and start_node['x'] >= curr_value and curr_min <= start_node['y'] <= curr_max:
                    min_left = min(min_left, start_node['x'] + delta_margin)
                curr_edge_index = start_node['next_edge']
                curr_edge = edges[curr_edge_index]
                end_node_index = curr_edge['end']
                end_node = nodes[end_node_index]
                if end_node['index'] == start_index:
                    break
                start_node = end_node

        return min_left
    elif axis == 'x' and direction == 0:
        max_right = float('-inf')
        for i in range(len(grid['columns'])):
            column = grid['columns'][- i - 1]
            if column['x'] < curr_value:
                for item in column['items']:
                    if item['state'][exploring_node_index] == bundle_state \
                            and if_intersaction_of_two_range(curr_min, curr_max, item['node_top'], item['node_bottom']):
                        max_right = max(max_right, item['node_right'])
                        if max_right == curr_value:
                            return curr_value
        if start_node is not None:
            while True:
                if start_node['index'] not in forbiden_index and start_node['x'] <= curr_value and curr_min <= start_node['y'] <= curr_max:
                    max_right = max(max_right, start_node['x'] + delta_margin)
                curr_edge_index = start_node['next_edge']
                curr_edge = edges[curr_edge_index]
                end_node_index = curr_edge['end']
                end_node = nodes[end_node_index]
                if end_node['index'] == start_index:
                    break
                start_node = end_node
        return max_right
    elif axis == 'y' and direction == 1:
        min_top = float('inf')
        for column in grid['columns']:
            if if_intersaction_of_two_range(curr_min, curr_max, column['left'], column['right']):
                for item in column['items']:
                    if item['state'][exploring_node_index] == bundle_state and item['y'] > curr_value \
                            and if_intersaction_of_two_range(curr_min, curr_max, item['node_left'], item['node_right']):
                        min_top = min(min_top, item['node_top'])
                        if min_top == curr_value:
                            return curr_value
                        break
        if start_node is not None:
            while True:
                if start_node['index'] not in forbiden_index and start_node['y'] >= curr_value and curr_min <= start_node['x'] <= curr_max:
                    min_top = min(min_top, start_node['y'] + delta_margin)
                curr_edge_index = start_node['next_edge']
                curr_edge = edges[curr_edge_index]
                end_node_index = curr_edge['end']
                end_node = nodes[end_node_index]
                if end_node['index'] == start_index:
                    break
                start_node = end_node
        return min_top
    else:
        max_bottom = float('-inf')
        for column in grid['columns']:
            if if_intersaction_of_two_range(curr_min, curr_max, column['left'], column['right']):
                for i in range(len(column['items'])):
                    item = column['items'][-1 - i]
                    if item['state'][exploring_node_index] == bundle_state and item['y'] < curr_value \
                            and if_intersaction_of_two_range(curr_min, curr_max, item['node_left'], item['node_right']):
                        max_bottom = max(max_bottom, item['node_bottom'])
                        if max_bottom == curr_value:
                            return curr_value
                        break
        if start_node is not None:
            while True:
                if start_node['index'] not in forbiden_index and start_node['y'] <= curr_value and curr_min <= start_node['x'] <= curr_max:
                    max_bottom = max(max_bottom, start_node['y'] + delta_margin)
                curr_edge_index = start_node['next_edge']
                curr_edge = edges[curr_edge_index]
                end_node_index = curr_edge['end']
                end_node = nodes[end_node_index]
                if end_node['index'] == start_index:
                    break
                start_node = end_node
        return max_bottom


def angle_process(new_hull, new_edges, grid, exploring_node, expand, bundle_state, margin=5, debug=False, strict=False, must_minus=False):
    if new_hull is None or new_edges is None:
        return None, None
    exploring_node_index = exploring_node['index']
    start_node = new_hull[0]
    while start_node['processed'] == 0:
        curr_edge_index = start_node['next_edge']
        curr_edge = new_edges[curr_edge_index]
        end_node_index = curr_edge['end']
        end_node = new_hull[end_node_index]
        pre_edge_index = start_node['pre_edge']
        pre_edge = new_edges[pre_edge_index]
        start_pre_node_index = pre_edge['start']
        start_pre_node = new_hull[start_pre_node_index]
        pre_pre_edge_index = start_pre_node['pre_edge']
        pre_pre_edge = new_edges[pre_pre_edge_index]

        next_edge_index = end_node['next_edge']
        next_edge = new_edges[next_edge_index]
        end_next_node_index = next_edge['end']
        end_next_node = new_hull[end_next_node_index]

        direction = curr_edge['direction']
        if debug:
            hull_plot1(new_hull, new_edges, start_node)
        # 顺时针旋转的
        if expand:
            if direction == 'top':
                pre_y = start_pre_node['y']
                next_y = end_next_node['y']
                curr_y = start_node['y']

                curr_right = max(start_node['x'], end_node['x'])
                curr_left = min(start_node['x'], end_node['x'])


                curr_min_top = search_value(grid, curr_y, curr_left, curr_right, exploring_node_index,
                                            axis='y', direction=0, bundle_state=bundle_state,
                                            start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if pre_y < curr_y < next_y:
                        if curr_min_top <= pre_y:
                            start_node['y'] = pre_y
                            end_node['y'] = pre_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_top < curr_y and not must_minus:
                            start_node['y'] = curr_min_top
                            end_node['y'] = curr_min_top
                    elif next_y < curr_y < pre_y:
                        if curr_min_top <= next_y:
                            start_node['y'] = next_y
                            end_node['y'] = next_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_top < curr_y and not must_minus:
                            start_node['y'] = curr_min_top
                            end_node['y'] = curr_min_top
                if pre_y <= next_y < curr_y:
                    if curr_min_top <= next_y:
                        start_node['y'] = next_y
                        end_node['y'] = next_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None
                        start_node['processed'] = 0
                        continue
                    elif curr_min_top < curr_y and not must_minus:
                        start_node['y'] = curr_min_top
                        end_node['y'] = curr_min_top
                elif next_y < pre_y < curr_y:
                    if curr_min_top <= pre_y:
                        start_node['y'] = pre_y
                        end_node['y'] = pre_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_top < curr_y and not must_minus:
                        start_node['y'] = curr_min_top
                        end_node['y'] = curr_min_top
            elif direction == 'bottom':
                pre_y = start_pre_node['y']
                next_y = end_next_node['y']
                curr_y = start_node['y']

                curr_right = max(start_node['x'], end_node['x'])
                curr_left = min(start_node['x'], end_node['x'])
                curr_max_bottom = search_value(grid, curr_y, curr_left, curr_right, exploring_node_index,
                                               axis='y', direction=1, bundle_state=bundle_state,
                                               start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if next_y < curr_y < pre_y:
                        if curr_max_bottom >= pre_y:
                            start_node['y'] = pre_y
                            end_node['y'] = pre_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_bottom > curr_y and not must_minus:
                            start_node['y'] = curr_max_bottom
                            end_node['y'] = curr_max_bottom
                    elif pre_y < curr_y < next_y:
                        if curr_max_bottom >= next_y:
                            start_node['y'] = next_y
                            end_node['y'] = next_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_bottom > curr_y and not must_minus:
                            start_node['y'] = curr_max_bottom
                            end_node['y'] = curr_max_bottom
                if curr_y < next_y <= pre_y:
                    if curr_max_bottom >= next_y:
                        start_node['y'] = next_y
                        end_node['y'] = next_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_bottom > curr_y and not must_minus:
                        start_node['y'] = curr_max_bottom
                        end_node['y'] = curr_max_bottom
                elif curr_y < pre_y < next_y:
                    if curr_max_bottom >= pre_y:
                        start_node['y'] = pre_y
                        end_node['y'] = pre_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_bottom > curr_y and not must_minus:
                        start_node['y'] = curr_max_bottom
                        end_node['y'] = curr_max_bottom
            elif direction == 'left':
                pre_x = start_pre_node['x']
                next_x = end_next_node['x']
                curr_x = start_node['x']

                curr_bottom = max(start_node['y'], end_node['y'])
                curr_top = min(start_node['y'], end_node['y'])
                curr_min_left = search_value(grid, curr_x, curr_top, curr_bottom, exploring_node_index,
                                             axis='x', direction=0, bundle_state=bundle_state,
                                             start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if pre_x < curr_x < next_x:
                        if curr_min_left <= pre_x:
                            start_node['x'] = pre_x
                            end_node['x'] = pre_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_left < curr_x and not must_minus:
                            start_node['x'] = curr_min_left
                            end_node['x'] = curr_min_left
                    elif next_x < curr_x < pre_x:
                        if curr_min_left <= next_x:
                            start_node['x'] = next_x
                            end_node['x'] = next_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_left < curr_x and not must_minus:
                            start_node['x'] = curr_min_left
                            end_node['x'] = curr_min_left
                if pre_x <= next_x < curr_x:
                    if curr_min_left <= next_x:
                        start_node['x'] = next_x
                        end_node['x'] = next_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_left < curr_x and not must_minus:
                        start_node['x'] = curr_min_left
                        end_node['x'] = curr_min_left
                elif next_x < pre_x < curr_x:
                    if curr_min_left <= pre_x:
                        start_node['x'] = pre_x
                        end_node['x'] = pre_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_left < curr_x and not must_minus:
                        start_node['x'] = curr_min_left
                        end_node['x'] = curr_min_left
            elif direction == 'right':
                pre_x = start_pre_node['x']
                next_x = end_next_node['x']
                curr_x = start_node['x']

                curr_bottom = max(start_node['y'], end_node['y'])
                curr_top = min(start_node['y'], end_node['y'])
                curr_max_right = search_value(grid, curr_x, curr_top, curr_bottom, exploring_node_index,
                                              axis='x', direction=1, bundle_state=bundle_state,
                                              start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if next_x < curr_x < pre_x:
                        if curr_max_right >= pre_x:
                            start_node['x'] = pre_x
                            end_node['x'] = pre_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_right > curr_x and not must_minus:
                            start_node['x'] = curr_max_right
                            end_node['x'] = curr_max_right
                    elif pre_x < curr_x < next_x:
                        if curr_max_right >= next_x:
                            start_node['x'] = next_x
                            end_node['x'] = next_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_right > curr_x and not must_minus:
                            start_node['x'] = curr_max_right
                            end_node['x'] = curr_max_right
                if curr_x < next_x <= pre_x:
                    if curr_max_right >= next_x:
                        start_node['x'] = next_x
                        end_node['x'] = next_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_right > curr_x and not must_minus:
                        start_node['x'] = curr_max_right
                        end_node['x'] = curr_max_right
                elif curr_x < pre_x < next_x:
                    if curr_max_right >= pre_x:
                        start_node['x'] = pre_x
                        end_node['x'] = pre_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_right > curr_x and not must_minus:
                        start_node['x'] = curr_max_right
                        end_node['x'] = curr_max_right
        else:
            if direction == 'top':
                pre_y = start_pre_node['y']
                next_y = end_next_node['y']
                curr_y = start_node['y']

                curr_right = max(start_node['x'], end_node['x'])
                curr_left = min(start_node['x'], end_node['x'])
                curr_max_top = search_value(grid, curr_y, curr_left, curr_right, exploring_node_index, axis='y',
                                            direction=1, bundle_state=bundle_state,
                                            start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if next_y < curr_y < pre_y:
                        if curr_max_top >= pre_y:
                            start_node['y'] = pre_y
                            end_node['y'] = pre_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_top > curr_y and not must_minus:
                            start_node['y'] = curr_max_top
                            end_node['y'] = curr_max_top
                    elif pre_y < curr_y < next_y:
                        if curr_max_top >= next_y:
                            start_node['y'] = next_y
                            end_node['y'] = next_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_top > curr_y and not must_minus:
                            start_node['y'] = curr_max_top
                            end_node['y'] = curr_max_top
                if curr_y < next_y <= pre_y:
                    if curr_max_top >= next_y:
                        start_node['y'] = next_y
                        end_node['y'] = next_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_top > curr_y and not must_minus:
                        start_node['y'] = curr_max_top
                        end_node['y'] = curr_max_top
                elif curr_y < pre_y < next_y:
                    if curr_max_top >= pre_y:
                        start_node['y'] = pre_y
                        end_node['y'] = pre_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_top > curr_y and not must_minus:
                        start_node['y'] = curr_max_top
                        end_node['y'] = curr_max_top
            elif direction == 'bottom':
                pre_y = start_pre_node['y']
                next_y = end_next_node['y']
                curr_y = start_node['y']

                curr_right = max(start_node['x'], end_node['x'])
                curr_left = min(start_node['x'], end_node['x'])
                curr_min_bottom = search_value(grid, curr_y, curr_left, curr_right, exploring_node_index, axis='y',
                                               direction=0, bundle_state=bundle_state,
                                            start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if pre_y < curr_y < next_y:
                        if curr_min_bottom <= pre_y:
                            start_node['y'] = pre_y
                            end_node['y'] = pre_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_bottom < curr_y and not must_minus:
                            start_node['y'] = curr_min_bottom
                            end_node['y'] = curr_min_bottom
                    elif next_y < curr_y < pre_y:
                        if curr_min_bottom <= next_y:
                            start_node['y'] = next_y
                            end_node['y'] = next_y
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_bottom < curr_y and not must_minus:
                            start_node['y'] = curr_min_bottom
                            end_node['y'] = curr_min_bottom
                if pre_y <= next_y < curr_y:
                    if curr_min_bottom <= next_y:
                        start_node['y'] = next_y
                        end_node['y'] = next_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_bottom < curr_y and not must_minus:
                        start_node['y'] = curr_min_bottom
                        end_node['y'] = curr_min_bottom
                elif next_y < pre_y < curr_y:
                    if curr_min_bottom <= pre_y:
                        start_node['y'] = pre_y
                        end_node['y'] = pre_y
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_bottom < curr_y and not must_minus:
                        start_node['y'] = curr_min_bottom
                        end_node['y'] = curr_min_bottom
            elif direction == 'left':
                pre_x = start_pre_node['x']
                next_x = end_next_node['x']
                curr_x = start_node['x']

                curr_bottom = max(start_node['y'], end_node['y'])
                curr_top = min(start_node['y'], end_node['y'])
                curr_max_left = search_value(grid, curr_x, curr_top, curr_bottom, exploring_node_index, axis='x',
                                             direction=1, bundle_state=bundle_state,
                                            start_node=start_node, nodes=new_hull, edges=new_edges)
                if not strict:
                    if next_x < curr_x < pre_x:
                        if curr_max_left >= pre_x:
                            start_node['x'] = pre_x
                            end_node['x'] = pre_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_left > curr_x and not must_minus:
                            start_node['x'] = curr_max_left
                            end_node['x'] = curr_max_left
                    elif pre_x < curr_x < next_x:
                        if curr_max_left >= next_x:
                            start_node['x'] = next_x
                            end_node['x'] = next_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_max_left > curr_x and not must_minus:
                            start_node['x'] = curr_max_left
                            end_node['x'] = curr_max_left
                if curr_x < next_x <= pre_x:
                    if curr_max_left >= next_x:
                        start_node['x'] = next_x
                        end_node['x'] = next_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_left > curr_x and not must_minus:
                        start_node['x'] = curr_max_left
                        end_node['x'] = curr_max_left
                elif curr_x < pre_x < next_x:
                    if curr_max_left >= pre_x:
                        start_node['x'] = pre_x
                        end_node['x'] = pre_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_max_left > curr_x and not must_minus:
                        start_node['x'] = curr_max_left
                        end_node['x'] = curr_max_left
            elif direction == 'right':
                pre_x = start_pre_node['x']
                next_x = end_next_node['x']
                curr_x = start_node['x']

                curr_bottom = max(start_node['y'], end_node['y'])
                curr_top = min(start_node['y'], end_node['y'])
                curr_min_right = search_value(grid, curr_x, curr_top, curr_bottom, exploring_node_index, axis='x',
                                              direction=0, bundle_state=bundle_state,
                                            start_node=start_node, nodes=new_hull, edges=new_edges)

                if not strict:
                    if pre_x < curr_x < next_x:
                        if curr_min_right <= pre_x:
                            start_node['x'] = pre_x
                            end_node['x'] = pre_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_right < curr_x and not must_minus:
                            start_node['x'] = curr_min_right
                            end_node['x'] = curr_min_right
                    elif next_x < curr_x < pre_x:
                        if curr_min_right <= next_x:
                            start_node['x'] = next_x
                            end_node['x'] = next_x
                            start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                            if start_node['index'] == start_pre_node['index']:
                                start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                                if start_node is None:
                                    return None, None
                            else:
                                temp = delete_duplicate_node(end_node, new_edges, new_hull)
                                if temp is None:
                                    return None, None

                            start_node['processed'] = 0
                            continue
                        elif curr_min_right < curr_x and not must_minus:
                            start_node['x'] = curr_min_right
                            end_node['x'] = curr_min_right
                if pre_x <= next_x < curr_x:
                    if curr_min_right <= next_x:
                        start_node['x'] = next_x
                        end_node['x'] = next_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_right < curr_x and not must_minus:
                        start_node['x'] = curr_min_right
                        end_node['x'] = curr_min_right
                elif next_x < pre_x < curr_x:
                    if curr_min_right <= pre_x:
                        start_node['x'] = pre_x
                        end_node['x'] = pre_x
                        start_node = delete_duplicate_node(start_pre_node, new_edges, new_hull)
                        if start_node is None:
                            return None, None
                        if start_node['index'] == start_pre_node['index']:
                            start_node = delete_duplicate_node(end_node, new_edges, new_hull)
                            if start_node is None:
                                return None, None
                        else:
                            temp = delete_duplicate_node(end_node, new_edges, new_hull)
                            if temp is None:
                                return None, None

                        start_node['processed'] = 0
                        continue
                    elif curr_min_right < curr_x and not must_minus:
                        start_node['x'] = curr_min_right
                        end_node['x'] = curr_min_right

        start_node['processed'] = 1
        start_node = end_node

    hull = []
    edges = []
    while start_node['processed'] != 0:
        start_node['processed'] = 0
        next_edge = new_edges[start_node['next_edge']]
        end_node_index = next_edge['end']
        start_node['index'] = len(hull)
        start_node['next_edge'] = len(edges)
        start_node['pre_edge'] = len(edges) - 1
        next_edge['start'] = len(hull)
        next_edge['end'] = len(hull) + 1
        edges.append(next_edge)
        hull.append(start_node)
        start_node = new_hull[end_node_index]

    hull[0]['pre_edge'] = len(edges) - 1
    hull[-1]['next_edge'] = len(edges) - 1
    edges[-1]['end'] = 0

    return hull, edges


def delete_duplicate_node(start_node, edges, nodes):
    curr_edge_index = start_node['next_edge']
    curr_edge = edges[curr_edge_index]
    end_node_index = curr_edge['end']
    end_node = nodes[end_node_index]
    pre_edge_index = start_node['pre_edge']
    pre_edge = edges[pre_edge_index]
    start_pre_node_index = pre_edge['start']
    start_pre_node = nodes[start_pre_node_index]
    pre_pre_edge_index = start_pre_node['pre_edge']
    pre_pre_edge = edges[pre_pre_edge_index]

    next_edge_index = end_node['next_edge']
    next_edge = edges[next_edge_index]
    end_next_node_index = next_edge['end']
    end_next_node = nodes[end_next_node_index]

    if start_pre_node_index == end_node_index:
        return None

    if start_node['x'] == end_node['x'] and start_node['y'] == end_node['y']:
        if abs(start_pre_node['x'] - end_node['x']) > abs(end_next_node['x'] - end_node['x']) \
                or abs(start_pre_node['y'] - end_node['y']) > abs(end_next_node['y'] - end_node['y']):
            pre_edge['end'] = end_next_node_index
            end_next_node['pre_edge'] = pre_edge_index
        else:
            next_edge['start'] = start_pre_node_index
            start_pre_node['next_edge'] = next_edge_index
        return delete_duplicate_node(start_pre_node, edges, nodes)
    return start_node


def get_max_var_node_group_size(all_nodes, all_edges):
    res = 0
    count = {}
    for node in all_nodes:
        if node['is_var']:
            pre_edge_ids = node['pre']
            next_edge_ids = node['next']
            pre_edges = [all_edges[i] for i in pre_edge_ids]
            next_edges = [all_edges[i] for i in next_edge_ids]
            pre_node_ids = [edge['start'] for edge in pre_edges]
            next_node_ids = [edge['end'] for edge in next_edges]
            pre_node_ids = sorted(pre_node_ids)
            next_node_ids = sorted(next_node_ids)
            pre_key = '$'.join([str(x) for x in pre_node_ids])
            next_key = '$'.join([str(x) for x in next_node_ids])
            var_key = pre_key + '->' + next_key
            if var_key not in count:
                count[var_key] = 0
            count[var_key] += 1
    for key in count:
        if count[key] > res:
            res = count[key]
    return res



import numpy as np
from tempfile import NamedTemporaryFile
import math
from flask import send_file

def normalize(arr):
    if sum(arr) == 0:
        return np.zeros_like(arr)
    return arr / sum(arr)

def create_feature_map_image(rawdata, data_id, shape, var_node_id):
    from PIL import Image
    data = rawdata['node_data']
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
    tmp_img = NamedTemporaryFile(suffix='.png')
    #print('name', tmp_img.name)
    img_path = tmp_img.name#os.path.join(SERVER_ROOT, 'data', 'feature_maps', 'feature_map_{}.png'.format(str(data_id)))
    img.save(img_path)
    return send_file(img_path)

def transform_to_color(img_arr):
    color1 = np.array([103, 0, 31])
    color2 = np.array([5, 48, 97])
    white = np.array([255, 255, 255])
    arr = [-((-x) ** 0.5) if x < 0 else (x ** 0.5) for x in img_arr.reshape(-1)]
    
    return np.array([((-x) * color1 + (1 + x) * white) if x < 0 else (x * color2 + (1 - x) * white) for x in arr], dtype=np.uint8)


def get_network_and_op_groups_from_data(data):
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

    #print('add all leaf node from leaf data')
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

    #print('add all edges connect all leaf nodes')
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

    #print('process the tree of nodes')
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


    #print('process the stack of all edges')
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


    #print('process reverse nodes and edges')
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
                #print('no next')
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
        #if has_feature > 1:
        #    print('two tail')
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
    #print('removing duplicate edges from', len(edges))
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

    #print('remove', duplicate_count, 'duplicate edges from', len(edges))
    return duplicate_count


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



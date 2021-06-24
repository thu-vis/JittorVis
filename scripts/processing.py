from os.path import basename, join
from scripts import config
import numpy as np
import time
import json
import os
import math
from scripts.utils import *


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_time_value_to_iter(value, iter):
    res = []
    start = 0
    for t in iter:
        res.append(value[start: t])
        start = t
    return res


def load_data(path):
    if not os.path.exists(path):
        statistics = {
            'loss': [1 - x * 0.09 for x in range(10)],
            'accuracy': [0.9 * pow(x, 2) / 100 for x in range(10)],
            'precision': [0.8 * pow(x, 3) / 1000 for x in range(10)],
            'recall': [0.7 * pow(x / 10, 0.5) for x in range(10)],
            'iter': [100 * x for x in range(1, 11)],
            'value': {
                'iter': ['loss', 'accuracy', 'precision', 'recall'],
                'time': ['memory']
            }
        }
        
        statistics['memory'] = split_time_value_to_iter([0.5 + 0.2 * math.sin(0.1 * x) for x in range(1000)], statistics['iter'])

        nodes = [{
            'index': 0,
            'title': 'root',
            'children': [i for i in range(1, 31)],
            'parent': -1,
            'size': 300,
            'level': 2,
            'next': [],
            'pre': []
        }]

        for i in range(1, 31):
            nodes.append({
                'index': i,
                'title': 'unit_' + str(i - 1),
                'children': [j for j in range(10 * i + 21, 10 * i + 31)],
                'parent': 0,
                'size': 10,
                'level': 1,
                'next': [] if i == 10 else [10 * i + 31],
                'pre': [] if i == 1 else [10 * i + 20]
            })

        for i in range(1, 31):
            nodes.append({
                'index': 10 * i + 21,
                'title': 'conv_0',
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 22],
                'pre': [] if i == 1 else [10 * i + 20],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 22,
                'title': 'conv_' + str(10 * i - 9),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 20 + j for j in [3, 4]],
                'pre': [10 * i + 21],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 23,
                'title': 'conv_' + str(10 * i - 8),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 20 + j for j in [5, 6]],
                'pre': [10 * i + 22],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 24,
                'title': 'conv_' + str(10 * i - 7),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 27],
                'pre': [10 * i + 22],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 25,
                'title': 'conv_' + str(10 * i - 6),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 28],
                'pre': [10 * i + 23],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 26,
                'title': 'conv_' + str(10 * i - 5),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 20 + j for j in [7, 8]],
                'pre': [10 * i + 23],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 27,
                'title': 'conv_' + str(10 * i - 4),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 29],
                'pre': [10 * i + 20 + j for j in [4, 6]],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 28,
                'title': 'conv_' + str(10 * i - 3),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 30],
                'pre': [10 * i + 20 + j for j in [5, 6]],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 29,
                'title': 'conv_' + str(10 * i - 2),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [10 * i + 30],
                'pre': [10 * i + 27],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
            nodes.append({
                'index': 10 * i + 30,
                'title': 'conv_' + str(10 * i - 1),
                'children': [],
                'parent': i,
                'size': 1,
                'level': 0,
                'next': [] if i == 30 else [10 * i + 31],
                'pre': [10 * i + 20 + j for j in [8, 9]],
                'time_cost': np.random.random(),
                'gpu_id': np.random.random(5)
            })
        for i in range(len(nodes)):
            node = nodes[len(nodes) - i - 1]
            node['gpu_id'] = normalize(node['gpu_id'])
            if i < len(nodes) - 1:
                parent = nodes[node['parent']]
                if 'time_cost' not in parent:
                    parent['time_cost'] = 0
                parent['time_cost'] += node['time_cost']
                if 'gpu_id' not in parent:
                    parent['gpu_id'] = np.zeros(5)
                parent['gpu_id'] += node['gpu_id']
            node['gpu_id'] = node['gpu_id'].tolist()
            
        for i in range(len(nodes)):
            node = nodes[i]
            gpu_id = []
            for j, value in enumerate(node['gpu_id']):
                gpu_id.append([j, value])
            # gpu_id = sorted(gpu_id, key=lambda x:x[1], reverse=True)
            node['gpu_id'] = gpu_id

        network = {
            'tree': nodes,
            'root': 0,
            'size_threshold': 10
        }
        return statistics, network
        # data = {
        #     'statistics': statistics,
        #     'network': network
        # }
        # json.dump(data, open(path, 'w', encoding='utf-8'))
    else:
        statistics = {
            'loss': [1 - x * 0.09 for x in range(10)],
            'accuracy': [0.9 * pow(x, 2) / 100 for x in range(10)],
            'precision': [0.8 * pow(x, 3) / 1000 for x in range(10)],
            'recall': [0.7 * pow(x / 10, 0.5) for x in range(10)],
            'iter': [100 * x for x in range(1, 11)],
            'value': {
                'iter': ['loss', 'accuracy', 'precision', 'recall'],
                'time': ['memory']
            }
        }
        statistics['memory'] = split_time_value_to_iter([0.5 + 0.2 * math.sin(0.1 * x) for x in range(1000)], statistics['iter'])


        data = pickle.load(open(path, 'rb'))
        network, op_groups, schedule = get_network_and_op_groups_from_data(data)
        return statistics, network, op_groups, schedule


def handle_raw_data():
    statistics, network, op_groups, schedule = load_data(config.DATA_PATH)

    res = {
        'statistics': statistics,
        'network': network,
        'op_groups': op_groups,
        'schedule': schedule
    }
    return res

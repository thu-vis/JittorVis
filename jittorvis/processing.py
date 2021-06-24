import math
from .utils import *

def split_time_value_to_iter(value, iter):
    res = []
    start = 0
    for t in iter:
        res.append(value[start: t])
        start = t
    return res

def handle_raw_data(data):
    network, op_groups, schedule = get_network_and_op_groups_from_data(data)

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

    res = {
        'statistics': statistics,
        'network': network,
        'op_groups': op_groups,
        'schedule': schedule
    }
    return res

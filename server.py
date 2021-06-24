from flask import Flask, jsonify
import os
import sys
import json
from scripts import config, processing
from flask import request
from flask import send_file
from scripts.utils import get_hull_of_exploring_nodes, get_exploring_height_and_level, create_feature_map_image
import argparse

SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)

app = Flask(__name__, static_url_path="/static")


@app.route('/api/process', methods=['GET'])
def process_data():
    data = processing.handle_raw_data()
    return jsonify({
        'status': 'success',
        'data': data
    })


@app.route('/api/get_exploring_node_hull', methods=['GET', 'POST'])
def get_exploring_node_hull():
    all_nodes = json.loads(request.form['all_nodes'])
    nodes = json.loads(request.form['nodes'])
    all_edges = json.loads(request.form['all_edges'])
    exploring_nodes = json.loads(request.form['exploring_nodes'])
    update_layout = request.form['update_layout'] == 'true'

    hull_points, nodes = get_hull_of_exploring_nodes(all_nodes, all_edges, exploring_nodes, nodes, update_layout)
    return jsonify({
        'status': 'success',
        'hulls': hull_points,
        'nodes': nodes,
        'update_layout': update_layout
    })


@app.route('/api/get_exploring_node_level_and_height', methods=['GET', 'POST'])
def get_exploring_node_level_and_height():
    all_nodes = json.loads(request.form['all_nodes'])
    exploring_nodes = json.loads(request.form['exploring_nodes'])

    exploring_nodes = get_exploring_height_and_level(all_nodes, exploring_nodes)
    return jsonify({
        'status': 'success',
        'exploring_nodes': exploring_nodes
    })


@app.route('/api/get_image', methods=['GET', 'POST'])
def get_image():
    data_id = int(request.args['data_id'])
    data_set = request.args['dataset']
    image_type = request.args['image_type']
    shape = request.args['shape']
    var_node_id = int(request.args['var_node_id'])
    filename = data_set.split('.')[0].replace('=', '').replace(' ', '_').replace('=', '')
    if image_type == "point_cloud":
        return send_file(os.path.join(SERVER_ROOT, 'data', 'point_cloud.jpg'))
    else:
        image_path = create_feature_map_image(data_id, shape, var_node_id)
        return send_file(os.path.join(SERVER_ROOT, image_path))


def run_server(port, host, data_path, threaded, debug):
    config.DATA_PATH = data_path
    config.PORT = port
    app.run(port=port, host=host, threaded=threaded, debug=debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default=config.DATA_PATH)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--threaded", type=bool, default=True)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    run_server(args.port, args.host, args.data_path, args.threaded, args.debug)
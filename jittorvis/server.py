from flask import Flask, jsonify, request, send_file
import os
import sys
import json
import pickle
import logging
import argparse
import jittor as jt
from multiprocessing import Process
from .processing import *
from .utils import get_exploring_height_and_level, create_feature_map_image

log = logging.getLogger('werkzeug')
log.disabled = True
SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)
p = None
app = Flask(__name__, static_url_path="/static")

@app.route('/api/process', methods=['GET'])
def process_data():
    global rawdata
    data = handle_raw_data(rawdata)
    return jsonify({
        'status': 'success',
        'data': data
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
    if image_type == "point_cloud":
        return send_file(os.path.join(SERVER_ROOT, 'data', 'point_cloud.jpg'))
    else:
        global rawdata
        image_path = create_feature_map_image(rawdata, data_id, shape, var_node_id)
        return image_path

def run_server(data, host = None, port = 5005):
    app.logger.disabled = True
    global rawdata
    if type(data) == str:
        if not os.path.exists(data):
            raise Exception("The path does not exist.")
        rawdata = pickle.load(open(data, 'rb'))
    else:
        rawdata = data
    app.run(port=port, host=host, threaded=True, debug=False)

def visualize(input, model, host = None, port = 5005):
    with jt.flag_scope(trace_py_var=2, trace_var_data=1):
        output = model(input)
        output.sync()
        data = jt.dump_trace_data()
        jt.clear_trace_data()
    run(data, host, port)

def run(data, host = None, port = 5005):
    global p
    if p is not None:
        p.terminate()
    p = Process(target = run_server, args=(data, host, port))
    p.start()
    print("JittorVis Start.")

def status():
    global p
    return p is not None

def stop():
    global p
    p.terminate()
    p = None
    print("JittorVis Stop.")

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='.')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise Exception("The path does not exist.")
    global rawdata
    rawdata = pickle.load(open(args.data_path, 'rb'))
    app.run(port=args.port, host=args.host, threaded=True, debug=False)


if __name__ == '__main__':
    main()
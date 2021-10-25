#! /usr/bin/python3
import os
import pickle
import argparse
from flask import Flask, jsonify, request, send_file, render_template
from data.dataCtrler import dataCtrler
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/api/allData', methods=['GET'])
def allData():
    alldata = {
        "network": dataCtrler.getBranchTree(),
        "statistic": dataCtrler.getStatisticData()
    }
    return jsonify(alldata)

@app.route('/api/featureInfo', methods=['POST'])
def featureInfo():
    branchID = request.json['branch']
    data = dataCtrler.getBranchNodeOutput(branchID)
    return jsonify(data)

@app.route('/api/feature', methods=["GET"])
def feature():
    leafID = int(request.args['leafID'])
    index = int(request.args['index'])
    return send_file(dataCtrler.getFeature(leafID, index))

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/home/zhaowei/JittorModels/Jittor-Image-Models/resnet26.pkl')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise Exception("The path does not exist.")
    rawdata = pickle.load(open(args.data_path, 'rb'))
    dataCtrler.process(rawdata)
    app.run(port=args.port, host=args.host, threaded=True, debug=False)

if __name__ == "__main__":
    main()
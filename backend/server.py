#! /usr/bin/python3
import os
import pickle
import argparse
from flask import Flask, jsonify
from data.dataCtrler import dataCtrler
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/api/allData', methods=['GET'])
def allData():
    global rawdata
    # data init
    dataCtrler.process(rawdata)
    alldata = {
        "network": dataCtrler.getBranchTree(),
        "statistic": dataCtrler.getStatisticData()
    }
    return jsonify(alldata)

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/home/zhaowei/JittorModels/Jittor-Image-Models/resnet26.pkl')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise Exception("The path does not exist.")
    global rawdata
    rawdata = pickle.load(open(args.data_path, 'rb'))
    app.run(port=args.port, host=args.host, threaded=True, debug=False)

if __name__ == "__main__":
    main()
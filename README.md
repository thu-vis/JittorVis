# JittorVis
==========

Visual analysis of deep learning model.

## Requirement
* Python 3.7
* Flask==1.1.2
* numpy==1.19.4
* itsdangerous==1.1.0
* Jinja2==2.11.2
* MarkupSafe==1.1.1
* Werkzeug==1.0.1


## Installation
1. You can create a new python3.7 environment by anaconda: conda create -n jittorvis python=3.7.
2. You can install the requirements in the requirements.txt, for example: pip install -r requirements.txt.


## Run
1. You can just start the server by "python server.py --port=8888 --data_path=example.pkl --host=0.0.0.0 --threaded=True --debug=False".
2. Example data "simple_model.pkl" and "simple_model_train.pkl" in "data".
3. Visit host:\<port\>/static/index.html.

## Interaction
1. Click "Statistics" and "ExecuteOpInfo" to switch the Statistics view and the ExecuteOpInfo view.
2. Statistics view:
    1) Switch the played statistic value by the top-right menu.
    2) Scale by click.
3. Network view:
    1) Drag the total panel to adapt its position and scale.
    2) Right-click on the network node to explore its detail information.
    3) Click on the network node to expand it, to explore its point cloud and feature map.
    4) Click on the top-right plus button of each network node to explore its children.

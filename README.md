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
pip install jittorvis


## Usage
from jittorvis import server
server.run('test.pkl', host='0.0.0.0', port=5005)
Open 'http://localhost:5005/static/index.html' in your browser.

## Interaction
1. Statistics view:
    1) Switch the played statistic value by the top-right menu.
    2) Scale by click.
2. Network view:
    1) Drag the total panel to adapt its position and scale.
    2) Right-click on the network node to explore its detail information.
    3) Click on the network node to expand it, to explore its point cloud and feature map.
    4) Click on the top-right plus button of each network node to explore its children.

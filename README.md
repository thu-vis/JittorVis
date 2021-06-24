# JittorVis: Visual understanding of deep learning model

![Image of JittorVis](https://github.com/swordsbird/JittorVis/raw/main/overview.png)

Deep neural networks have achieved breakthrough performance in many tasks such as image classification. However, the development of high-quality deep models typically relies on a substantial amount of trial-and-error, as there is still no clear understanding of when and why a deep model works. Also, the complexity of the deep neural network architecture brings difficulties to debugging and modifying the model. The visualization of the computational graph of the deep neural network at different levels can bring users a deeper understanding of the computational graph from the whole to the part, so as to debug and modify the model more effectively.

JittorVis provides the visualization and tooling needed for machine learning experimentation:
* Observe the hierarchical structure of the model computational graph 
* Visualizing the model computational graph in the different level (ops and layers)
* Profiling JittorVis programs

Features to be supported in the future:
* Tracking and visualizing metrics such as loss and accuracy
* Viewing linechart of weights, biases, or other tensors as they change over time
* And much more

Related Links:
*  [Jittor Github](https://github.com/jittor/jittor/)
*  [Jittor Website](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor Tutorials](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor Models](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor Documents](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Gitee](https://gitee.com/jittor/jittor)

## Pip install
pip install jittorvis

## Usage
```python
from jittorvis import server
server.run('test.pkl', host='0.0.0.0', port=5005)
```
Then open the link 'http://localhost:5005/static/index.html' in your browser.

## Interaction

![Image of Navigation View](https://github.com/swordsbird/JittorVis/raw/main/navigation.png =400x600)
1. Navigation view:
    1) Each leaf node represents a computational node in the computational graph.
    2) Click one intermediate node to selected its computational nodes.


![Image of Graph View](https://github.com/swordsbird/JittorVis/raw/main/graph.png =400x600)
2. Graph structure view:
    1) Each rectangle represents a computational node, and each link represents data flows among computational nodes.
    2) Drag the total panel to adapt its position and scale.
    3) Click on the network node to expand it, to explore its point cloud and feature map.
    4) Click on the top-right plus button of each network node to explore its children.
    5) Right-click on the network node to explore its detail information.

## Requirement
* Python 3.7
* Flask==1.1.2
* numpy==1.19.4
* itsdangerous==1.1.0
* Jinja2==2.11.2
* MarkupSafe==1.1.1
* Werkzeug==1.0.1

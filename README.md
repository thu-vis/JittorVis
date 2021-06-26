# JittorVis: Visual understanding of deep learning model

<img src="https://github.com/swordsbird/JittorVis/raw/main/assets/logo.png" alt="Drawing" width="500px" />

**JittorVis** is an open-source library for understanding the inner workings of [**Jittor**](https://github.com/jittor/jittor/) models by visually illustrating their dataflow graphs.

Deep neural networks have achieved breakthrough performance in many tasks such as image recognition, detection, segmentation, generation, etc. However, the development of high-quality deep models typically relies on a substantial amount of trial and error, as there is still no clear understanding of when and why a deep model works. Also, the complexity of the deep neural network architecture brings difficulties to debugging and modifying the model. **JittorVis** facilitates the visualization of the dataflow graph of the deep neural network at different levels, which brings users a deeper understanding of the dataflow graph from the whole to the part to debug and modify the model more effectively.

![Image of JittorVis](https://github.com/swordsbird/JittorVis/raw/main/assets/overview.png)

**JittorVis** provides the visualization and tooling needed for machine learning experimentation:
* Observing the hierarchical structure of the model dataflow graph
* Visualizing the dataflow model graph in the different level (ops and layers)
* Profiling JittorVis programs

Features to be supported in the future:
* Tracking and visualizing metrics such as loss and accuracy
* Viewing line chart of weights, biases, or other tensors as they change over time
* And much more

Related Links:
*  [Jittor](https://github.com/jittor/jittor/)
*  [Jittor Website](https://cg.cs.tsinghua.edu.cn/jittor/)

## Installation

JittorVis need python version >= 3.7.
```
pip install jittorvis
or
pip3 install jittorvis
```

## Usage

[Download link for test.pkl](https://cloud.tsinghua.edu.cn/lib/246262e5-8d6d-4b94-bd29-3b33a4442fef/file/test.zip?dl=1)
```python
from jittorvis import server
server.run('test.pkl', host='0.0.0.0', port=5005)
# JittorVis start.
server.stop()
# JittorVis stop.
```
Then open the link 'http://localhost:5005/static/index.html' in your browser.

## Visualization

JittorVis contains three main views, statistics view, navigation view, and graph structure view.

1. **Statistics view**:

    The statistics view provides statistics information for the deep neuron network, such as loss and accuracy

2. **Navigation view**:

    The graph structure view can visualize a hierarchical structure of a Jittor model, enabling exploration of the model. Each leaf node represents a computational node in the dataflow graph.
    * Click one intermediate node to selected its computational nodes.

<img src="https://github.com/swordsbird/JittorVis/raw/main/assets/navigation.png" alt="Drawing" width="400px" />


3. **Graph structure view**:

    The graph structure view can visualize a Jittor graph, enabling inspection of the Jittor model. In the graph structure view, each rectangle represents a computational node, and each link represents data flows among computational nodes. The graph structure view has the following interactions:
    * Drag the total panel to adapt its position and scale.
    * Click on the network node to expand it, to explore its point cloud and feature map.
    * Click on the top-right plus button of each network node to explore its children.
    * Right-click on the network node to explore its detail information.

<img src="https://github.com/swordsbird/JittorVis/raw/main/assets/graph.png" alt="Drawing" width="600px" />

## Citation

**Towards Better Analysis of Deep Convolutional Neural Networks**
```
@article {
    liu2017convolutional,
    author={Liu, Mengchen and Shi, Jiaxin and Li, Zhen and Li, Chongxuan and Zhu, Jun and Liu, Shixia},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    title={Towards Better Analysis of Deep Convolutional Neural Networks},
    year={2017},
    volume={23},
    number={1},
    pages={91-100}
}
```

**Analyzing the Training Processes of Deep Generative Models**
```
@article {
    liu2018generative,
    author={Liu, Mengchen and Shi, Jiaxin and Cao, Kelei and Zhu, Jun and Liu, Shixia},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    title={Analyzing the Training Processes of Deep Generative Models},
    year={2018},
    volume={24},
    number={1},
    pages={77-87}
}
```

**Analyzing the Noise Robustness of Deep Neural Networks**
```
@article {
    cao2021robustness,
    author={Cao, Kelei and Liu, Mengchen and Su, Hang and Wu, Jing and Zhu, Jun and Liu, Shixia},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    title={Analyzing the Noise Robustness of Deep Neural Networks},
    year={2021},
    volume={27},
    number={7},
    pages={3289-3304}
}
```

## The Team

JittorVis is currently maintained by the THUVIS Group. If you are also interested in JittorVis and want to improve it, Please [**join us!**](http://shixialiu.com/)

## License

JittorVis is Apache 2.0 licensed, as found in the LICENSE.txt file.

# JittorVis: Visual understanding of deep learning model

![Image of JittorVis](https://github.com/thu-vis/JittorVis/blob/main/assets/overview.png)

**JittorVis** is an open-source library for understanding the inner workings of [**Jittor**](https://github.com/jittor/jittor/) models by visually illustrating their dataflow graphs.

Deep neural networks have achieved breakthrough performance in many tasks such as image recognition, detection, segmentation, generation, etc. However, the development of high-quality deep models typically relies on a substantial amount of trial and error, as there is still no clear understanding of when and why a deep model works. Also, the complexity of the deep neural network architecture brings difficulties to debugging and modifying the model. **JittorVis** facilitates the visualization of the dataflow graph of the deep neural network at different levels, which brings users a deeper understanding of the dataflow graph from the whole to the part to debug and modify the model more effectively.

**JittorVis** provides the visualization and tooling needed for machine learning experimentation:
* Displaying the hierarchical structure of the model dataflow graph
* Visualizing the dataflow graph at different levels (ops and layers)
* Profiling Jittor programs

Features to be supported in the future:
* Tracking and visualizing metrics such as loss and accuracy
* Viewing line charts of weights, biases, or other tensors as they change over time
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

## How to Develop
1. run backend
```bash
cd backend
python server.py
```
2. run frontend
```bash
cd frontend
yarn
yarn start
```
3. generate doc
```bash
# frontend
cd frontend
yarn styleguide:build
# backend
cd ..
pdoc backend/ -o doc --html --force
```

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

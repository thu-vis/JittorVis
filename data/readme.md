# 计图可视化

可视化文件包含：

1. simple_model.pkl 一个简单的双层神经网络推理计算图
2. simple_model_train.pkl 一个简单的双层神经网络训练计算图
3. resnet.pkl resnet18推理计算图
4. resnet_train.pkl resnet18训练计算图
5. 下载地址  https\://cloud.tsinghua.edu.cn/d/ec491e4bf0bf4688baf9/

## 计算图格式说明

pkl文件中包含一个python的dict，dict的格式如下：

```
主文件data包含的key有：
1. data['node_data'] -> Dict[int, NodeData]
   保存了所有结点的信息，并且提供了id到NodeData类型的映射
2. data['execute_op_info'] -> Dict[int, ExecuteOpInfo]
   保存了所有算子的运行信息，int类型的key从小到达和计算图实际运算的拓扑顺序一致
   
NodeData类型包含的key有：
1. node_data['id'] -> int
   代表这个节点的id
2. node_data['inputs'] -> List[int]
   代表这个节点的输入id
3. node_data['outputs'] -> List[int]
   代表这个节点的输出id
4. node_data['stacks'] -> List[Stack]
   代表这个节点的栈信息，0号元素代表最外层的栈（or group）
   a. stack['name'] -> str
      栈名称，可用做group 名称(e.g. conv1)
   b. stack['type'] -> str
      栈类型（e.g. Conv2d）
   c. stack['file_path'] -> str
      调用这个栈的python代码文件名
   d. stack['lineno'] -> int
      调用这个栈的python代码文件的行号
5. node_data['attrs'] -> Dict
   存放了这个节点的一些属性
   a. attrs['is_var'] -> str
      '1' 代表这是个变量，‘0’代表这是个算子
   b. attrs['shape'] -> str
      变量的形状, e.g. '[10,10,]'
   c. attrs['ndim'] -> str
      变量的维度大小, e.g. '2'
   d. attrs['dtype']
      变量的数据类型, e.g. 'float32'
   e. attrs['name']
      变量的名称, e.g. 'layer1.conv1.w'
      或者是算子的名称, e.g. 'binary.add'
   f. attrs['grad_op_id']
      该属性仅算子有效，代表了这个算子是哪个算子的反向传播算子。
      有可能多个反向传播算子的grad_op_id指向同一个前向算子，
      反向传播算子还有可能进一步成为其他算子的反向传播（高阶导数）
      

ExecuteOpInfo类型包含的key有：
1. execute_op_info["fused_ops"] -> List[int]
   代表该次融合执行的算子的编号,需要注意的是,一个算子可能出现在多个融合执行组里.
2. execute_op_info["jit_key"] -> str
   这个融合算子的key
3. execute_op_info["file_path"] -> str
   这个融合算子动态编译的c++代码文件路径
4. execute_op_info["attrs"] -> Dict
   这个融合算子的一些属性

```

## 可视化数据生成脚本

代码中的四个测试分别生成了以上4个数据文件

```python
import unittest
import jittor as jt
import numpy as np
from jittor import Module
from jittor.models import resnet
import pickle

f32 = jt.float32

def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)


def relu(x):
    return jt.maximum(x, 0.0)
Relu = jt.make_module(relu)

class Model(Module):
    def __init__(self, input_size):
        self.linear1 = Linear(input_size, 10)
        self.relu1 = Relu()
        self.linear2 = Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.w = (jt.random((in_features, out_features))-0.5) / in_features**0.5
        self.b = jt.random((out_features,))-0.5 if bias else None
    def execute(self, x):
        x = matmul(x, self.w)
        if self.b is not None: 
            return x+self.b
        return x


class TestTraceVar(unittest.TestCase):
    def test_simple_model(self):
        with jt.flag_scope(trace_py_var=2):

            model = Model(input_size=1)
            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            y.sync()


            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open("/tmp/simple_model.pkl", "wb") as f:
                pickle.dump(data, f)
            # for k,v in data["node_data"].items():
            #     print(v)
            # for k,v in data["execute_op_info"].items():
            #     print(v)
            # print(data)

    def test_simple_model_train(self):
        with jt.flag_scope(trace_py_var=2):
            
            model = Model(input_size=1)
            opt = jt.optim.SGD(model.parameters(), 0.1)

            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            opt.step(y**2)
            jt.sync_all()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open("/tmp/simple_model_train.pkl", "wb") as f:
                pickle.dump(data, f)
            # for k,v in data["node_data"].items():
            #     print(v)
            # for k,v in data["execute_op_info"].items():
            #     print(v)
            # print(data)

    def test_resnet(self):
        with jt.flag_scope(trace_py_var=2):

            resnet18 = resnet.Resnet18()
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)
            y.sync()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open("/tmp/resnet.pkl", "wb") as f:
                pickle.dump(data, f)

    def test_resnet_train(self):
        with jt.flag_scope(trace_py_var=2):

            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)

            opt.step(y**2)
            jt.sync_all()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open("/tmp/resnet_train.pkl", "wb") as f:
                pickle.dump(data, f)


if __name__ == "__main__":
    unittest.main()

```



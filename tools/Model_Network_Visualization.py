"""
作者：汪永杰
编写时间：2023/5/5 2:14
其他：无
"""
# import torch
# from torchviz import make_dot
# from models.yolo import Model  # 这里的Model是YOLOv5的网络结构
#
#
# def visualize_layer_weights(model, layer_name):
#     layer = model.named_parameters(layer_name)
#     weight_tensor = layer.next()[1].detach()
#     make_dot(weight_tensor, params=dict(model.named_parameters())).render(layer_name, format="svg")
#
#
# def visualize_network(model, input_shape):
#     sample_input = torch.randn(input_shape)  # 示例输入的形状
#     make_dot(model(sample_input), params=dict(model.named_parameters())).render("network_graph", format="svg")
#
#
# model = Model()
# visualize_layer_weights(model, "model.layers.3.conv.weight")
# visualize_network(model, (1, 3, 640, 640))

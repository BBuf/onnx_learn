import torch


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()
        self.mean = torch.randn(2, 3, 4, 5)
        self.std = torch.randn(2, 3, 4, 5)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = '../model/just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
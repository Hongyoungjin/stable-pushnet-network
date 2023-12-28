import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineNormalization(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CosineNormalization, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.eta = nn.Parameter(torch.empty((1), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.eta, 10)

    def forward(self, input):
        input_norm = F.normalize(input, p=2, dim=1)  # N X D
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # D X 2
        return torch.mul(F.linear(input_norm, weight_norm), self.eta)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features is not None
        )

if __name__ == '__main__':
    model = GQCNN()

    print('model structure: \n', model)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} \n")
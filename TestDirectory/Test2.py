import pretrainedmodels
import torchvision
import torch
import archs
import arch_test.AGSandSA
from torch.nn.parameter import Parameter
from torch import nn

def main():
    resnet = arch_test.AGSandSA.sa_resnet50(pretrained=True)
    res = list(resnet.children())
    #print(net)
    #print(resnet)

    input = torch.rand(1, 3, 224, 160)
    for i, model in enumerate(res):
        print(model)
        input = model(input)
        print(input.shape)

if __name__ == '__main__':
    main()
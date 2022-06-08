## https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The below imports are commented out so that any library incompatibility issues
are eliminated during validation when project1_model.pt is used.
"""

# import numpy as np
# import random
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils import mixup_data
# import settings

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        # lam = np.random.beta(alpha, alpha)
        lam = float(torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample())
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, per_img_std = False):
        super(ResNet, self).__init__()
        self.per_img_std = per_img_std
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        
        numOfChannels = 32
        for layerNum in range(2,5):
            numOfChannels = numOfChannels * 2
            expression = "self.layer{} = self._make_layer(block, {}, {}, stride=2)".format(
                layerNum, numOfChannels, num_blocks[layerNum - 1])
            exec(expression)
        
        self.linear = nn.Linear(32*2*2*2, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup_hidden = False,  mixup_alpha = 0.0, layer_mix=None):
            
        global out
        global a_self
        a_self = self
        
        if mixup_hidden == True:
            if layer_mix == None:
                # layer_mix = random.randint(0,2) 
                layer_mix = torch.randint(0,3,(1,1)).item() 
            
            out = x
            
            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            
            out = F.relu(self.bn1(self.conv1(x)))
            
            for layerNum in range(1,5):
                
                expression = "out = a_self.layer{}(out)".format(layerNum)
                
                exec(expression,globals())
                
                if layer_mix == layerNum:
                    out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
                    
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            
            lam = torch.tensor(lam).cuda()
            lam = lam.repeat(y_a.size())
            return out, y_a, y_b, lam
        
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            for layerNum in range(1,5):
                expression = "out = a_self.layer{}(out)".format(layerNum)
                exec(expression, globals())
                
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        

def resnet18(num_classes=10, dropout = False, per_img_std = False):
    """Constructs a ResNet-20 model.
    """
    blocksList = [2 for i in range(4)]
    model = ResNet(BasicBlock, blocksList, num_classes, per_img_std = per_img_std)
    return model

def resnet20(num_classes=10, dropout = False, per_img_std = False):
    """Constructs a ResNet-20 model.
    """
    model = ResNet(BasicBlock, [3,3,3,3], num_classes, per_img_std = per_img_std)
    return model

def project1_model():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

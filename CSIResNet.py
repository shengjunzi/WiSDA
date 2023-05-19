import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def conv3x3(in_channel, out_channel, stride=1, padding=1):
    return nn.Conv2D(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias_attr=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2D(in_channel, out_channel, kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    """
    50层以下的，用BasicBlock
    结构是2个3x3的卷积单元 1个block中channel数不变
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        # 这里传入stride是因为从第二个stage开始，第一个block中的第一层卷积stride是2 map大小发生了改变
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果没有传入downsample 则不发生下采样 identify=x,传入downsample 则发生下采样。是为了前后数据形状相同，做相加运算
        # downsample是在make_layer中使用的，发生条件是 stride=1或者in_channel != in_channel*self.expansion
        if self.downsample is not None:
            identify = self.downsample(x)
        
        out += identify
        out = self.relu(out)

        return out


class BottleNeck(nn.Layer):
    """
    50层以上的resnet使用BottleNeck
    结构是 三个卷积单元：1x1  3x3  1x1
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        
        self.conv1 = conv1x1(in_channel, out_channel)
        self.bn1 = norm_layer(out_channel)
        
        # 这里传入stride，是因为从第二个stage开始，第一个block中的第2层卷积stride是2，map大小发生了改变
        self.conv2 = conv3x3(out_channel, out_channel, stride) 
        self.bn2 = norm_layer(out_channel)

        self.conv3 = conv1x1(out_channel, out_channel*self.expansion)
        self.bn3 = norm_layer(out_channel*self.expansion)
        
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # 如果没有传入downsample 则不发生下采样 identify=x,传入downsample 则发生下采样。是为了前后数据形状相同，做相加运算
        # downsample是在make_layer中使用的，发生条件是 stride=1或者in_channel != in_channel*self.expansion
        if self.downsample is not None:
            identify = self.downsample(x)
        
        out += identify
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    
    def __init__(self, block, layers,norm_layer=None,):
        super(ResNet, self).__init__()
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2D
        self.norm_layer = norm_layer

        self.in_channel = 64
        # [N, 3, 224, 224] -> [N, 64, 56, 56]
        self.conv1 = nn.Conv2D(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        #-----------------重点部分 4个stage----------------------------
        # 从第二个stage开始，第一个block
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # ------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        #self.fc = nn.Linear(512*block.expansion, num_class)

        # 参数初始化
        

    def _make_layer(self, block, out_channel, blocks, stride=1):

        """实现stage，stage由block堆叠而成"""

        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.in_channel != out_channel*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel*block.expansion, stride),
                norm_layer(out_channel*block.expansion)
            )
        
        layers = []
        # 核心部分
        # 这里是stage中的第一个block，从第二个stage开始，第一个block和后面的处理不一样，stride=2，map大小发生了变化
        layers.append(block(self.in_channel, out_channel, stride, downsample, norm_layer))

        self.in_channel = out_channel*block.expansion
        for _ in range(1, blocks): # 这里是从第二个block开始，block是 列表[3, 8, 36, 3]中的某一个元素，表示1个stage中的block数量
            layers.append(block(self.in_channel, out_channel, norm_layer=norm_layer))
        
        return nn.Sequential(*layers) # 返回组建好的stage


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x =x.flatten(1)
        #x = self.fc(x)

        return x
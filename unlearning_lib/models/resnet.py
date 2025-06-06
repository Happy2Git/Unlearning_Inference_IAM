'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

# Define the new model by subclassing and modifying the existing ResNet model


class ResNetBackbone(nn.Module):
    def __init__(self, original_resnet, num_classes):
        super(ResNetBackbone, self).__init__()
        # Remove the original ResNet's fully connected layer
        self.feature_extractor = copy.deepcopy(nn.Sequential(
            *(list(original_resnet.children())[:-1])))

        # Create a new fully connected layer for the new number of classes
        # The number of input features to this layer is determined by the output of the last feature extraction layer
        num_features = list(
            original_resnet.named_children())[-1][1].in_features
        self.new_linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Use the feature extractor layers
        x = self.feature_extractor(x)
        x = F.avg_pool2d(x, 4)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        # Apply the new fully connected layer
        x = self.new_linear(x)
        return x

class LocationBackbone(nn.Module):
    def __init__(self, original_model, num_classes):
        super(LocationBackbone, self).__init__()

        # 定义特征提取层序列
        self.feature_extractor = copy.deepcopy(nn.Sequential(
            *(list(original_model.children())[:-1])))
        num_features = list(
            original_model.named_children())[-1][1].in_features
        self.new_linear = nn.Linear(num_features, num_classes)

    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.new_linear(x)
        return x

class TexasBackbone(nn.Module):
    def __init__(self, original_model, num_classes):
        super(TexasBackbone, self).__init__()

        # 定义特征提取层序列
        self.feature_extractor = copy.deepcopy(nn.Sequential(
            *(list(original_model.children())[:-1])))
        num_features = list(
            original_model.named_children())[-1][1].in_features
        self.new_linear = nn.Linear(num_features, num_classes)

    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.new_linear(x)
        return x

class PurchaseBackbone(nn.Module):
    def __init__(self, original_model, num_classes):
        super(PurchaseBackbone, self).__init__()

        # 定义特征提取层序列
        self.feature_extractor = copy.deepcopy(nn.Sequential(
            *(list(original_model.children())[:-1])))
        num_features = list(
            original_model.named_children())[-1][1].in_features
        self.new_linear = nn.Linear(num_features, num_classes)

    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.new_linear(x)
        return x
    
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 替代BatchNorm的LayerNorm实现 - 适用于CNN
class LayerNormCNN(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNormCNN, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x的形状: [N, C, H, W]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized + self.bias

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = LayerNormCNN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = LayerNormCNN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.norm1 = LayerNormCNN(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.norm2 = LayerNormCNN(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.norm3 = LayerNormCNN(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 简单的特征处理层 - 不进行复杂转换
class SimpleFeatureProcessor(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleFeatureProcessor, self).__init__()
        self.norm = LayerNormCNN(feature_dim)
        
    def forward(self, x):
        return self.norm(x)

# 基本的本地分类器
class LocalClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LocalClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 修改TierAwareClientModel类
class TierAwareClientModel(nn.Module):
    def __init__(self, num_classes=10, tier=1):
        super(TierAwareClientModel, self).__init__()
        self.tier = tier
        
        # 基础层(共享层) - 所有客户端完全一致
        self.shared_base = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Layer1
            self._make_layer(BasicBlock, 16, 16, 2),
            # Layer2前1/3
            self._make_layer(BasicBlock, 16, 32, 1, stride=2)
        )
        
        # 共享层输出通道数固定为32
        self.output_channels = 32
        
        # 根据tier调整个性化路径深度和宽度
        self._create_personalized_path()
        
        # 本地分类器
        local_dim = 64  # 个性化路径输出维度
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(local_dim, num_classes)
        )
    
    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        
        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))
        
        return nn.Sequential(*layers)
    
    def _create_personalized_path(self):
        # 个性化路径 - 根据tier调整
        if self.tier <= 2:  # 高性能设备
            # 更深的个性化路径
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 32, 2),
                self._make_layer(BasicBlock, 32, 64, 2, stride=2)
            )
        elif self.tier <= 4:  # 中等性能设备
            # 中等深度个性化路径
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 32, 1),
                self._make_layer(BasicBlock, 32, 64, 1, stride=2)
            )
        else:  # 低性能设备
            # 浅层个性化路径
            self.personalized_path = nn.Sequential(
                self._make_layer(BasicBlock, 32, 64, 1, stride=2)
            )
    
    def forward(self, x):
        # 共享基础层（返回给服务器）
        shared_features = self.shared_base(x)
        
        # 个性化路径
        personal_features = self.personalized_path(shared_features)
        
        # 本地分类
        local_logits = self.local_classifier(personal_features)
        
        return local_logits, shared_features, personal_features
    
    def get_shared_params(self):
        """获取共享层参数"""
        shared_params = {}
        for name, param in self.named_parameters():
            if 'shared_base' in name:
                shared_params[name] = param
        return shared_params
    
    def get_personalized_params(self):
        """获取个性化路径参数"""
        personalized_params = {}
        for name, param in self.named_parameters():
            if 'personalized_path' in name or 'local_classifier' in name:
                personalized_params[name] = param
        return personalized_params

# 修改EnhancedServerModel类
class EnhancedServerModel(nn.Module):
    def __init__(self, feature_dim=128):
        super(EnhancedServerModel, self).__init__()
        
        # 直接继续ResNet结构，从Layer2剩余部分开始
        self.server_layers = nn.Sequential(
            # Layer2剩余部分
            self._make_layer(BasicBlock, 32, 32, 2),
            # Layer3
            self._make_layer(BasicBlock, 32, 64, 2, stride=2)
        )
        
        # 特征转换，保持输出维度一致
        self.feature_transform = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        
        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 确保输入是4D张量
        if x.dim() == 2:
            batch_size, channels = x.shape
            x = x.view(batch_size, channels, 1, 1)
        
        # 服务器层处理
        x = self.server_layers(x)
        
        # 特征转换
        features = self.feature_transform(x)
        
        return features
        
    def get_params(self):
        """获取所有参数"""
        return {name: param for name, param in self.named_parameters()}

# 简化的全局分类器
class ImprovedGlobalClassifier(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super(ImprovedGlobalClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
    
    def get_params(self):
        """获取所有参数"""
        return {name: param for name, param in self.named_parameters()}
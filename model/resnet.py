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

# TierHFL-ResNet客户端模型
class TierHFLClientModel(nn.Module):
    def __init__(self, block, layers, num_classes=10, tier=1):
        super(TierHFLClientModel, self).__init__()
        self.inplanes = 16
        self.tier = tier
        
        # 基础层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = LayerNormCNN(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # 根据tier决定包含哪些层
        self.layers = nn.ModuleList()
        planes = [16, 16, 32, 32, 64, 64]
        strides = [1, 1, 2, 1, 2, 1]
        
        # 根据tier级别决定要实现的层数
        self.active_layers = min(7 - tier, 6)  # tier 1有6层，tier 7有0层
        
        for i in range(self.active_layers):
            if i == 0:
                self.layers.append(self._make_layer(block, planes[i], layers[i], stride=strides[i]))
            else:
                self.inplanes = planes[i-1] * block.expansion
                self.layers.append(self._make_layer(block, planes[i], layers[i], stride=strides[i]))
        
        # 本地分类器 - 对数据分布的个性化适应
        self.classifier = None
        if self.active_layers == 0:  # tier 7
            self.classifier = LocalClassifier(16, num_classes)
        elif self.active_layers == 1:  # tier 6
            self.classifier = LocalClassifier(16 * block.expansion, num_classes)
        elif self.active_layers == 2:  # tier 5
            self.classifier = LocalClassifier(16 * block.expansion, num_classes)
        elif self.active_layers == 3:  # tier 4
            self.classifier = LocalClassifier(32 * block.expansion, num_classes)
        elif self.active_layers == 4:  # tier 3
            self.classifier = LocalClassifier(32 * block.expansion, num_classes)
        elif self.active_layers == 5:  # tier 2
            self.classifier = LocalClassifier(64 * block.expansion, num_classes)
        elif self.active_layers == 6:  # tier 1
            self.classifier = LocalClassifier(64 * block.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, LayerNormCNN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                LayerNormCNN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 基础层处理
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        # 根据tier执行相应的层
        extracted_features = x
        for i, layer in enumerate(self.layers):
            extracted_features = layer(extracted_features)
        
        # 应用本地分类器
        local_logits = self.classifier(extracted_features) if self.classifier else None
        
        return local_logits, extracted_features
    
    # 区分共享参数和个性化参数
    def get_shared_params(self):
        shared_params = {}
        for name, param in self.named_parameters():
            # 排除分类器参数
            if 'classifier' not in name:
                shared_params[name] = param
        return shared_params
    
    def get_personalized_params(self):
        personalized_params = {}
        for name, param in self.named_parameters():
            if 'classifier' in name:
                personalized_params[name] = param
        return personalized_params

# TierHFL-ResNet服务器模型
class TierHFLServerModel(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(TierHFLServerModel, self).__init__()
        self.num_classes = num_classes
        
        # 服务器端处理模块 - 针对不同tier
        self.tier_modules = nn.ModuleDict()
        planes = [16, 16, 32, 32, 64, 64]
        strides = [1, 1, 2, 1, 2, 1]
        
        # 创建每个tier对应的处理模块
        for tier in range(1, 8):
            tier_str = f'tier_{tier}'
            self.tier_modules[tier_str] = nn.ModuleList()
            
            # 根据tier确定起始层和inplanes
            start_layer = max(0, 7 - tier)
            inplanes = 16  # 默认起始通道数
            
            # 调整inplanes基于tier
            if tier == 7:
                inplanes = 16
            elif tier == 6:
                inplanes = 16 * block.expansion
            elif tier == 5:
                inplanes = 16 * block.expansion
            elif tier == 4:
                inplanes = 32 * block.expansion
            elif tier == 3:
                inplanes = 32 * block.expansion
            elif tier == 2:
                inplanes = 64 * block.expansion
            elif tier == 1:
                inplanes = 64 * block.expansion
                
            # 创建该tier需要的层
            for i in range(start_layer, 6):
                if i == start_layer:
                    this_layer = self._make_layer(block, planes[i], layers[i], 
                                                 stride=strides[i], inplanes=inplanes)
                else:
                    inplanes = planes[i-1] * block.expansion
                    this_layer = self._make_layer(block, planes[i], layers[i], 
                                                 stride=strides[i], inplanes=inplanes)
                self.tier_modules[tier_str].append(this_layer)
        
        # 全局分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_classifier = nn.Linear(64 * block.expansion, num_classes)
        
        # 简化的特征处理
        self.feature_processor = SimpleFeatureProcessor(64 * block.expansion)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, LayerNormCNN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, inplanes=None):
        inplanes = self.inplanes if inplanes is None else inplanes
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                LayerNormCNN(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, tier=1):
        # 获取对应tier的处理模块
        tier_str = f'tier_{tier}'
        
        # 处理特征
        for layer in self.tier_modules[tier_str]:
            x = layer(x)
        
        # 应用特征处理
        x = self.feature_processor(x)
        
        # 全局分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.global_classifier(x)
        
        return logits, x  # 返回分类结果和特征

# 创建客户端模型函数
def create_tierhfl_client_model(tier, num_classes=10, model_type='resnet56'):
    if model_type == 'resnet56':
        return TierHFLClientModel(BasicBlock, [3, 3, 3, 3, 3, 3], num_classes, tier)
    elif model_type == 'resnet110':
        return TierHFLClientModel(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes, tier)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# 创建服务器模型函数
def create_tierhfl_server_model(num_classes=10, model_type='resnet56'):
    if model_type == 'resnet56':
        return TierHFLServerModel(BasicBlock, [3, 3, 3, 3, 3, 3], num_classes)
    elif model_type == 'resnet110':
        return TierHFLServerModel(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
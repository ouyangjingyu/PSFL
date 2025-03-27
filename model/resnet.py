
import logging

import torch
import torch.nn as nn

import torch.nn.functional as F



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 构建更高效的分类器
# class ComplexClassifier(nn.Module):
#     def __init__(self, in_features, num_classes):
#         super(ComplexClassifier, self).__init__()
#         # # 根据 in_features 调整中间层维度
#         # mid_features = min(128, max(64, in_features))  # 限制中间层大小
        
#         self.fc1 = nn.Linear(in_features, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_classes)
        
#         self.dropout = nn.Dropout(0.5)
#         # print(f"Classifier dimensions: in={in_features}, hidden1=128, hidden2=64, out={num_classes}")

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# 在model/resnet.py中修改ComplexClassifier类

# 在 model/resnet.py 中添加

class LayerNormClassifier(nn.Module):
    """使用LayerNorm的分类器，适用于异质性数据环境"""
    def __init__(self, in_features, num_classes, is_local=False):
        super(LayerNormClassifier, self).__init__()
        # 使用LayerNorm替代BatchNorm - 不依赖批次统计量
        self.norm_input = nn.LayerNorm(in_features)
        
        # 客户端本地分类器和全局分类器使用不同的隐藏层大小
        hidden1 = 128 if not is_local else 96
        hidden2 = 64 if not is_local else 48
        
        self.fc1 = nn.Linear(in_features, hidden1)
        self.norm1 = nn.LayerNorm(hidden1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)
        
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        # 本地分类器使用更高的Dropout以适应局部数据
        # 全局分类器使用较低的Dropout提高泛化性
        self.dropout = nn.Dropout(0.6 if is_local else 0.3)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        # 使用更好的初始化方法
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)  # 输出层使用Xavier初始化
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.norm_input(x)
        
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# 简化的分类器，适用于低tier客户端
class SimpleClassifier(nn.Module):
    """简化的分类器，减少参数量，适用于低计算能力设备"""
    def __init__(self, in_features, num_classes):
        super(SimpleClassifier, self).__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features, num_classes)
        
        # 初始化
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 创建分类器的函数
def create_classifier(in_features, num_classes, tier=None, is_global=False):
    """
    创建适合不同场景的分类器
    
    Args:
        in_features: 输入特征维度
        num_classes: 分类数
        tier: 客户端tier级别，None表示全局分类器
        is_global: 是否为全局分类器
        
    Returns:
        适合指定场景的分类器实例
    """
    if is_global:
        # 全局分类器始终使用LayerNormClassifier
        return LayerNormClassifier(in_features, num_classes, is_local=False)
    
    # 客户端本地分类器根据tier选择不同复杂度
    if tier is not None and tier >= 5:
        # 低tier客户端使用简化分类器
        return SimpleClassifier(in_features, num_classes)
    else:
        # 高tier客户端使用更复杂的分类器
        return LayerNormClassifier(in_features, num_classes, is_local=True)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, fedavg_base=False, tier=7, local_loss=False, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.local_loss = local_loss
        self.fedavg_base = fedavg_base
        self.local_v2 = False
        if kwargs:
            self.local_v2 = kwargs.get('local_v2', False)

        self.tier = tier  
 
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # initialization is defined here:https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)  # init: kaiming_uniform
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # tier=1的时候构造的客户端模型层数越多，tier=7时构造的层数越少
        if self.tier == 6 or self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer1 = self._make_layer(block, 16, layers[0])
        if self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1:# or self.local_v2:
             self.layer2 = self._make_layer(block, 16, layers[1])
        if self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        if self.tier == 3 or self.tier == 2 or self.tier == 1:# or self.local_v2:
             self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
        if self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
        if self.tier == 1:# or self.local_v2:
             self.layer6 = self._make_layer(block, 64, layers[5], stride=1)

        # 标准化输入维度
        if self.local_loss == True:
            # 为所有tier级别添加平均池化层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 标准化特征维度，设置为与tier=1相同（64 * block.expansion）
            final_channels = 64 * block.expansion  # 通常是256
            
            # 根据tier确定当前输出通道数
            if self.tier == 1 or self.tier == 2:
                in_features = 64 * block.expansion  # 已经是标准维度
            elif self.tier == 3 or self.tier == 4:
                in_features = 32 * block.expansion  # 需要从128投影到256
            elif self.tier == 5 or self.tier == 6:
                in_features = 16 * block.expansion  # 需要从64投影到256
            elif self.tier == 7:
                in_features = 16  # 需要从16投影到256

            # 添加改进的投影层（使用LayerNorm替代BatchNorm）
            if in_features == final_channels:
                self.projection = nn.Sequential(
                    nn.Identity(),
                    nn.LayerNorm(final_channels)
                )
            else:
                self.projection = nn.Sequential(
                    nn.Linear(in_features, final_channels),
                    nn.LayerNorm(final_channels),
                    nn.ReLU(inplace=True)
                )

            # 创建适合当前tier的本地分类器
            self.classifier = create_classifier(final_channels, num_classes, tier=self.tier, is_global=False)


        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播，每个tier级别都会返回(logits, features)元组
        """
        # Tier 1：客户端有所有层
        if self.tier == 1:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            
            extracted_features = x
            
            # 确保每个tier都有分类器
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)  # 对于tier 1，这将是个Identity操作
            logits = self.classifier(x_proj)
            
            return logits, extracted_features
        
        # Tier 2：客户端到layer5
        elif self.tier == 2:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            extracted_features = x
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features  
        
        # Tier 3：客户端到layer4
        elif self.tier == 3: 
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            extracted_features = x
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features
        
        # Tier 4：客户端到layer3
        elif self.tier == 4:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            extracted_features = x
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features
        
        # Tier 5：客户端到layer2
        elif self.tier == 5:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            extracted_features = x
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features
        
        # Tier 6：客户端到layer1
        elif self.tier == 6:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            extracted_features = x  
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features
        
        # Tier 7：客户端只有基础层
        elif self.tier == 7:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            extracted_features = x  
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features          

            
class ResNet_server(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, tier=5, local_loss=False, **kwargs):
        super(ResNet_server, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.local_loss = local_loss
        
        self.tier = tier

        # 根据tier设置正确的输入通道数（与客户端输出匹配）
        if tier == 7:  # 客户端只有基础层
            self.inplanes = 16
        elif tier == 6:  # 客户端到layer1
            self.inplanes = 16 * block.expansion  # 64
        elif tier == 5:  # 客户端到layer2
            self.inplanes = 16 * block.expansion  # 64
        elif tier == 4:  # 客户端到layer3
            self.inplanes = 32 * block.expansion  # 128
        elif tier == 3:  # 客户端到layer4
            self.inplanes = 32 * block.expansion  # 128
        elif tier == 2:  # 客户端到layer5
            self.inplanes = 64 * block.expansion  # 256
        elif tier == 1:  # 客户端到layer6
            self.inplanes = 64 * block.expansion  # 256

        # 重要：确保所有tier级别都有avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        
        # 根据不同的 tier 初始化服务器端模型
        if self.tier == 7:  # 客户端只有基础层
            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 16, layers[1])
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
            self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)
            
        elif self.tier == 6:  # 客户端到 layer1
            self.layer2 = self._make_layer(block, 16, layers[1])
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
            self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)
            
        elif self.tier == 5:  # 客户端到 layer2
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
            self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)

        elif self.tier == 4:  # 客户端到 layer3
            self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
            self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)
        
        elif self.tier == 3:  # 客户端到 layer4
            self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)
            
        elif self.tier == 2:  # 客户端到 layer5
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)
        
        # 所有 tier 最终都是通过 layer6 输出，输出通道为 64 * block.expansion        
        # 为所有tier添加改进的投影层
        in_features = self.inplanes
        final_channels = 64 * block.expansion  # 256

        # 使用LayerNorm的投影层
        if in_features == final_channels:
            self.projection = nn.Sequential(
                nn.Identity(),
                nn.LayerNorm(final_channels)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(in_features, final_channels),
                nn.LayerNorm(final_channels),
                nn.ReLU(inplace=True)
            )

        # self.classifier = create_classifier(final_channels, num_classes, is_global=True)
        
        # 所有 tier 使用相同的分类器结构，保证聚合时参数维度一致
        # self.classifier = ComplexClassifier(final_channels, num_classes)

        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                          self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        """
        前向传播，返回特征而不是分类结果
        """
        # Tier 1 处理
        if self.tier == 1:
            # Tier 1 直接返回输入特征
            features = x
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)  # 使用投影层进行特征映射
            
            return features
        
        # Tier 2：服务器只有 layer6
        elif self.tier == 2:
            features = self.layer6(x)
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)  # 如果输入特征已经是所需维度，Identity层不进行转换
            
            return features
        
        # Tier 3：服务器有 layer5, layer6
        elif self.tier == 3:
            features = self.layer5(x)
            features = self.layer6(features)
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
            
            return features
            
        # Tier 4：服务器有 layer4, layer5, layer6
        elif self.tier == 4:
            features = self.layer4(x)
            features = self.layer5(features)
            features = self.layer6(features)
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
            
            return features
            
        # Tier 5：服务器有 layer3, layer4, layer5, layer6
        elif self.tier == 5:
            features = self.layer3(x)
            features = self.layer4(features)
            features = self.layer5(features)
            features = self.layer6(features)
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
            
            return features

        # Tier 6：服务器有 layer2, layer3, layer4, layer5, layer6
        elif self.tier == 6:
            features = self.layer2(x)
            features = self.layer3(features)
            features = self.layer4(features)
            features = self.layer5(features)
            features = self.layer6(features)
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
            
            return features
            
        # Tier 7：服务器有所有层
        elif self.tier == 7:
            features = self.layer1(x)
            features = self.layer2(features)
            features = self.layer3(features)
            features = self.layer4(features)
            features = self.layer5(features)
            features = self.layer6(features)
            
            # 提取特征，不进行分类
            if len(features.shape) > 2:  # 如果是卷积特征
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
            
            return features

def resnet56_server(c, pretrained=False, path=None, tier=5, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [6, 6, 6], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
    

def resnet56_server_tier(c, pretrained=False, path=None, tier=5, **kwargs):
    """
    Constructs a ResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [6, 6, 6], num_classes=c, tier=tier, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
    
    


''' SFL model, my first resnet18 model '''

class Baseblock_SFL(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock_SFL, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output
 
    


def resnet110_SFL_local_tier_7(classes, tier=5, **kwargs):
    """
    构造ResNet-110的拆分学习客户端和服务器模型
    
    Args:
        classes: 分类数量
        tier: 层次级别(1-7)，决定客户端和服务器端模型的划分
        **kwargs: 额外参数
        
    Returns:
        net_glob_client: 客户端模型
        net_glob_server: 服务器端模型
    """
    # 从kwargs中提取local_loss参数，如果不存在则默认为True
    local_loss = kwargs.pop('local_loss', True) if 'local_loss' in kwargs else True
    if tier == 1:
        # 客户端包含所有层，服务器只有分类器
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
    elif tier == 2:
        # 客户端到layer5，服务器包含layer6
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
    elif tier == 3:
        # 客户端到layer4，服务器包含layer5和layer6
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 0, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 6, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
    elif tier == 4:
        # 客户端到layer3，服务器包含layer4、layer5和layer6
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 0, 0, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 6, 6, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
    elif tier == 5:
        # 客户端到layer2，服务器包含layer3、layer4、layer5和layer6
        net_glob_client = ResNet(Bottleneck, [6, 6, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 6, 6, 6, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
    elif tier == 6:
        # 客户端到layer1，服务器包含layer2、layer3、layer4、layer5和layer6
        net_glob_client = ResNet(Bottleneck, [6, 0, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 6, 6, 6, 6, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
    elif tier == 7:
        # 客户端只有基础层，服务器包含所有层
        net_glob_client = ResNet(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)
        net_glob_server = ResNet_server(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes=classes, tier=tier, local_loss=local_loss, **kwargs)

    return net_glob_client, net_glob_server

def resnet56_SFL_local_tier_7(classes, tier=5, **kwargs):
    """
    构造ResNet-56的拆分学习客户端和服务器模型
    
    Args:
        classes: 分类数量
        tier: 层次级别(1-7)，决定客户端和服务器端模型的划分
        **kwargs: 额外参数
        
    Returns:
        net_glob_client: 客户端模型
        net_glob_server: 服务器端模型
    """
    if tier == 1:
        # 客户端包含所有层，服务器只有分类器
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 3, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs)
    elif tier == 2:
        # 客户端到layer5，服务器包含layer6
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 3, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs)
    elif tier == 3:
        # 客户端到layer4，服务器包含layer5和layer6
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 0, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 3, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs)
    elif tier == 4:
        # 客户端到layer3，服务器包含layer4、layer5和layer6
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 0, 0, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 3, 3, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs)
    elif tier == 5:
        # 客户端到layer2，服务器包含layer3、layer4、layer5和layer6
        net_glob_client = ResNet(Bottleneck, [3, 3, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs)
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 3, 3, 3, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs)
    elif tier == 6:
        # 客户端到layer1，服务器包含layer2、layer3、layer4、layer5和layer6
        net_glob_client = ResNet(Bottleneck, [3, 0, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 3, 3, 3, 3, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs)
    elif tier == 7:
        # 客户端只有基础层，服务器包含所有层
        net_glob_client = ResNet(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes=classes, tier=tier, local_loss=True, **kwargs)
        net_glob_server = ResNet_server(Bottleneck, [3, 3, 3, 3, 3, 3], num_classes=classes, tier=tier, local_loss=True, **kwargs)

    return net_glob_client, net_glob_server
def resnet56_base(classes, tier=1, **kwargs):
    """ResNet-56 base model.

    Args:
      classes: Number of classes.
      tier: Tier of the model.
      **kwargs: Additional keyword arguments.
    
    Returns:
      A ResNet-56 model.
    """
    net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
    return net_glob_client

def resnet110_base(classes, tier=1, **kwargs):
    """ResNet-110 base model.

    Args:
      classes: Number of classes.
      tier: Tier of the model.
      **kwargs: Additional keyword arguments.
    
    Returns:
      A ResNet-110 model.
    """
    net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
    return net_glob_client

def resnet110_SFL_fedavg_base(classes, tier=1, **kwargs):
    net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
    return net_glob_client
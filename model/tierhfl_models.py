import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_base import BasicBlock, Bottleneck, ResNetBase

# 修改TierHFLClientModel类，确保与ResNet结构一致
class TierHFLClientModel(nn.Module):
    """TierHFL客户端双路径模型 - 优化版"""
    def __init__(self, base_model='resnet56', num_classes=10, tier=1):
        super(TierHFLClientModel, self).__init__()
        self.tier = tier
        self.num_classes = num_classes
        self.inplanes = 16
        
        # 确定基础架构配置
        if base_model == 'resnet56':
            layers = [9, 9, 9]  # 总共9x6+2=56层
            block = BasicBlock
        elif base_model == 'resnet110':
            layers = [18, 18, 18]  # 总共18x6+2=110层
            block = BasicBlock
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 共享基础层 - 设计为ResNet前部分
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建layer1和部分layer2 (共享层)
        self.layer1 = self._make_layer(block, 16, layers[0])
        # 只使用layer2的一部分，作为共享层的结束点
        shared_blocks = max(1, layers[1] // 3)  # 使用layer2的1/3块作为共享层
        self.layer2_shared = self._make_layer(block, 32, shared_blocks, stride=2)
        
        # 提取特征维度，作为服务器模型输入
        self.output_channels = 32
        
        # 个性化特征路径 - 根据tier调整复杂度
        self.inplanes = 32  # 重置inplanes为layer2_shared的输出通道数
        local_layers = self._adjust_local_path_for_tier(layers, tier)
        self.local_path = self._make_local_path(32, local_layers, block)
        
        # 本地分类器
        self.local_classifier = self._create_local_classifier()
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """构建ResNet层"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _adjust_local_path_for_tier(self, layers, tier):
        """根据Tier调整个性化路径深度"""
        # 余下的layer2块数
        remaining_layer2 = max(1, layers[1] - layers[1]//3)
        
        if tier == 1:
            return [remaining_layer2, max(1, layers[2]//2)]  # 高性能设备使用较多层
        elif tier == 2:
            return [remaining_layer2//2, max(1, layers[2]//3)]  # 中高性能减少层数
        elif tier == 3:
            return [1, max(1, layers[2]//4)]  # 中性能进一步减少
        else:
            return [1, 1]  # 低性能设备使用最少层数
    
    def _make_local_path(self, in_channels, layers, block):
        """构建个性化特征路径"""
        modules = []
        channels = in_channels
        
        # 添加剩余的layer2块
        if layers[0] > 0:
            for _ in range(layers[0]):
                modules.append(block(channels, channels))
            
        # 添加layer3块
        if layers[1] > 0:
            # 第一个layer3块使用stride=2
            downsample = nn.Sequential(
                nn.Conv2d(channels, 64 * block.expansion,
                         kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64 * block.expansion)
            )
            modules.append(block(channels, 64, stride=2, downsample=downsample))
            channels = 64 * block.expansion
            
            # 添加剩余layer3块
            for _ in range(1, layers[1]):
                modules.append(block(channels, 64))
        
        # 如果没有任何模块，添加一个简单的卷积层
        if not modules:
            return nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        
        return nn.Sequential(*modules)
    
    def _create_local_classifier(self):
        """创建本地分类器"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 共享基础层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_base = self.layer2_shared(x)
        
        # 个性化特征路径
        local_features = self.local_path(x_base)
        
        # 本地分类
        local_logits = self.local_classifier(local_features)
        
        return local_logits, x_base, local_features

# 修改TierHFLServerModel类，补全ResNet结构
class TierHFLServerModel(nn.Module):
    """TierHFL服务器特征处理模型 - 优化版"""
    def __init__(self, base_model='resnet56', feature_dim=128, in_channels=32, **kwargs):
        super(TierHFLServerModel, self).__init__()
        self.feature_dim = feature_dim
        
        # 确定基础架构配置
        if base_model == 'resnet56':
            layers = [9, 9, 9]
            block = BasicBlock
        elif base_model == 'resnet110':
            layers = [18, 18, 18]
            block = BasicBlock
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 初始化inplanes为客户端layer2_shared的输出通道
        self.inplanes = in_channels
        
        # 继续构建ResNet结构: 剩余的layer2和完整的layer3
        # 计算剩余的layer2块数
        remaining_layer2 = max(1, layers[1] - layers[1]//3)
        
        # 构建服务器部分网络
        self.layer2_server = self._make_layer(block, 32, remaining_layer2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # 特征转换层，将ResNet特征映射到固定维度
        self.feature_transform = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64 * block.expansion, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """构建ResNet层"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播，处理不同维度的输入"""
        # 确保输入是4D的
        if x.dim() == 2:
            # 如果是[batch_size, channels]，重塑为[batch_size, channels, 1, 1]
            batch_size, channels = x.shape
            x = x.view(batch_size, channels, 1, 1)
        
        # 服务器ResNet部分
        x = self.layer2_server(x)
        x = self.layer3(x)
        
        # 特征转换
        features = self.feature_transform(x)
        
        return features

# 修改全局分类器，增加经验回放功能
class TierHFLGlobalClassifier(nn.Module):
    """TierHFL全局分类器 - 带经验回放功能"""
    def __init__(self, feature_dim=128, num_classes=10, buffer_size=1000):
        super(TierHFLGlobalClassifier, self).__init__()
        self.feature_dim = feature_dim  # 添加记录特征维度
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 经验回放缓冲区
        self.feature_buffer = []
        self.label_buffer = []
        self.buffer_size = buffer_size
        self.max_per_class = buffer_size // num_classes
        self.num_classes = num_classes
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 检查输入维度
        if x.size(-1) != self.feature_dim:
            # 打印警告
            print(f"警告: 特征维度不匹配! 期望 {self.feature_dim}, 得到 {x.size(-1)}")
            # 尝试调整维度
            if len(x.shape) == 3 and x.size(1) == self.feature_dim:
                x = x.view(x.size(0), self.feature_dim)
            elif len(x.shape) == 2 and x.size(0) == self.feature_dim:
                x = x.view(1, self.feature_dim)
            else:
                # 如果无法调整，打印错误
                raise ValueError(f"无法调整特征维度: {x.shape} 到 {self.feature_dim}")
                
        return self.classifier(x)
    
    def update_buffer(self, features, labels, device=None):
        """更新特征缓冲区，保持类别平衡，确保特征维度正确"""
        # 确保数据在CPU上
        features = features.detach().cpu()
        labels = labels.cpu() if torch.is_tensor(labels) else labels
        
        # 确保特征维度正确
        if features.size(-1) != self.feature_dim:
            print(f"警告: 缓冲区特征维度不匹配! 期望 {self.feature_dim}, 得到 {features.size(-1)}")
            # 如果是4D特征（带通道），做特殊处理
            if len(features.shape) == 4:
                # 可能是卷积特征，展平它们
                features = features.view(features.size(0), -1)
                # 如果仍然不匹配，跳过存储
                if features.size(-1) != self.feature_dim:
                    print(f"特征维度调整后仍不匹配，跳过缓冲: {features.shape}")
                    return
            else:
                print(f"特征维度不匹配，跳过缓冲")
                return
        
        # 按类别统计当前缓冲区样本数
        class_counts = {}
        for i, label in enumerate(self.label_buffer):
            label_item = label.item() if hasattr(label, 'item') else label
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
        
        # 添加新样本到缓冲区
        for i in range(len(labels)):
            label = labels[i].item() if hasattr(labels[i], 'item') else labels[i]
            
            # 如果该类别样本数未达到上限，添加到缓冲区
            if class_counts.get(label, 0) < self.max_per_class:
                self.feature_buffer.append(features[i])
                self.label_buffer.append(label)
                class_counts[label] = class_counts.get(label, 0) + 1
        
        # 如果缓冲区超过大小限制，随机移除样本
        if len(self.feature_buffer) > self.buffer_size:
            # 随机选择要保留的索引
            keep_indices = torch.randperm(len(self.feature_buffer))[:self.buffer_size]
            
            # 更新缓冲区
            self.feature_buffer = [self.feature_buffer[i] for i in keep_indices]
            self.label_buffer = [self.label_buffer[i] for i in keep_indices]
    
    def sample_from_buffer(self, batch_size=64, device=None):
        """从缓冲区采样数据，确保特征维度正确"""
        if not self.feature_buffer:
            return None, None
        
        # 确定采样大小
        sample_size = min(batch_size, len(self.feature_buffer))
        
        # 随机选择索引
        indices = torch.randperm(len(self.feature_buffer))[:sample_size]
        
        # 提取特征和标签
        sampled_features = torch.stack([self.feature_buffer[i] for i in indices])
        sampled_labels = torch.tensor([self.label_buffer[i] for i in indices])
        
        # 确保特征维度正确
        if sampled_features.size(-1) != self.feature_dim:
            print(f"警告: 采样特征维度不匹配! 期望 {self.feature_dim}, 得到 {sampled_features.size(-1)}")
            return None, None
        
        # 如果指定了设备，将数据移到该设备
        if device:
            sampled_features = sampled_features.to(device)
            sampled_labels = sampled_labels.to(device)
        
        return sampled_features, sampled_labels
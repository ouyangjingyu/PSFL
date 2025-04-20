import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_base import BasicBlock, Bottleneck, ResNetBase

class TierHFLClientModel(nn.Module):
    """TierHFL客户端双路径模型 - 简化版（移除global_path）"""
    def __init__(self, base_model='resnet56', num_classes=10, tier=1):
        super(TierHFLClientModel, self).__init__()
        self.tier = tier
        self.num_classes = num_classes
        self.inplanes = 16  # 添加这一行初始化inplanes属性
        
        # 确定基础架构配置
        if base_model == 'resnet56':
            layers = [9, 9, 9]  # 总共9x6+2=56层
            block = BasicBlock
        elif base_model == 'resnet110':
            layers = [18, 18, 18]  # 总共18x6+2=110层
            block = BasicBlock
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 共享基础层 - 所有Tier统一结构和输出
        # 初始卷积层单独定义，不放入Sequential中
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 然后分别构建各个层
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1]//3, stride=2)  # 减少深度但保留降采样
        
        # 全局特征适配层 - 统一输出通道为64
        self.global_adapter = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # 统一空间维度
        )
        
        # 个性化特征路径 - 根据tier调整复杂度
        self.inplanes = 32  # 重置inplanes为layer2的输出通道数
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
        if tier == 1:
            return [0, 0, max(1, layers[2]//2)]  # 完整深度的一半
        elif tier == 2:
            return [0, 0, max(1, layers[2]//3)]  # 1/3深度
        elif tier == 3:
            return [0, 0, max(1, layers[2]//4)]  # 1/4深度
        else:
            return [0, 0, 1]  # 最小深度
    
    def _make_local_path(self, in_channels, layers, block):
        """构建个性化特征路径"""
        modules = []
        
        # 只处理第三层
        if layers[2] > 0:
            stride = 2  # 第三层的stride
            out_channels = 64
            
            # 添加残差块
            downsample = None
            if stride != 1 or in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * block.expansion,
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )
            
            modules.append(block(in_channels, out_channels, stride, downsample))
            in_channels = out_channels * block.expansion
            
            # 添加剩余块
            for _ in range(1, layers[2]):
                modules.append(block(in_channels, out_channels))
        
        # 如果没有模块，添加一个简单的卷积层
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
        x_base = self.layer2(x)
        
        # 全局特征 - 简单适配，不执行复杂处理
        global_features = self.global_adapter(x_base)
        
        # 个性化特征路径
        local_features = self.local_path(x_base)
        
        # 本地分类
        local_logits = self.local_classifier(local_features)
        
        return local_logits, global_features, local_features


class TierHFLServerModel(nn.Module):
    """TierHFL服务器特征处理模型 - 适配简化客户端"""
    def __init__(self, base_model='resnet56', feature_dim=128, in_channels_list=None, **kwargs):
        super(TierHFLServerModel, self).__init__()
        self.feature_dim = feature_dim
        
        # 固定输入通道为64（与客户端全局特征通道匹配）
        self.in_channels = 64
        
        # 特征处理网络 - 简化版，专注于特征转换而非复杂处理
        self.feature_extractor = nn.Sequential(
            # 添加通道注意力增强特征质量
            self._make_channel_attention(self.in_channels),
            
            # 特征转换层
            nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            # 空间降维
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 特征归一化
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_channel_attention(self, channels):
        """创建通道注意力模块"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取和转换
        features = self.feature_extractor(x)
        
        # 特征归一化
        normalized_features = self.layer_norm(features)
        
        return normalized_features


class TierHFLGlobalClassifier(nn.Module):
    """TierHFL全局分类器"""
    def __init__(self, feature_dim=128, num_classes=10):
        super(TierHFLGlobalClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
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
        return self.classifier(x)
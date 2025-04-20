import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_base import BasicBlock, Bottleneck, ResNetBase

class TierHFLClientModel(nn.Module):
    """TierHFL客户端双路径模型"""
    def __init__(self, base_model='resnet56', num_classes=10, tier=1):
        super(TierHFLClientModel, self).__init__()
        self.tier = tier
        self.num_classes = num_classes
        
        # 确定基础架构配置
        if base_model == 'resnet56':
            layers = [9, 9, 9]  # 总共9x6+2=56层
            block = BasicBlock
        elif base_model == 'resnet110':
            layers = [18, 18, 18]  # 总共18x6+2=110层
            block = BasicBlock
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 共享基础层 - 所有Tier统一
        self.shared_base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 计算输出通道基于Tier级别
        tier_channels = self._calculate_tier_channels(tier)
        
        # 1. 通用特征路径 - 较浅网络，结果会上传到服务器
        global_layers = self._adjust_layers_for_tier(layers, tier)
        self.global_path = self._make_global_path(16, global_layers, block)
        
        # 2. 个性化特征路径 - 根据设备能力调整复杂度
        local_layers = self._adjust_layers_for_tier(layers, tier, is_local=True)
        self.local_path = self._make_local_path(16, local_layers, block)
        

        # 计算该tier输出的特征维度
        self.out_channels = self._calculate_tier_channels(tier)
        # 本地分类器
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.out_channels, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _calculate_tier_channels(self, tier):
        """根据Tier级别计算特征通道数"""
        if tier == 1:  # 最强设备
            return 64
        elif tier == 2:
            return 48
        elif tier == 3:
            return 32
        else:  # tier >= 4，最弱设备
            return 16
    
    def _adjust_layers_for_tier(self, layers, tier, is_local=False):
        """根据Tier调整网络深度"""
        if is_local:  # 个性化路径深度
            if tier == 1:
                return layers  # 完整深度
            elif tier == 2:
                return [max(2, l//2) for l in layers]  # 半深度
            elif tier == 3:
                return [max(1, l//3) for l in layers]  # 1/3深度
            else:
                return [1, 1, 1]  # 最小深度
        else:  # 全局路径深度
            if tier == 1 or tier == 2:
                return [2, 2, 0]  # 高性能设备，轻量全局路径
            elif tier == 3:
                return [2, 1, 0]  # 中等性能
            else:
                return [1, 0, 0]  # 低性能设备，极简全局路径
    
    def _make_global_path(self, in_channels, layers, block):
        """构建全局特征路径"""
        strides = [1, 2, 2]
        channels = [16, 32, 64]
        modules = []
        
        for i, num_blocks in enumerate(layers):
            if num_blocks == 0:
                continue
                
            stride = strides[i]
            out_channels = channels[i]
            
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
            for _ in range(1, num_blocks):
                modules.append(block(in_channels, out_channels))
        
        # 按序列方式构建
        if not modules:
            # 如果没有模块，添加一个恒等映射
            return nn.Identity()
        return nn.Sequential(*modules)
    
    def _make_local_path(self, in_channels, layers, block):
        """构建个性化特征路径"""
        strides = [1, 2, 2]
        channels = [16, 32, 64]
        modules = []
        
        for i, num_blocks in enumerate(layers):
            if num_blocks == 0:
                continue
                
            stride = strides[i]
            out_channels = channels[i]
            
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
            for _ in range(1, num_blocks):
                modules.append(block(in_channels, out_channels))
        
        # 按序列方式构建
        if not modules:
            # 如果没有模块，添加一个恒等映射
            return nn.Identity()
        return nn.Sequential(*modules)
    
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
        x_base = self.shared_base(x)
        
        # 全局特征路径
        global_features = self.global_path(x_base)
        
        # 个性化特征路径
        local_features = self.local_path(x_base)
        
        # 本地分类
        local_logits = self.local_classifier(local_features)
        
        # 返回本地分类结果和全局特征(用于上传到服务器)
        return local_logits, global_features, local_features


class TierHFLServerModel(nn.Module):
    """TierHFL服务器特征处理模型"""
    def __init__(self, base_model='resnet56', in_channels_list=[16, 32, 64], feature_dim=128):
        super(TierHFLServerModel, self).__init__()
        self.in_channels_list = in_channels_list
        self.feature_dim = feature_dim
        
        # 确定基础架构配置
        if base_model == 'resnet56':
            block = BasicBlock
        elif base_model == 'resnet110':
            block = Bottleneck
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")
        
        # 为不同输入通道创建特征处理器
        self.feature_processors = nn.ModuleDict()
        
        for channels in in_channels_list:
            # 每个处理器包含残差连接
            layers = []
            
            # 添加适配层
            if channels != 64:
                downsample = nn.Sequential(
                    nn.Conv2d(channels, 64, kernel_size=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                layers.append(downsample)
                current_channels = 64
            else:
                current_channels = channels
            
            # 添加2个残差块
            for _ in range(2):
                layers.append(block(current_channels, 64))
                current_channels = 64 * block.expansion
            
            # 添加输出层
            layers.extend([
                nn.Conv2d(current_channels, feature_dim, kernel_size=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ])
            
            self.feature_processors[f"ch_{channels}"] = nn.Sequential(*layers)
        
        # 特征处理后的统一输出层
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(feature_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
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
        # 获取输入通道数
        in_channels = x.size(1)
        
        # 选择最接近的处理器
        closest_channels = min(self.in_channels_list, key=lambda c: abs(c - in_channels))
        processor_key = f"ch_{closest_channels}"
        
        # 如果输入通道与处理器不匹配，进行调整
        if in_channels != closest_channels:
            if in_channels < closest_channels:
                # 通道数不足，进行填充
                padding = torch.zeros(x.size(0), closest_channels - in_channels, 
                                     x.size(2), x.size(3), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                # 通道数过多，进行裁剪
                x = x[:, :closest_channels, :, :]
        
        # 应用特征处理器
        x = self.feature_processors[processor_key](x)
        
        # 应用输出层
        x = self.output_layer(x)
        
        return x


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
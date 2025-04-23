import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_base import BasicBlock, Bottleneck, ResNetBase

# 修改TierHFLClientModel类，确保与ResNet结构一致
class TierHFLClientModel(nn.Module):
    """TierHFL客户端双路径模型 - 修复版"""
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
        
        # 共享基础层 - 设计为ResNet前部分
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建layer1 - 16通道输入输出
        self.layer1 = self._make_shared_layer(
            block=block, 
            in_channels=16, 
            out_channels=16, 
            blocks=layers[0], 
            stride=1
        )
        
        # 构建部分layer2作为共享层 - 16通道输入，32通道输出
        shared_blocks = max(1, layers[1] // 3)  # 使用layer2的1/3块作为共享层
        self.layer2_shared = self._make_shared_layer(
            block=block, 
            in_channels=16, 
            out_channels=32, 
            blocks=shared_blocks, 
            stride=2
        )
        
        # 提取特征维度，作为服务器模型输入 - 确保统一为32
        self.output_channels = 32
        
        # 个性化特征路径 - 根据tier调整复杂度
        local_layers = self._adjust_local_path_for_tier(layers, tier)
        self.local_path = self._make_local_path(
            in_channels=32, 
            layers=local_layers, 
            block=block
        )
        
        # 本地分类器
        self.local_classifier = self._create_local_classifier()
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_shared_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """构建共享ResNet层 - 不依赖self.inplanes，确保不同tier之间的一致性"""
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块处理输入/输出通道变化和步长
        layers.append(block(in_channels, out_channels, stride, downsample))
        
        # 剩余块的输入输出通道一致
        current_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(current_channels, out_channels))

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
        """构建个性化特征路径 - 确保所有参数都是显式传递的"""
        modules = []
        current_channels = in_channels
        
        # 添加剩余的layer2块
        if layers[0] > 0:
            for _ in range(layers[0]):
                modules.append(block(current_channels, current_channels//block.expansion))
            
        # 添加layer3块
        if layers[1] > 0:
            # 第一个layer3块使用stride=2
            downsample = nn.Sequential(
                nn.Conv2d(current_channels, 64 * block.expansion,
                         kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64 * block.expansion)
            )
            modules.append(block(current_channels, 64, stride=2, downsample=downsample))
            current_channels = 64 * block.expansion
            
            # 添加剩余layer3块
            for _ in range(1, layers[1]):
                modules.append(block(current_channels, 64))
        
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

    def debug_architecture(self):
        """打印模型架构调试信息"""
        print(f"\n===== 客户端模型架构调试 (Tier {self.tier}) =====")
        
        # 打印各层模块信息
        for name, module in self.named_modules():
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2_shared']):
                print(f"{name}:")
                # 尝试打印模块的关键属性
                for attr in ['in_channels', 'out_channels', 'inplanes', 'planes']:
                    if hasattr(module, attr):
                        print(f"  {attr}={getattr(module, attr)}")
        
        # 打印各参数形状
        print("\n参数形状:")
        for name, param in self.named_parameters():
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2_shared']):
                print(f"  {name}: {param.shape}")
        
        # 测试前向传播与特征形状
        try:
            device = next(self.parameters()).device  # 获取模型当前设备
            dummy_input = torch.randn(1, 3, 32, 32, device=device)  # 确保输入在同一设备
            _, x_base, _ = self(dummy_input)
            print(f"\n共享层输出(x_base)形状: {x_base.shape}")
        except Exception as e:
            print(f"前向传播测试失败: {str(e)}")

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

class TierHFLGlobalClassifier(nn.Module):
    """TierHFL全局分类器 - 移除经验回放功能"""
    def __init__(self, feature_dim=128, num_classes=10):
        super(TierHFLGlobalClassifier, self).__init__()
        self.feature_dim = feature_dim
        
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
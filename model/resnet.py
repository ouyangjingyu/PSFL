import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

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


# class EnhancedServerModel(nn.Module):
#     def __init__(self):
#         super(EnhancedServerModel, self).__init__()
        
#         # 针对不同tier级别的特征处理模块
#         self.tier_processors = nn.ModuleDict({
#             # Tier 7客户端输出16通道特征
#             'tier_7': nn.Sequential(
#                 nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#                 LayerNormCNN(32),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#             # Tier 6客户端输出16通道特征
#             'tier_6': nn.Sequential(
#                 nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#                 LayerNormCNN(32),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#             # Tier 5客户端输出32通道特征
#             'tier_5': nn.Sequential(
#                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#             # Tier 4客户端输出64通道特征
#             'tier_4': nn.Sequential(
#                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#                 LayerNormCNN(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#             # Tier 3客户端输出64通道特征 - 轻量处理
#             'tier_3': nn.Sequential(
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#             # Tier 2客户端输出128通道特征 - 轻量处理
#             'tier_2': nn.Sequential(
#                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#             # Tier 1客户端输出128通道特征 - 仅做标准化和处理
#             'tier_1': nn.Sequential(
#                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#                 LayerNormCNN(128),
#                 nn.ReLU(inplace=True),
#             ),
#         })
        
#         # 统一特征处理 - 所有处理后都输出128维特征
#         self.feature_adapter = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.LayerNorm(128)  # 添加LayerNorm规范化特征分布
#         )
    
#     def forward(self, x, tier=1):
#         tier_str = f'tier_{tier}'
        
#         if tier_str in self.tier_processors:
#             x = self.tier_processors[tier_str](x)
        
#         # 特征适配和归一化
#         features = self.feature_adapter(x)
        
#         return features
#     # 添加获取共享参数方法以支持聚合
#     def get_shared_params(self):
#         return {name: param for name, param in self.named_parameters()}


# class TierAwareClientModel(nn.Module):
#     def __init__(self, num_classes=10, tier=1):
#         super(TierAwareClientModel, self).__init__()
#         self.tier = tier
        
#         # 基础层 - 输出16通道
#         self.base = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
#             LayerNormCNN(16),
#             nn.ReLU(inplace=True)
#         )
        
#         # 计算每层的输出通道数
#         self.channels = [16, 16, 32, 64, 64, 128, 128]  # 各个层的输出通道数
#         self.out_channels = 16  # 默认基础层输出通道
        
#         # 根据tier创建层
#         if tier <= 6:  # Tier 6及以下都有第一层
#             self.layer1 = self._make_residual_group(16, 16, 2)
#             self.out_channels = 16
        
#         if tier <= 5:  # Tier 5及以下都有第二层
#             self.layer2 = self._make_residual_group(16, 32, 2, stride=2)
#             self.out_channels = 32
        
#         if tier <= 4:  # Tier 4及以下都有第三层
#             self.layer3 = self._make_residual_group(32, 64, 2, stride=2)
#             self.out_channels = 64
        
#         if tier <= 3:  # Tier 3及以下都有第四层
#             self.layer4 = self._make_residual_group(64, 64, 2)
#             self.out_channels = 64
        
#         if tier <= 2:  # Tier 2及以下都有第五层
#             self.layer5 = self._make_residual_group(64, 128, 2, stride=2)
#             self.out_channels = 128
        
#         if tier <= 1:  # 只有Tier 1有第六层
#             self.layer6 = self._make_residual_group(128, 128, 2)
#             self.out_channels = 128
            
#         # 本地分类器 - 根据out_channels确定输入维度
#         self.local_classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Dropout(0.3),
#             nn.Linear(self.out_channels, num_classes)
#         )
        
#     def _make_residual_group(self, in_planes, out_planes, blocks, stride=1):
#         layers = []
        
#         # 第一个可能需要下采样的块
#         downsample = None
#         if stride != 1 or in_planes != out_planes:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
#                 LayerNormCNN(out_planes)
#             )
            
#         layers.append(self._make_residual_block(in_planes, out_planes, stride, downsample))
        
#         # 添加剩余的块
#         for _ in range(1, blocks):
#             layers.append(self._make_residual_block(out_planes, out_planes))
            
#         return nn.Sequential(*layers)
    
#     def _make_residual_block(self, in_planes, out_planes, stride=1, downsample=None):
#         return BasicBlock(in_planes, out_planes, stride, downsample)
        
#     def forward(self, x):

#         # 添加调试信息 - 监控输入数据
#         client_id = getattr(self, 'client_id', None)
#         if client_id == 6:  # 针对client6添加调试
#             print(f"\n[Client 6 DEBUG] 输入形状: {x.shape}")
#             print(f"[Client 6 DEBUG] 输入范围: {x.min().item():.4f} to {x.max().item():.4f}")
#             print(f"[Client 6 DEBUG] 输入是否包含NaN: {torch.isnan(x).any().item()}")
#             print(f"[Client 6 DEBUG] 输入是否包含Inf: {torch.isinf(x).any().item()}")
#             print(f"[Client 6 DEBUG] Tier: {self.tier}")

#         # 基础层
#         x = self.base(x)
        
#         # 根据tier级别应用相应的层
#         if hasattr(self, 'layer1'):
#             x = self.layer1(x)
#         if hasattr(self, 'layer2'):
#             x = self.layer2(x)
#         if hasattr(self, 'layer3'):
#             x = self.layer3(x)
#         if hasattr(self, 'layer4'):
#             x = self.layer4(x)
#         if hasattr(self, 'layer5'):
#             x = self.layer5(x)
#         if hasattr(self, 'layer6'):
#             x = self.layer6(x)
        
#         # 保存特征以传递给服务器
#         features = x
        
#         # 应用本地分类器
#         x_pool = self.local_classifier[0:2](x)  # 应用池化和展平
#         local_features = x_pool
#         local_logits = self.local_classifier[2:](x_pool)  # 应用剩余分类器层

#         # 添加特征调试
#         if client_id == 6:
#             print(f"[Client 6 DEBUG] 特征输出形状: {features.shape}")
#             print(f"[Client 6 DEBUG] 特征范围: {features.min().item():.4f} to {features.max().item():.4f}")
#             print(f"[Client 6 DEBUG] 特征是否有NaN: {torch.isnan(features).any().item()}")
            
#         return local_logits, features
    
#     def get_shared_params(self):
#         shared_params = {}
#         for name, param in self.named_parameters():
#             if 'local_classifier' not in name:
#                 shared_params[name] = param
#         return shared_params
    
#     def get_personalized_params(self):
#         personalized_params = {}
#         for name, param in self.named_parameters():
#             if 'local_classifier' in name:
#                 personalized_params[name] = param
#         return personalized_params
class OptimizedClientModel(nn.Module):
    """改进的客户端模型，支持底层特征提取共享和冻结，保留4个tier级别"""
    def __init__(self, num_classes=10, tier=1):
        super(OptimizedClientModel, self).__init__()
        self.tier = tier
        
        # 共享的底层特征提取（这部分将冻结且在客户端间共享）
        self.shared_base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            LayerNormCNN(16),
            nn.ReLU(inplace=True),
            # 第一层卷积组
            self._make_residual_group(16, 16, 2),
        )
        self.out_channels = 16
        
        # 中间层（根据tier动态配置）
        self.mid_layers = nn.ModuleList()
        
        # 根据tier级别添加相应的层 - 4个tier级别
        if tier <= 4:  # Tier 4级 - 基础特征
            # 只包含共享底层，不添加额外层
            pass
            
        if tier <= 3:  # Tier 3级
            self.mid_layers.append(self._make_residual_group(16, 32, 2, stride=2))
            self.out_channels = 32
        
        if tier <= 2:  # Tier 2级
            self.mid_layers.append(self._make_residual_group(32, 64, 2, stride=2))
            self.out_channels = 64
        
        if tier <= 1:  # Tier 1级 - 最复杂模型
            self.mid_layers.append(self._make_residual_group(64, 128, 2, stride=2))
            self.out_channels = 128
            
        # 本地分类器 - 完全个性化
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.out_channels, num_classes)
        )
        
        # 冻结共享底层
        self._freeze_shared_layers()
        
    def _make_residual_group(self, in_planes, out_planes, blocks, stride=1):
        layers = []
        
        # 第一个可能需要下采样的块
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                LayerNormCNN(out_planes)
            )
            
        layers.append(self._make_residual_block(in_planes, out_planes, stride, downsample))
        
        # 添加剩余的块
        for _ in range(1, blocks):
            layers.append(self._make_residual_block(out_planes, out_planes))
            
        return nn.Sequential(*layers)
    
    def _make_residual_block(self, in_planes, out_planes, stride=1, downsample=None):
        return BasicBlock(in_planes, out_planes, stride, downsample)
    
    def _freeze_shared_layers(self):
        """冻结共享的底层特征提取"""
        for param in self.shared_base.parameters():
            param.requires_grad = False
            
    def unfreeze_shared_layers(self):
        """解冻共享层，用于全局更新时"""
        for param in self.shared_base.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        # 共享底层特征提取
        x = self.shared_base(x)
        
        # 中间层处理
        for layer in self.mid_layers:
            x = layer(x)
        
        # 保存特征以传递给服务器
        features = x
        
        # 应用本地分类器
        x_pool = self.local_classifier[0:2](features)  # 池化和展平
        local_logits = self.local_classifier[2:](x_pool)  # 应用剩余分类器层

        return local_logits, features
    
    def get_shared_params(self):
        """获取共享参数"""
        shared_params = {}
        for name, param in self.shared_base.named_parameters():
            shared_params[f"shared_base.{name}"] = param
        return shared_params
    
    def get_personal_params(self):
        """获取个性化参数"""
        personal_params = {}
        # 中间层参数
        for i, layer in enumerate(self.mid_layers):
            for name, param in layer.named_parameters():
                personal_params[f"mid_layers.{i}.{name}"] = param
        
        # 分类器参数
        for name, param in self.local_classifier.named_parameters():
            personal_params[f"local_classifier.{name}"] = param
            
        return personal_params

class AdaptiveServerModel(nn.Module):
    """自适应特征处理的服务器模型，支持4个tier级别"""
    def __init__(self, in_channels_list=[16, 32, 64, 128], feature_dim=128):
        super(AdaptiveServerModel, self).__init__()
        self.in_channels_list = in_channels_list
        self.feature_dim = feature_dim
        
        # 为每种输入通道数创建自适应处理模块
        self.processors = nn.ModuleDict()
        for channels in in_channels_list:
            self.processors[f'channel_{channels}'] = nn.Sequential(
                # 自适应特征处理模块
                AdaptiveFeatureProcessor(channels, feature_dim),
                # 输出层归一化
                nn.LayerNorm(feature_dim)
            )
            
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x, tier=1):
        # 根据输入特征的通道数选择对应的处理模块
        channels = x.size(1)
        
        if channels not in self.in_channels_list:
            # 如果不匹配任何预设通道数，使用最接近的
            closest_channels = min(self.in_channels_list, key=lambda c: abs(c - channels))
            print(f"警告: 输入通道 {channels} 不在预设列表中, 使用最接近的 {closest_channels}")
            channels = closest_channels
            
        # 选择处理模块
        processor = self.processors[f'channel_{channels}']
        
        # 应用处理
        features = processor(x)
        
        # 应用特征融合
        fused_features = self.feature_fusion(features)
        
        return fused_features

class AdaptiveFeatureProcessor(nn.Module):
    """自适应特征处理模块，使用注意力机制动态调整处理方式"""
    def __init__(self, in_channels, out_dim=128):
        super(AdaptiveFeatureProcessor, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 卷积特征提取
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True)
        )
        
        # 自适应池化和展平
        self.pool_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 输出投影
        self.output_projection = nn.Linear(in_channels*2, out_dim)
        
    def forward(self, x):
        # 应用注意力
        channel_attn = self.channel_attention(x)
        spatial_attn = self.spatial_attention(x)
        
        # 加权特征
        x = x * channel_attn * spatial_attn
        
        # 特征提取
        x = self.conv_blocks(x)
        
        # 池化和展平
        x = self.pool_flatten(x)
        
        # 输出投影
        x = self.output_projection(x)
        
        return x

def align_shared_layers(client_models, client_weights=None, target_device=None):
    """对齐所有客户端模型的共享层参数"""
    if not client_models:
        return None
    
    # 获取共享层参数
    shared_params_list = []
    client_ids = []
    
    for client_id, model in client_models.items():
        shared_params = model.get_shared_params()
        if shared_params:
            shared_params_list.append(shared_params)
            client_ids.append(client_id)
    
    if not shared_params_list:
        return None
    
    # 如果没有提供权重，使用均等权重
    if client_weights is None:
        client_weights = {client_id: 1.0 / len(client_ids) for client_id in client_ids}
    
    # 确定目标设备
    if target_device is None:
        # 检查第一个参数的设备并使用它作为目标设备
        first_param = next(iter(shared_params_list[0].values()))
        target_device = first_param.device
    
    # 聚合共享层参数
    aggregated_params = {}
    for key in shared_params_list[0].keys():
        weighted_sum = None
        weight_sum = 0
        
        for i, client_id in enumerate(client_ids):
            if client_id in client_weights:
                weight = client_weights[client_id]
            else:
                weight = 1.0 / len(client_ids)
                
            weight_sum += weight
            
            # 确保参数在目标设备上
            param = shared_params_list[i][key].to(target_device)
            
            # 加权累加
            if weighted_sum is None:
                weighted_sum = weight * param
            else:
                weighted_sum += weight * param
            
        if weight_sum > 0 and weighted_sum is not None:
            aggregated_params[key] = weighted_sum / weight_sum
    
    # 更新每个客户端模型的共享层
    for client_id, model in client_models.items():
        if hasattr(model, 'shared_base'):
            # 暂时解冻以更新参数
            model.unfreeze_shared_layers()
            
            # 更新共享层参数
            for name, param in model.shared_base.named_parameters():
                key = f"shared_base.{name}"
                if key in aggregated_params:
                    # 将聚合的参数移到与原始参数相同的设备上
                    param.data = aggregated_params[key].to(param.device).clone()
            
            # 再次冻结
            model._freeze_shared_layers()
    
    return aggregated_params

class ImprovedGlobalClassifier(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super(ImprovedGlobalClassifier, self).__init__()
        self.num_classes = num_classes
        
        # 使用LayerNorm替代Dropout，提高特征空间稳定性
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )
        
        # 改进初始化方法
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        return self.classifier(features)
        
    def get_params(self):
        return {name: param for name, param in self.named_parameters()}


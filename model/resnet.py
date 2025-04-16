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

# # TierHFL-ResNet客户端模型
# class TierHFLClientModel(nn.Module):
#     def __init__(self, block, layers, num_classes=10, tier=1):
#         super(TierHFLClientModel, self).__init__()
#         self.inplanes = 16
#         self.tier = tier
        
#         # 基础层
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.norm1 = LayerNormCNN(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
        
#         # 根据tier决定包含哪些层
#         self.layers = nn.ModuleList()
#         planes = [16, 16, 32, 32, 64, 64]
#         strides = [1, 1, 2, 1, 2, 1]
        
#         # 根据tier级别决定要实现的层数
#         self.active_layers = min(7 - tier, 6)  # tier 1有6层，tier 7有0层
        
#         for i in range(self.active_layers):
#             if i == 0:
#                 self.layers.append(self._make_layer(block, planes[i], layers[i], stride=strides[i]))
#             else:
#                 self.inplanes = planes[i-1] * block.expansion
#                 self.layers.append(self._make_layer(block, planes[i], layers[i], stride=strides[i]))
        
#         # 本地分类器 - 对数据分布的个性化适应
#         self.classifier = None
#         if self.active_layers == 0:  # tier 7
#             self.classifier = LocalClassifier(16, num_classes)
#         elif self.active_layers == 1:  # tier 6
#             self.classifier = LocalClassifier(16 * block.expansion, num_classes)
#         elif self.active_layers == 2:  # tier 5
#             self.classifier = LocalClassifier(16 * block.expansion, num_classes)
#         elif self.active_layers == 3:  # tier 4
#             self.classifier = LocalClassifier(32 * block.expansion, num_classes)
#         elif self.active_layers == 4:  # tier 3
#             self.classifier = LocalClassifier(32 * block.expansion, num_classes)
#         elif self.active_layers == 5:  # tier 2
#             self.classifier = LocalClassifier(64 * block.expansion, num_classes)
#         elif self.active_layers == 6:  # tier 1
#             self.classifier = LocalClassifier(64 * block.expansion, num_classes)
        
#         # 初始化权重
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, LayerNormCNN):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 LayerNormCNN(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # 基础层处理
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu(x)
        
#         # 根据tier执行相应的层
#         extracted_features = x
#         for i, layer in enumerate(self.layers):
#             extracted_features = layer(extracted_features)
        
#         # 应用本地分类器
#         local_logits = self.classifier(extracted_features) if self.classifier else None
        
#         return local_logits, extracted_features
    
#     # 区分共享参数和个性化参数
#     def get_shared_params(self):
#         shared_params = {}
#         for name, param in self.named_parameters():
#             # 排除分类器参数
#             if 'classifier' not in name:
#                 shared_params[name] = param
#         return shared_params
    
#     def get_personalized_params(self):
#         personalized_params = {}
#         for name, param in self.named_parameters():
#             if 'classifier' in name:
#                 personalized_params[name] = param
#         return personalized_params

# TierHFL-ResNet服务器模型
# class TierHFLServerModel(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(TierHFLServerModel, self).__init__()
#         self.num_classes = num_classes
        
#         # 服务器端处理模块 - 针对不同tier
#         self.tier_modules = nn.ModuleDict()
#         planes = [16, 16, 32, 32, 64, 64]
#         strides = [1, 1, 2, 1, 2, 1]
        
#         # 创建每个tier对应的处理模块
#         for tier in range(1, 8):
#             tier_str = f'tier_{tier}'
#             self.tier_modules[tier_str] = nn.ModuleList()
            
#             # 根据tier确定起始层和inplanes
#             start_layer = max(0, 7 - tier)
#             inplanes = 16  # 默认起始通道数
            
#             # 调整inplanes基于tier
#             if tier == 7:
#                 inplanes = 16
#             elif tier == 6:
#                 inplanes = 16 * block.expansion
#             elif tier == 5:
#                 inplanes = 16 * block.expansion
#             elif tier == 4:
#                 inplanes = 32 * block.expansion
#             elif tier == 3:
#                 inplanes = 32 * block.expansion
#             elif tier == 2:
#                 inplanes = 64 * block.expansion
#             elif tier == 1:
#                 inplanes = 64 * block.expansion
                
#             # 创建该tier需要的层
#             for i in range(start_layer, 6):
#                 if i == start_layer:
#                     this_layer = self._make_layer(block, planes[i], layers[i], 
#                                                  stride=strides[i], inplanes=inplanes)
#                 else:
#                     inplanes = planes[i-1] * block.expansion
#                     this_layer = self._make_layer(block, planes[i], layers[i], 
#                                                  stride=strides[i], inplanes=inplanes)
#                 self.tier_modules[tier_str].append(this_layer)
        
#         # 全局分类器
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.global_classifier = nn.Linear(64 * block.expansion, num_classes)
        
#         # 简化的特征处理
#         self.feature_processor = SimpleFeatureProcessor(64 * block.expansion)
        
#         # 初始化权重
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, LayerNormCNN):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, inplanes=None):
#         inplanes = self.inplanes if inplanes is None else inplanes
#         downsample = None
#         if stride != 1 or inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(inplanes, planes * block.expansion, stride),
#                 LayerNormCNN(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x, tier=1):
#         # 获取对应tier的处理模块
#         tier_str = f'tier_{tier}'
        
#         # 处理特征
#         for layer in self.tier_modules[tier_str]:
#             x = layer(x)
        
#         # 应用特征处理
#         x = self.feature_processor(x)
        
#         # 全局分类
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         logits = self.global_classifier(x)
        
#         return logits, x  # 返回分类结果和特征

# 创建客户端模型函数

class EnhancedServerModel(nn.Module):
    def __init__(self):
        super(EnhancedServerModel, self).__init__()
        
        # 针对不同tier级别的特征处理模块
        self.tier_processors = nn.ModuleDict({
            # Tier 7客户端输出16通道特征
            'tier_7': nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                LayerNormCNN(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
            # Tier 6客户端输出16通道特征
            'tier_6': nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                LayerNormCNN(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
            # Tier 5客户端输出32通道特征
            'tier_5': nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
            # Tier 4客户端输出64通道特征
            'tier_4': nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                LayerNormCNN(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
            # Tier 3客户端输出64通道特征 - 轻量处理
            'tier_3': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
            # Tier 2客户端输出128通道特征 - 轻量处理
            'tier_2': nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
            # Tier 1客户端输出128通道特征 - 仅做标准化和处理
            'tier_1': nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                LayerNormCNN(128),
                nn.ReLU(inplace=True),
            ),
        })
        
        # 统一特征处理 - 所有处理后都输出128维特征
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(128)  # 添加LayerNorm规范化特征分布
        )
    
    def forward(self, x, tier=1):
        tier_str = f'tier_{tier}'
        
        if tier_str in self.tier_processors:
            x = self.tier_processors[tier_str](x)
        
        # 特征适配和归一化
        features = self.feature_adapter(x)
        
        return features
    # 添加获取共享参数方法以支持聚合
    def get_shared_params(self):
        return {name: param for name, param in self.named_parameters()}


class TierAwareClientModel(nn.Module):
    def __init__(self, num_classes=10, tier=1):
        super(TierAwareClientModel, self).__init__()
        self.tier = tier
        
        # 基础层 - 输出16通道
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            LayerNormCNN(16),
            nn.ReLU(inplace=True)
        )
        
        # 计算每层的输出通道数
        self.channels = [16, 16, 32, 64, 64, 128, 128]  # 各个层的输出通道数
        self.out_channels = 16  # 默认基础层输出通道
        
        # 根据tier创建层
        if tier <= 6:  # Tier 6及以下都有第一层
            self.layer1 = self._make_residual_group(16, 16, 2)
            self.out_channels = 16
        
        if tier <= 5:  # Tier 5及以下都有第二层
            self.layer2 = self._make_residual_group(16, 32, 2, stride=2)
            self.out_channels = 32
        
        if tier <= 4:  # Tier 4及以下都有第三层
            self.layer3 = self._make_residual_group(32, 64, 2, stride=2)
            self.out_channels = 64
        
        if tier <= 3:  # Tier 3及以下都有第四层
            self.layer4 = self._make_residual_group(64, 64, 2)
            self.out_channels = 64
        
        if tier <= 2:  # Tier 2及以下都有第五层
            self.layer5 = self._make_residual_group(64, 128, 2, stride=2)
            self.out_channels = 128
        
        if tier <= 1:  # 只有Tier 1有第六层
            self.layer6 = self._make_residual_group(128, 128, 2)
            self.out_channels = 128
            
        # 本地分类器 - 根据out_channels确定输入维度
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.out_channels, num_classes)
        )
        
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
        
    def forward(self, x):

        # 添加调试信息 - 监控输入数据
        client_id = getattr(self, 'client_id', None)
        if client_id == 6:  # 针对client6添加调试
            print(f"\n[Client 6 DEBUG] 输入形状: {x.shape}")
            print(f"[Client 6 DEBUG] 输入范围: {x.min().item():.4f} to {x.max().item():.4f}")
            print(f"[Client 6 DEBUG] 输入是否包含NaN: {torch.isnan(x).any().item()}")
            print(f"[Client 6 DEBUG] 输入是否包含Inf: {torch.isinf(x).any().item()}")
            print(f"[Client 6 DEBUG] Tier: {self.tier}")

        # 基础层
        x = self.base(x)
        
        # 根据tier级别应用相应的层
        if hasattr(self, 'layer1'):
            x = self.layer1(x)
        if hasattr(self, 'layer2'):
            x = self.layer2(x)
        if hasattr(self, 'layer3'):
            x = self.layer3(x)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)
        if hasattr(self, 'layer5'):
            x = self.layer5(x)
        if hasattr(self, 'layer6'):
            x = self.layer6(x)
        
        # 保存特征以传递给服务器
        features = x
        
        # 应用本地分类器
        x_pool = self.local_classifier[0:2](x)  # 应用池化和展平
        local_features = x_pool
        local_logits = self.local_classifier[2:](x_pool)  # 应用剩余分类器层

        # 添加特征调试
        if client_id == 6:
            print(f"[Client 6 DEBUG] 特征输出形状: {features.shape}")
            print(f"[Client 6 DEBUG] 特征范围: {features.min().item():.4f} to {features.max().item():.4f}")
            print(f"[Client 6 DEBUG] 特征是否有NaN: {torch.isnan(features).any().item()}")
            
        return local_logits, features
    
    def get_shared_params(self):
        shared_params = {}
        for name, param in self.named_parameters():
            if 'local_classifier' not in name:
                shared_params[name] = param
        return shared_params
    
    def get_personalized_params(self):
        personalized_params = {}
        for name, param in self.named_parameters():
            if 'local_classifier' in name:
                personalized_params[name] = param
        return personalized_params

# class GlobalClassifier(nn.Module):
#     def __init__(self, feature_dim=128, num_classes=10):
#         super(GlobalClassifier, self).__init__()
#         self.num_classes = num_classes
        
#         # 多层分类器，有更强的表达能力
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, features):
#         # 设置调试标志
#         debug_mode = features.shape[0] < 10  # 对小批量进行调试(通常是测试批量)
        
#         if debug_mode:
#             print(f"\n[Classifier Debug] 处理特征:")
#             print(f"- 输入形状: {features.shape}")
#             print(f"- 输入维度与预期匹配: {features.shape[1] == self.classifier[0].in_features}")
#             print(f"- 输入范围: {features.min().item():.4f} 到 {features.max().item():.4f}")
            
#             # 检查是否有异常值或全零特征
#             has_nan = torch.isnan(features).any().item()
#             has_inf = torch.isinf(features).any().item()
#             zero_ratio = (torch.abs(features) < 1e-5).float().mean().item()
#             print(f"- 输入有NaN: {has_nan}, 有Inf: {has_inf}, 接近零值比例: {zero_ratio*100:.2f}%")
            
#             # 分析第一个样本的特征分布
#             if features.shape[0] > 0:
#                 sample = features[0].detach().cpu().numpy()
#                 print(f"- 样本特征极值: min={sample.min():.4f}, max={sample.max():.4f}")
#                 print(f"- 样本特征均值: {sample.mean():.4f}, 标准差: {sample.std():.4f}")
                
#                 # 检查是否有不正常分布
#                 if sample.std() < 0.01:
#                     print("!!! 警告: 特征分布异常，标准差过小 !!!")
        
#         try:
#             # 检查分类器第一层
#             first_layer = self.classifier[0]
#             if debug_mode and isinstance(first_layer, nn.Linear):
#                 w = first_layer.weight
#                 w_stats = {
#                     'mean': w.mean().item(),
#                     'std': w.std().item(),
#                     'min': w.min().item(),
#                     'max': w.max().item()
#                 }
#                 print(f"- 分类器第一层权重统计: {w_stats}")
            
#             # 前向传播
#             logits = self.classifier(features)
            
#             if debug_mode:
#                 print(f"- 输出形状: {logits.shape}")
#                 print(f"- 输出范围: {logits.min().item():.4f} 到 {logits.max().item():.4f}")
                
#                 # 分析预测
#                 preds = torch.argmax(logits, dim=1)
#                 unique_preds = torch.unique(preds)
#                 print(f"- 预测的类别: {preds.cpu().numpy()}")
#                 print(f"- 不同类别数量: {len(unique_preds)}")
                
#                 # 分析softmax输出
#                 softmax_out = torch.softmax(logits, dim=1)
#                 max_probs = torch.max(softmax_out, dim=1)[0]
#                 print(f"- 最大概率值: {max_probs.mean().item():.4f}")
#                 print(f"- 第一个样本的概率分布: {softmax_out[0].cpu().numpy().round(3)}")
                
#                 # 严重问题检测
#                 if len(unique_preds) == 1:
#                     print("!!! 严重问题: 所有样本预测相同类别 !!!")
#                     # 检查是否是模型偏置问题
#                     bias_term = None
#                     if hasattr(self.classifier[-1], 'bias'):
#                         bias_term = self.classifier[-1].bias
#                         max_bias_idx = torch.argmax(bias_term).item()
#                         print(f"- 最后一层偏置最大值索引: {max_bias_idx}")
#                         print(f"- 偏置值: {bias_term.cpu().numpy().round(3)}")
                        
#                         # 如果最大偏置对应的类别与预测一致，很可能是偏置问题
#                         if max_bias_idx == unique_preds[0].item():
#                             print("!!! 分类器存在严重偏置问题 !!!")
#         except Exception as e:
#             print(f"[Classifier Error] 前向传播出错: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             # 返回一个全零logits
#             logits = torch.zeros(features.size(0), self.num_classes, device=features.device)
        
#         return logits
        
#     # 添加获取参数方法以支持聚合
#     def get_params(self):
#         return {name: param for name, param in self.named_parameters()}


# def create_tierhfl_client_model(tier, num_classes=10, model_type='resnet56'):
#     if model_type == 'resnet56':
#         return TierHFLClientModel(BasicBlock, [3, 3, 3, 3, 3, 3], num_classes, tier)
#     elif model_type == 'resnet110':
#         return TierHFLClientModel(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes, tier)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")

# # 创建服务器模型函数
# def create_tierhfl_server_model(num_classes=10, model_type='resnet56'):
#     if model_type == 'resnet56':
#         return TierHFLServerModel(BasicBlock, [3, 3, 3, 3, 3, 3], num_classes)
#     elif model_type == 'resnet110':
#         return TierHFLServerModel(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")

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
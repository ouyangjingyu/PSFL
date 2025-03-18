import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFeatureProjection(nn.Module):
    """特征适配层，处理不同tier客户端输出的特征维度差异"""
    def __init__(self, in_features, out_features, activation='relu', dropout_rate=0.3):
        super(AdaptiveFeatureProjection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 只有当输入维度与输出维度不同时才进行投影
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
            
            # 添加批标准化层稳定特征分布
            self.bn = nn.BatchNorm1d(out_features)
            
            # 根据传入的参数选择激活函数
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            elif activation == 'elu':
                self.activation = nn.ELU(inplace=True)
            else:
                self.activation = nn.Identity()
                
            # 添加Dropout增强鲁棒性
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        else:
            # 如果维度相同，使用Identity层
            self.projection = nn.Identity()
            self.bn = nn.Identity()
            self.activation = nn.Identity()
            self.dropout = nn.Identity()
    
    def forward(self, x):
        # 对于不同维度的输入，执行投影、正则化和激活
        if self.in_features != self.out_features:
            x = self.projection(x)
            x = self.bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class UnifiedClassifier(nn.Module):
    """统一的全局分类器，所有tier客户端共享"""
    def __init__(self, in_features, hidden_dims, num_classes, dropout_rate=0.5):
        super(UnifiedClassifier, self).__init__()
        
        # 参数验证
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        
        # 构建多层分类器网络
        layers = []
        prev_dim = in_features
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # 添加最终分类层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # 将所有层组合为分类器
        self.classifier = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)


class EnhancedFeatureExtractor(nn.Module):
    """增强的特征提取器基类，用于客户端模型"""
    def __init__(self, base_model, output_dim, target_dim, activation='relu'):
        super(EnhancedFeatureExtractor, self).__init__()
        self.base_model = base_model

        # 运行一次前向传播以获取实际的输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)  # 假设输入是图像
            features = base_model(dummy_input)
            if isinstance(features, tuple):
                features = features[0]
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            actual_dim = features.shape[1]

        self.feature_adapter = AdaptiveFeatureProjection(
            actual_dim, target_dim, activation=activation
        )
    
    def forward(self, x):
        # 获取基础模型输出的特征
        features = self.base_model(x)
        
        # 如果特征是元组（某些ResNet变体会返回多个输出），取第一个元素
        if isinstance(features, tuple):
            features = features[0]
        
        # 对于卷积特征，需要先进行全局平均池化
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # 使用特征适配层处理特征
        adapted_features = self.feature_adapter(features)
        
        return adapted_features, features


# 增强的ResNet服务器模型，带有特征适配层和统一分类器
class EnhancedServerModel(nn.Module):
    def __init__(self, base_server_model, input_dim, target_dim, hidden_dims, num_classes):
        super(EnhancedServerModel, self).__init__()
        self.base_model = base_server_model
        
        # 添加特征适配层
        self.feature_adapter = AdaptiveFeatureProjection(
            input_dim, target_dim, activation='leaky_relu'
        )
        
        # 添加统一分类器
        self.classifier = UnifiedClassifier(
            target_dim, hidden_dims, num_classes
        )
    
    def forward(self, x):
        # 通过基础服务器模型处理特征
        if hasattr(self.base_model, 'forward'):
            features = self.base_model(x)
        else:
            features = x
            
        # 如果特征是元组，取第一个元素
        if isinstance(features, tuple):
            features = features[0]
        
        # 对于卷积特征，进行全局平均池化
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # 使用特征适配层处理特征
        adapted_features = self.feature_adapter(features)
        
        # 通过统一分类器生成预测
        logits = self.classifier(adapted_features)
        
        return logits


# 创建增强的客户端特征提取器
def create_enhanced_client_model(base_model, tier, target_dim=256):
    """基于tier级别创建适当的客户端模型"""
    # 定义不同tier级别的输出维度
    tier_output_dims = {
        1: 256,  # 完整的客户端模型
        2: 256,  # 客户端到layer5
        3: 128,  # 客户端到layer4
        4: 128,  # 客户端到layer3
        5: 64,   # 客户端到layer2
        6: 64,   # 客户端到layer1
        7: 16    # 只有基础层
    }
    
    # 获取当前tier的输出维度
    output_dim = tier_output_dims.get(tier, 64)
    
    # 创建增强的特征提取器
    enhanced_model = EnhancedFeatureExtractor(
        base_model,
        output_dim,
        target_dim
    )
    
    return enhanced_model


# 创建增强的服务器模型
def create_enhanced_server_model(base_server_model, tier, target_dim=256, num_classes=10):
    """基于tier级别创建适当的服务器模型"""
    # 定义不同tier级别的输入维度
    tier_input_dims = {
        1: 256,  # 完整的客户端模型
        2: 256,  # 客户端到layer5
        3: 128,  # 客户端到layer4
        4: 128,  # 客户端到layer3
        5: 64,   # 客户端到layer2
        6: 64,   # 客户端到layer1
        7: 16    # 只有基础层
    }
    
    # 获取当前tier的输入维度
    input_dim = tier_input_dims.get(tier, 64)
    
    # 创建增强的服务器模型
    enhanced_model = EnhancedServerModel(
        base_server_model,
        input_dim,
        target_dim,
        [128, 64],  # 隐藏层维度
        num_classes
    )
    
    return enhanced_model
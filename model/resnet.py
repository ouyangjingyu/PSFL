import torch
import torch.nn as nn
import torch.nn.functional as F

# 标记特征提取层和非特征提取层
FEATURE_EXTRACTION_MODULES = ['conv', 'gn', 'layer', 'downsample']
NON_FEATURE_EXTRACTION_MODULES = ['classifier', 'projection', 'fc']

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
                 base_width=64, dilation=1, norm_layer=None, groups_per_channel=32):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = lambda channels: nn.GroupNorm(num_groups=max(1, channels // groups_per_channel), num_channels=channels)
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = norm_layer(planes)  # 使用GN替代BN
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = norm_layer(planes)  # 使用GN替代BN
        self.downsample = downsample
        self.stride = stride
        
        # 标记为特征提取层
        self.is_feature_extraction = True

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, groups_per_channel=32):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = lambda channels: nn.GroupNorm(num_groups=max(1, channels // groups_per_channel), num_channels=channels)
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.gn1 = norm_layer(width)  # 使用GN替代BN
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.gn2 = norm_layer(width)  # 使用GN替代BN
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.gn3 = norm_layer(planes * self.expansion)  # 使用GN替代BN
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # 标记为特征提取层
        self.is_feature_extraction = True

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 自适应特征归一化层 - 新增
class AdaptiveFeatureNormalization(nn.Module):
    """自适应特征归一化，保留原始特征分布特性"""
    def __init__(self, num_channels=None, eps=1e-5, affine=True, mode=None):
        super(AdaptiveFeatureNormalization, self).__init__()
        self.eps = eps
        self.affine = affine
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.num_channels = num_channels
        # mode参数可以忽略，添加只是为了兼容现有调用
        
        # 不再预先定义运行时统计量，而是动态创建
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
    
    def _initialize_buffers(self, shape, device=None):
        """根据输入特征的形状初始化统计量
        
        Args:
            shape: 输入张量的形状（元组）
            device: 张量所在设备
        """
        # 如果未提供设备，使用默认设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if len(shape) == 4:  # 卷积特征
            c = shape[1]
            self.register_buffer('running_mean', torch.zeros(c, device=device))
            self.register_buffer('running_var', torch.ones(c, device=device))
        else:  # 1D特征
            c = shape[-1]
            self.register_buffer('running_mean', torch.zeros(c, device=device))
            self.register_buffer('running_var', torch.ones(c, device=device))
            
        if self.affine:
            self.weight = nn.Parameter(torch.ones(c, device=device))
            self.bias = nn.Parameter(torch.zeros(c, device=device))
            
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=device))
        self.initialized.fill_(1)
        
    def forward(self, x):
        # 检查是否初始化
        if not self.initialized:
            self._initialize_buffers(x.shape, x.device)
        
        # 计算统计量
        if len(x.shape) == 4:  # 卷积特征
            # 获取通道维度
            c = x.shape[1]
            
            # 计算批次统计量
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # 检查通道维度是否匹配
            if batch_mean.shape[0] != c:
                print(f"通道维度不匹配: 预期 {c}, 实际 {batch_mean.shape[0]}")
                # 处理不匹配情况
                if batch_mean.shape[0] > c:
                    batch_mean = batch_mean[:c]
                    batch_var = batch_var[:c]
                else:
                    # 零填充
                    pad_mean = torch.zeros(c, device=batch_mean.device)
                    pad_var = torch.ones(c, device=batch_var.device)
                    pad_mean[:batch_mean.shape[0]] = batch_mean
                    pad_var[:batch_var.shape[0]] = batch_var
                    batch_mean = pad_mean
                    batch_var = pad_var
            
            # 归一化 - 确保形状正确
            x_normalized = (x - batch_mean.view(1, c, 1, 1)) / torch.sqrt(batch_var.view(1, c, 1, 1) + self.eps)
        else:  # 1D特征
            # 计算1D特征统计量
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # 归一化
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        # 更新运行时统计量
        if self.training:
            with torch.no_grad():
                # 确保batch统计量在正确的设备上
                if batch_mean.device != self.running_mean.device:
                    batch_mean = batch_mean.to(self.running_mean.device)
                    batch_var = batch_var.to(self.running_var.device)
                    
                self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
                self.running_var = 0.9 * self.running_var + 0.1 * batch_var
                self.num_batches_tracked += 1
        
        # 仿射变换
        if self.affine:
            # 确保权重和偏置在与x相同的设备上
            if self.weight.device != x.device:
                weight = self.weight.to(x.device)
                bias = self.bias.to(x.device)
            else:
                weight = self.weight
                bias = self.bias
                
            if len(x.shape) == 4:
                return x_normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
            else:
                return x_normalized * weight + bias
        else:
            return x_normalized
    
    def get_feature_stats(self):
        """返回特征统计信息用于监控"""
        stats = {}
        
        # 检查是否已初始化
        if hasattr(self, 'running_mean') and hasattr(self, 'running_var'):
            stats['mean'] = self.running_mean.clone().detach()
            stats['var'] = self.running_var.clone().detach()
            stats['num_batches'] = self.num_batches_tracked.item() if hasattr(self, 'num_batches_tracked') else 0
        else:
            # 如果未初始化，返回空统计信息
            stats['initialized'] = False
            stats['mean'] = None
            stats['var'] = None
            stats['num_batches'] = 0
            
        return stats


# 特征缩放层 - 新增
class FeatureScaling(nn.Module):
    """特征缩放层，控制特征范数在合理范围内"""
    def __init__(self, target_norm=20.0, eps=1e-5, adaptive=True):
        super(FeatureScaling, self).__init__()
        self.target_norm = target_norm
        self.eps = eps
        self.adaptive = adaptive
        
        # 运行时统计量
        self.register_buffer('running_norm', torch.tensor(0.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        # 计算当前特征范数
        orig_shape = x.shape
        if x.dim() > 2:
            # 对于卷积特征，先展平
            x_flat = x.view(x.size(0), -1)
            norms = torch.norm(x_flat, dim=1, keepdim=True)
            
            # 计算平均范数
            avg_norm = norms.mean()
            
            # 更新统计量
            with torch.no_grad():
                if self.training:
                    self.running_norm = 0.9 * self.running_norm + 0.1 * avg_norm
                    self.num_batches_tracked += 1
            
            # 自适应缩放
            if self.adaptive:
                # 只有当范数过大时才进行缩放
                scale_factor = torch.min(
                    self.target_norm / (norms + self.eps),
                    torch.ones_like(norms)
                )
                x_flat = x_flat * scale_factor
                x = x_flat.view(orig_shape)
            else:
                # 固定缩放到目标范数
                scale_factor = self.target_norm / (norms + self.eps)
                x_flat = x_flat * scale_factor
                x = x_flat.view(orig_shape)
        else:
            # 对于全连接层特征
            norms = torch.norm(x, dim=1, keepdim=True)
            avg_norm = norms.mean()
            
            # 更新统计量
            with torch.no_grad():
                if self.training:
                    self.running_norm = 0.9 * self.running_norm + 0.1 * avg_norm
                    self.num_batches_tracked += 1
            
            # 自适应缩放
            if self.adaptive:
                scale_factor = torch.min(
                    self.target_norm / (norms + self.eps),
                    torch.ones_like(norms)
                )
            else:
                scale_factor = self.target_norm / (norms + self.eps)
            
            x = x * scale_factor
        
        return x
    
    def get_norm_stats(self):
        """返回范数统计信息"""
        return {
            'avg_norm': self.running_norm.item(),
            'num_batches': self.num_batches_tracked.item()
        }


# 特征还原层 - 新增
class FeatureRestoration(nn.Module):
    """特征还原层，恢复特征的判别信息"""
    def __init__(self, in_features, hidden_dim=None):
        super(FeatureRestoration, self).__init__()
        hidden_dim = hidden_dim or in_features
        
        # 添加卷积特征处理能力
        self.is_conv = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 使用残差结构恢复特征
        self.residual_block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_features),
            nn.LayerNorm(in_features)
        )
        
        # 特征门控机制，决定多少原始特征和恢复特征混合
        self.gate = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        original_shape = x.shape
        
        # 处理卷积特征
        if len(x.shape) > 2:
            self.is_conv = True
            # 保存原始形状用于还原
            n, c, h, w = x.shape
            
            # 对于卷积特征，我们先进行池化以获取通道特征
            x_flat = x.view(n, c, -1).mean(dim=2)
        else:
            # 已经是扁平特征
            x_flat = x
        
        # 残差处理
        identity = x_flat
        restored = self.residual_block(x_flat)
        
        # 自适应混合原始特征和恢复特征
        gate_value = torch.sigmoid(self.gate)
        out_flat = gate_value * restored + (1 - gate_value) * identity
        
        # 如果是卷积特征，保持原始维度
        if self.is_conv:
            # 把通道特征广播回原始形状
            out = out_flat.view(n, c, 1, 1).expand(n, c, h, w)
            return out
        else:
            return out_flat


class LayerNormClassifier(nn.Module):
    """使用LayerNorm的分类器，适用于异质性数据环境"""
    def __init__(self, in_features, num_classes, hidden_dims=None, is_local=False, dropout_rate=None, device=None):
        super(LayerNormClassifier, self).__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hidden_dims is None:
            # 客户端本地分类器和全局分类器使用统一的隐藏层设计
            hidden_dims = [128, 64]
        
        if dropout_rate is None:
            # 统一设置dropout率
            dropout_rate = 0.5 if is_local else 0.3
        
        # 使用LayerNorm替代BatchNorm - 不依赖批次统计量
        self.norm_input = nn.LayerNorm(in_features)
        
        # 构建多层分类器
        layers = []
        prev_dim = in_features
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # 构建分类器
        self.classifier = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
        # 标记为非特征提取层
        self.is_feature_extraction = False
        
    def _init_weights(self):
        # 使用更好的初始化方法
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 确保输入在正确设备上
        if hasattr(self, 'device'):
            target_device = self.device
        else:
            # 找到模型参数的设备
            for param in self.parameters():
                target_device = param.device
                break
        
        # 确保输入在正确设备上
        if x.device != target_device:
            x = x.to(target_device)
            
        # 确保权重在正确设备上
        if self.norm_input.weight is not None and self.norm_input.weight.device != target_device:
            self.norm_input.weight = nn.Parameter(self.norm_input.weight.to(target_device))
        if self.norm_input.bias is not None and self.norm_input.bias.device != target_device:
            self.norm_input.bias = nn.Parameter(self.norm_input.bias.to(target_device))
            
        x = self.norm_input(x)
        x = self.classifier(x)
        return x


# 添加类别平衡损失函数 - 新增
class ClassBalancedLoss(nn.Module):
    """类别平衡损失函数，解决类别不平衡问题"""
    def __init__(self, num_classes, beta=0.9999, samples_per_class=None, device=None):
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.samples_per_class = samples_per_class
        self.device = device  # 添加设备参数
        
        # 如果提供了每个类别的样本数，计算类别权重
        if samples_per_class is not None:
            self.weights = self._compute_class_weights(samples_per_class)
        else:
            # 默认所有类别权重相同
            self.weights = torch.ones(num_classes)
            
        # 将权重移至指定设备
        if self.device is not None:
            self.weights = self.weights.to(self.device)
        
        # 使用交叉熵损失作为基础损失函数
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def _compute_class_weights(self, samples_per_class):
        """计算类别权重"""
        if self.device is not None:
            samples_per_class = samples_per_class.to(self.device)
            
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.num_classes
        return weights
    
    def update_samples_per_class(self, samples_per_class):
        """更新每个类别的样本数并重新计算权重"""
        self.samples_per_class = samples_per_class
        self.weights = self._compute_class_weights(samples_per_class)
        if self.device is not None:
            self.weights = self.weights.to(self.device)
    
    def update_weights_from_predictions(self, predictions):
        """根据预测分布更新权重"""
        # 计算每个类别的预测频率
        if self.device is not None:
            predictions = predictions.to(self.device)
            
        total_preds = torch.sum(predictions)
        class_freq = predictions / total_preds
        
        # 计算逆频率权重
        inv_freq = 1.0 / (class_freq + 1e-10)
        weights = inv_freq / torch.sum(inv_freq) * self.num_classes
        
        # 更新权重
        self.weights = weights
    
    def forward(self, logits, targets):
        """前向传播计算损失"""
        # 基础交叉熵损失
        ce_loss = self.criterion(logits, targets)

        # 确保权重在与目标相同的设备上
        if self.weights.device != targets.device:
            weights = self.weights.to(targets.device)
        else:
            weights = self.weights
        
        # 获取目标的权重
        target_weights = weights[targets]
        
        # 加权损失
        weighted_loss = ce_loss * target_weights
        
        # 返回平均损失
        return weighted_loss.mean()

class EnhancedFeatureTransformer(nn.Module):
    """增强的特征转换器，支持跨tier特征适配"""
    def __init__(self, in_channels, out_channels, spatial_transform=None):
        super(EnhancedFeatureTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 通道转换
        self.channel_transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 32), out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空间变换
        self.spatial_transform = spatial_transform
        
        # 残差连接
        self.use_residual = (in_channels == out_channels and spatial_transform is None)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        identity = x
        
        # 通道转换
        x = self.channel_transform(x)
        
        # 空间变换
        if self.spatial_transform is not None:
            x = self.spatial_transform(x)
        
        # 注意力机制
        att = self.attention(x)
        x = x * att
        
        # 残差连接
        if self.use_residual:
            x = x + identity
            
        return x



# 重新设计ResNet - 添加特征范数控制和特征还原机制
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, 
                 KD=False, fedavg_base=False, tier=7, local_loss=False, groups_per_channel=32, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            # 使用GroupNorm替代BatchNorm
            norm_layer = lambda channels: nn.GroupNorm(
                num_groups=max(1, channels // groups_per_channel), 
                num_channels=channels
            )
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

        # 初始化基础卷积层和GroupNorm层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = norm_layer(self.inplanes)  # 使用GN替代BN
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 标记为特征提取层
        self.is_feature_extraction = True
        
        # tier=1的时候构造的客户端模型层数越多，tier=7时构造的层数越少
        if self.tier == 6 or self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer1 = self._make_layer(block, 16, layers[0], groups_per_channel=groups_per_channel)
        if self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1:# or self.local_v2:
             self.layer2 = self._make_layer(block, 16, layers[1], groups_per_channel=groups_per_channel)
        if self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer3 = self._make_layer(block, 32, layers[2], stride=2, groups_per_channel=groups_per_channel)
        if self.tier == 3 or self.tier == 2 or self.tier == 1:# or self.local_v2:
             self.layer4 = self._make_layer(block, 32, layers[3], stride=1, groups_per_channel=groups_per_channel)
        if self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer5 = self._make_layer(block, 64, layers[4], stride=2, groups_per_channel=groups_per_channel)
        if self.tier == 1:# or self.local_v2:
             self.layer6 = self._make_layer(block, 64, layers[5], stride=1, groups_per_channel=groups_per_channel)

        # 标准化输入维度
        if self.local_loss == True:
            # 为所有tier级别添加平均池化层
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 根据tier确定当前输出通道数
            if self.tier == 1 or self.tier == 2:
                in_features = 64 * block.expansion  # 256
            elif self.tier == 3 or self.tier == 4:
                in_features = 32 * block.expansion  # 128
            elif self.tier == 5 or self.tier == 6:
                in_features = 16 * block.expansion  # 64
            elif self.tier == 7:
                in_features = 16  # 16

            # 添加特征缩放层
            self.feature_scaling = FeatureScaling(target_norm=20.0, adaptive=True)

            # 为每个tier创建匹配其特征维度的分类器
            # 移除投影层，直接使用特征缩放后的原始特征维度
            self.projection = nn.Sequential(
                self.feature_scaling,
                AdaptiveFeatureNormalization(eps=1e-5, affine=True)
            )

            # 为当前tier创建匹配的分类器
            self.classifier = LayerNormClassifier(
                in_features=in_features,  # 使用实际特征维度，不再强制转为256
                num_classes=num_classes,
                hidden_dims=[128, 64], 
                is_local=True
            )

        self.KD = KD
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last GN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.gn3.weight, 0)  # 使用gn3替代bn3
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.gn2.weight, 0)  # 使用gn2替代bn2

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups_per_channel=32):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),  # 使用GN替代BN
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                          self.base_width, previous_dilation, norm_layer, groups_per_channel))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              norm_layer=norm_layer, groups_per_channel=groups_per_channel))

        return nn.Sequential(*layers)

    def get_feature_extraction_params(self):
        """获取特征提取层的参数"""
        feature_params = {}
        for name, param in self.named_parameters():
            if any(substr in name for substr in FEATURE_EXTRACTION_MODULES) and not any(substr in name for substr in NON_FEATURE_EXTRACTION_MODULES):
                feature_params[name] = param
        return feature_params
    
    def get_non_feature_extraction_params(self):
        """获取非特征提取层的参数"""
        non_feature_params = {}
        for name, param in self.named_parameters():
            if any(substr in name for substr in NON_FEATURE_EXTRACTION_MODULES):
                non_feature_params[name] = param
        return non_feature_params
    
    def get_feature_stats(self):
        """获取特征统计信息"""
        stats = {}
        
        # 收集特征缩放层的统计信息
        if hasattr(self, 'feature_scaling'):
            stats['feature_scaling'] = self.feature_scaling.get_norm_stats()
        
        # 收集特征归一化层的统计信息
        for name, module in self.named_modules():
            if isinstance(module, AdaptiveFeatureNormalization):
                stats[f'norm_{name}'] = module.get_feature_stats()
        
        return stats

    def forward(self, x):
        """
        前向传播，每个tier级别都会返回(logits, features)元组
        """
        # Tier 1：客户端有所有层
        if self.tier == 1:  
            x = self.conv1(x)
            x = self.gn1(x)
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
            x = self.gn1(x)
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
            x = self.gn1(x)
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
            x = self.gn1(x)
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
            x = self.gn1(x)
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
            x = self.gn1(x)
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
            x = self.gn1(x)
            x = self.relu(x)
            extracted_features = x  
            
            # 分类器输出
            x_pool = self.avgpool(x)
            x_flat = x_pool.view(x_pool.size(0), -1)
            x_proj = self.projection(x_flat)
            logits = self.classifier(x_proj)
            
            return logits, extracted_features          


# 统一服务器模型 - 完全重构
class UnifiedServerModel(nn.Module):
    """统一的服务器模型，处理所有tier客户端的特征"""
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, 
                 groups_per_channel=32, device=None, **kwargs):
        super(UnifiedServerModel, self).__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if norm_layer is None:
            # 使用GroupNorm替代BatchNorm
            norm_layer = lambda channels: nn.GroupNorm(
                num_groups=max(1, channels // groups_per_channel), 
                num_channels=channels
            )
        self._norm_layer = norm_layer
        self.inplanes = 16  # 初始通道数
        # 每层输入特征维度表
        # tier=7: conv1+gn1+relu -> 16
        # tier=6: layer1 -> 16*block.expansion = 64
        # tier=5: layer2 -> 16*block.expansion = 64
        # tier=4: layer3 -> 32*block.expansion = 128
        # tier=3: layer4 -> 32*block.expansion = 128
        # tier=2: layer5 -> 64*block.expansion = 256
        # tier=1: layer6 -> 64*block.expansion = 256
        self.tier_inplanes = {
            7: 16,
            6: 16 * block.expansion,
            5: 16 * block.expansion,
            4: 32 * block.expansion,
            3: 32 * block.expansion,
            2: 64 * block.expansion,
            1: 64 * block.expansion
        }
        
        # 特征适配层 - 将不同tier客户端输出的特征调整为对应层所需的尺寸和通道数
        # 为每个tier创建特征适配层 - 使用增强的特征转换器
        self.feature_adapters = nn.ModuleDict()
        # 每个tier的预期通道数
        tier_channels = {
            1: 64 * block.expansion,  # 通常是256
            2: 64 * block.expansion,  # 256
            3: 32 * block.expansion,  # 128
            4: 32 * block.expansion,  # 128
            5: 16 * block.expansion,  # 64
            6: 16 * block.expansion,  # 64
            7: 16                     # 16
        }

        for tier in range(1, 8):
            # 特征预处理 - 使用自适应归一化，显式指定通道数
            self.feature_adapters[f'process_{tier}'] = nn.Sequential(
                AdaptiveFeatureNormalization(num_channels=tier_channels[tier], eps=1e-5, affine=True)
            )
            
        # 设置服务器模型每层的初始通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        self.groups = groups
        self.base_width = width_per_group
        
        # 构建完整的服务器端模型
        self.layer1 = self._make_layer(block, 16, layers[0], groups_per_channel=groups_per_channel)
        self.layer2 = self._make_layer(block, 16, layers[1], groups_per_channel=groups_per_channel)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, groups_per_channel=groups_per_channel)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1, groups_per_channel=groups_per_channel)
        self.layer5 = self._make_layer(block, 64, layers[4], stride=2, groups_per_channel=groups_per_channel)
        self.layer6 = self._make_layer(block, 64, layers[5], stride=1, groups_per_channel=groups_per_channel)
        
        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 最终特征维度统一为 64 * block.expansion = 256
        final_dim = 64 * block.expansion
        
        # 最终特征处理
        self.final_feature_processing = nn.Sequential(
            AdaptiveFeatureNormalization(final_dim, mode='layer'),
            nn.LayerNorm(final_dim)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last GN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.gn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.gn2.weight, 0)

        # 将模型移动到指定设备
        self.to(self.device)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups_per_channel=32):
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
                          self.base_width, previous_dilation, norm_layer, groups_per_channel))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              norm_layer=norm_layer, groups_per_channel=groups_per_channel))

        return nn.Sequential(*layers)

    def forward(self, x, tier=7):
        """
        前向传播 - 根据客户端tier决定从哪层开始处理
        
        Args:
            x: 客户端特征
            tier: 客户端tier级别
            
        Returns:
            处理后的特征
        """
        # 保存原始形状信息
        original_shape = x.shape
        original_is_conv = len(original_shape) > 2
        
        # 特征预处理
        x = self.feature_adapters[f'process_{tier}'](x)
        
        # 根据tier选择开始层 - 确保特征是卷积格式
        if tier == 7:  # 客户端只有基础层
            # 这里直接使用卷积特征
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
        elif tier == 6:  # 客户端到layer1
            # 如果已经是卷积特征，直接使用
            if original_is_conv:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
            else:
                # 如果已经展平，需要先转回卷积格式
                # 这里可能需要更复杂的转换，简化起见使用注释
                raise ValueError(f"Tier {tier} expects convolutional features, but got flat features")
        elif tier == 5:  # 客户端到layer2
            if original_is_conv:
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
            else:
                raise ValueError(f"Tier {tier} expects convolutional features, but got flat features")
        elif tier == 4:  # 客户端到layer3
            if original_is_conv:
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
            else:
                raise ValueError(f"Tier {tier} expects convolutional features, but got flat features")
        elif tier == 3:  # 客户端到layer4
            if original_is_conv:
                x = self.layer5(x)
                x = self.layer6(x)
            else:
                raise ValueError(f"Tier {tier} expects convolutional features, but got flat features")
        elif tier == 2:  # 客户端到layer5
            if original_is_conv:
                x = self.layer6(x)
            else:
                raise ValueError(f"Tier {tier} expects convolutional features, but got flat features")
        # tier == 1的情况不需要额外处理，直接返回特征
        
        # 全局池化
        if len(x.shape) > 2:  # 如果是卷积特征
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        
        # 最终特征处理
        x = self.final_feature_processing(x)
        
        return x
    
    def get_feature_stats(self):
        """获取特征统计信息"""
        stats = {}
        
        # 收集特征归一化层的统计信息
        for name, module in self.named_modules():
            if isinstance(module, AdaptiveFeatureNormalization):
                stats[f'norm_{name}'] = module.get_feature_stats()
        
        return stats


# 优化的全局分类器 - 类别平衡和监控
class EnhancedGlobalClassifier(nn.Module):
    """增强的全局分类器，使用类别平衡机制和特征监控"""
    def __init__(self, in_features, num_classes, hidden_dims=None, dropout_rate=0.3, device=None):
        super(EnhancedGlobalClassifier, self).__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # 使用统一的分类器设计
        self.classifier = LayerNormClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            is_local=False,
            dropout_rate=dropout_rate,
            device=self.device
        ).to(self.device)
        
        # 类别预测统计
        self.register_buffer('class_predictions', torch.zeros(num_classes, device=self.device))
        self.register_buffer('num_samples_processed', torch.tensor(0, dtype=torch.long, device=self.device))
        
        # 特征统计
        self.register_buffer('feature_mean', torch.zeros(in_features, device=self.device))
        self.register_buffer('feature_var', torch.ones(in_features, device=self.device))
        
        # 类别平衡损失
        self.class_balanced_loss = ClassBalancedLoss(num_classes, device=self.device)
        
        # 标志变量，指示是否使用类别平衡损失
        self.use_balanced_loss = True
        
    def forward(self, x):
        """前向传播"""
        # 更新特征统计
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                
                # 更新移动平均
                self.feature_mean = 0.9 * self.feature_mean + 0.1 * batch_mean
                self.feature_var = 0.9 * self.feature_var + 0.1 * batch_var
        
        # 前向传播
        logits = self.classifier(x)
        
        # 记录类别预测统计
        if not self.training:
            with torch.no_grad():
                _, preds = torch.max(logits, dim=1)
                for c in range(self.class_predictions.size(0)):
                    self.class_predictions[c] += (preds == c).sum().item()
                self.num_samples_processed += preds.size(0)
                
                # 根据预测分布更新类别平衡损失
                if self.num_samples_processed > 100:
                    self.class_balanced_loss.update_weights_from_predictions(self.class_predictions)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """计算损失"""
        if self.use_balanced_loss:
            return self.class_balanced_loss(logits, targets)
        else:
            return F.cross_entropy(logits, targets)
    
    def get_prediction_stats(self):
        """获取预测统计"""
        total_preds = self.num_samples_processed.item()
        if total_preds > 0:
            class_distribution = self.class_predictions.cpu().numpy() / total_preds
        else:
            class_distribution = self.class_predictions.cpu().numpy()
        
        return {
            'class_distribution': class_distribution,
            'num_samples': total_preds
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.class_predictions.zero_()
        self.num_samples_processed.zero_()


# 创建统一分类器函数 - 修改
def create_unified_classifier(in_features, num_classes, is_global=True, device=None):
    """
    创建统一设计的分类器
    
    Args:
        in_features: 输入特征维度
        num_classes: 分类数
        is_global: 是否为全局分类器
        device: 计算设备
        
    Returns:
        统一设计的分类器
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_global:
        return EnhancedGlobalClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=[128, 64],
            dropout_rate=0.3,
            device=device
        )
    else:
        return LayerNormClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=[128, 64],
            is_local=True,
            dropout_rate=0.5,
            device=device
        )

# 创建统一服务器模型函数
def create_unified_server_model(block, layers, num_classes=10, groups_per_channel=32, device=None):
    """
    创建统一的服务器模型
    
    Args:
        block: 残差块类型
        layers: 每层的块数
        num_classes: 类别数
        groups_per_channel: GroupNorm的组数
        device: 计算设备
        
    Returns:
        server_model: 统一的服务器模型
        global_classifier: 全局分类器
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建统一服务器模型
    server_model = UnifiedServerModel(
        block=block, 
        layers=layers, 
        num_classes=num_classes, 
        groups_per_channel=groups_per_channel,
        device=device
    )
    
    # 创建全局分类器
    global_classifier = create_unified_classifier(
        in_features=64 * block.expansion,  # 通常是256
        num_classes=num_classes,
        is_global=True,
        device=device
    )
    
    return server_model, global_classifier


# 创建客户端模型函数 - 修改为支持tier
def create_tier_client_model(block, layers, tier, num_classes=10, groups_per_channel=32):
    """
    创建指定tier的客户端模型
    
    Args:
        block: 残差块类型
        layers: 每层的块数
        tier: 客户端tier级别
        num_classes: 类别数
        groups_per_channel: GroupNorm的组数
        
    Returns:
        客户端模型
    """
    client_model = ResNet(
        block=block,
        layers=layers,
        num_classes=num_classes,
        tier=tier,
        local_loss=True,
        groups_per_channel=groups_per_channel
    )
    
    return client_model


# 创建所有tier的客户端模型
def create_all_tier_client_models(block, layers, num_classes=10, groups_per_channel=32):
    """
    创建所有tier级别的客户端模型
    
    Args:
        block: 残差块类型
        layers: 每层的块数
        num_classes: 类别数
        groups_per_channel: GroupNorm的组数
        
    Returns:
        client_models: 所有tier级别的客户端模型字典
    """
    client_models = {}
    
    for tier in range(1, 8):
        client_models[tier] = create_tier_client_model(
            block=block,
            layers=layers,
            tier=tier,
            num_classes=num_classes,
            groups_per_channel=groups_per_channel
        )
    
    return client_models
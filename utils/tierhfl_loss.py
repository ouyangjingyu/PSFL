import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TierHFLLoss(nn.Module):
    """TierHFL损失函数，包含本地和全局损失平衡"""
    def __init__(self, alpha=0.5, lambda_feature=0.1, temperature=0.07, max_cache_size=10):
        super(TierHFLLoss, self).__init__()
        self.alpha = alpha  # 本地损失权重
        self.lambda_feature = lambda_feature  # 特征对齐权重
        self.temperature = temperature  # 对比学习温度参数
        self.ce = nn.CrossEntropyLoss()
        # 添加投影矩阵缓存字典
        self.projection_matrix_cache = {}
        self.max_cache_size = max_cache_size  # 添加缓存大小限制
        self.cache_hits = 0
        self.cache_misses = 0
    
    def forward(self, local_logits, global_logits, local_features, 
                global_features, targets, round_idx=0):
        """计算总损失
        
        Args:
            local_logits: 本地分类器输出
            global_logits: 全局分类器输出
            local_features: 本地特征
            global_features: 全局特征
            targets: 目标标签
            round_idx: 当前训练轮次
            
        Returns:
            总损失，本地损失，全局损失，特征对齐损失
        """
        # 分类损失
        local_loss = self.ce(local_logits, targets)
        global_loss = self.ce(global_logits, targets)
        
        # 特征对齐损失 - 使用简化的方法
        feature_loss = self._feature_alignment(local_features, global_features)
        
        # 根据轮次动态调整权重
        dynamic_alpha = self._adjust_alpha(self.alpha, round_idx)
        dynamic_lambda = self._adjust_lambda(self.lambda_feature, round_idx)
        
        # 总损失计算
        total_loss = dynamic_alpha * local_loss + (1 - dynamic_alpha) * global_loss + \
                    dynamic_lambda * feature_loss
                    
        return total_loss, local_loss, global_loss, feature_loss
    
    def _feature_alignment(self, local_feat, global_feat):
        """改进的特征对齐损失计算，使用缓存的投影矩阵"""
        # 确保特征是二维张量
        if local_feat.dim() > 2:
            local_feat = F.adaptive_avg_pool2d(local_feat, (1, 1))
            local_feat = local_feat.view(local_feat.size(0), -1)
        
        if global_feat.dim() > 2:
            global_feat = F.adaptive_avg_pool2d(global_feat, (1, 1))
            global_feat = global_feat.view(global_feat.size(0), -1)
        
        # 如果维度不匹配，使用或创建对应的投影矩阵
        if local_feat.size(1) != global_feat.size(1):
            # 创建维度组合的键
            dim_key = (local_feat.size(1), global_feat.size(1))
            device_str = str(local_feat.device)
            cache_key = f"{dim_key}_{device_str}"
            
            # 检查缓存中是否已有该维度组合的投影矩阵
            if cache_key in self.projection_matrix_cache:
                projection = self.projection_matrix_cache[cache_key]
                self.cache_hits += 1
            else:
                # 选择较小的维度作为共同维度
                common_dim = min(local_feat.size(1), global_feat.size(1))
                
                # 创建新的投影矩阵并添加到缓存
                projection = torch.randn(
                    max(local_feat.size(1), global_feat.size(1)), 
                    common_dim, 
                    device=local_feat.device
                )
                self.projection_matrix_cache[cache_key] = projection
                self.cache_misses += 1
                
                # 限制缓存大小
                self._clean_cache()
            
            # 应用投影
            if local_feat.size(1) > global_feat.size(1):
                local_feat = torch.matmul(local_feat, projection[:local_feat.size(1), :])
            else:
                global_feat = torch.matmul(global_feat, projection[:global_feat.size(1), :])
        
        # 特征归一化
        local_norm = F.normalize(local_feat, dim=1)
        global_norm = F.normalize(global_feat, dim=1)
        
        # 计算余弦相似度
        cos_sim = torch.sum(local_norm * global_norm, dim=1)
        
        # 转换为距离损失
        return torch.mean(1.0 - cos_sim)
    
    def _clean_cache(self):
        """清理旧的投影矩阵缓存"""
        if len(self.projection_matrix_cache) > self.max_cache_size:
            # 移除最早添加的项
            keys = list(self.projection_matrix_cache.keys())
            for key in keys[:len(keys) - self.max_cache_size]:
                del self.projection_matrix_cache[key]
    
    def _adjust_alpha(self, base_alpha, round_idx):
        """动态调整alpha值
        
        随着训练进行，增加本地损失权重(个性化)
        """
        if round_idx < 20:
            # 前20轮更注重全局损失
            return max(0.2, base_alpha - 0.2)
        elif round_idx > 80:
            # 后期更注重本地损失
            return min(0.8, base_alpha + 0.2)
        else:
            # 中期使用基础值
            return base_alpha
    
    def _adjust_lambda(self, base_lambda, round_idx):
        """动态调整lambda值
        
        前期保持较高的特征对齐权重，后期逐渐降低
        """
        if round_idx < 30:
            # 前期强调特征对齐
            return min(0.5, base_lambda * 2.0)
        elif round_idx > 70:
            # 后期降低特征对齐约束
            return max(0.05, base_lambda * 0.5)
        else:
            # 中期使用基础值
            return base_lambda
    # 在现有代码中添加to方法
    def to(self, device):
        """将损失函数移动到指定设备"""
        super().to(device)
        # 如果有projection_matrix，也将其移到对应设备
        if hasattr(self, 'projection_matrix'):
            self.projection_matrix = self.projection_matrix.to(device)
        return self

class GradientGuideModule:
    """梯度引导模块，平衡本地和服务器梯度"""
    def __init__(self, beta=0.7):
        self.beta = beta  # 服务器梯度权重初始值
    
    def guide_gradient(self, local_grad, server_grad, round_idx):
        """合并本地和服务器梯度
        
        Args:
            local_grad: 本地损失梯度
            server_grad: 服务器返回梯度
            round_idx: 当前训练轮次
            
        Returns:
            引导后的梯度
        """
        # 动态调整beta值
        dynamic_beta = self._adjust_beta(round_idx)
        
        # 合并梯度
        guided_grad = dynamic_beta * server_grad + (1 - dynamic_beta) * local_grad
        
        return guided_grad
    
    def _adjust_beta(self, round_idx):
        """根据训练进度调整beta值"""
        if round_idx < 20:
            # 前期更依赖服务器梯度
            return min(0.9, self.beta + 0.1)
        elif round_idx > 70:
            # 后期更依赖本地梯度
            return max(0.3, self.beta - 0.3)
        else:
            # 中期保持原值
            return self.beta

class ContrastiveLearningLoss(nn.Module):
    """对比学习损失，用于增强特征泛化性"""
    def __init__(self, temperature=0.07):
        super(ContrastiveLearningLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels, all_features=None, all_labels=None):
        """计算对比损失
        
        Args:
            features: 当前批次特征 [B, D]
            labels: 当前批次标签 [B]
            all_features: 所有客户端特征 [N, D]，如果为None则只使用当前批次
            all_labels: 所有客户端标签 [N]，如果为None则只使用当前批次
            
        Returns:
            对比损失
        """
        if all_features is None or all_labels is None:
            # 仅使用当前批次
            anchor_features = features
            anchor_labels = labels
            contrast_features = features
            contrast_labels = labels
        else:
            # 使用所有特征
            anchor_features = features
            anchor_labels = labels
            contrast_features = all_features
            contrast_labels = all_labels
        
        # 特征归一化
        anchor_features = F.normalize(anchor_features, dim=1)
        contrast_features = F.normalize(contrast_features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(anchor_features, contrast_features.T) / self.temperature
        
        # 创建标签匹配矩阵 (1表示相同类别)
        batch_size = anchor_labels.size(0)
        mask = torch.eq(anchor_labels.unsqueeze(1), contrast_labels.unsqueeze(0)).float()
        
        # 移除对角线 (自身匹配)
        if all_features is None:
            mask = mask.fill_diagonal_(0)
        
        # 对每个锚点特征，计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 生成正样本掩码和总掩码
        pos_mask = mask.bool()
        neg_mask = ~pos_mask
        
        # 计算分母 (所有样本的exp和)
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # 计算每个正样本对的损失
        pos_sim = torch.zeros_like(similarity_matrix)
        pos_sim.masked_fill_(pos_mask, 1)
        pos_exp = exp_sim * pos_sim
        
        # 正样本对的数量
        num_pos = pos_mask.sum(dim=1)
        
        # 防止除零
        num_pos = torch.clamp(num_pos, min=1)
        
        # 计算正样本对的损失 (负对数似然)
        pos_term = torch.sum(pos_exp, dim=1) / denominator
        pos_term = torch.clamp(pos_term, min=1e-8)
        loss = -torch.log(pos_term)
        
        # 平均损失
        return loss.mean()
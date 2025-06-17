import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class GradientProjector:
    """梯度投影器，处理共享层的梯度冲突"""
    def __init__(self, similarity_threshold=0.3, projection_frequency=5):
        self.similarity_threshold = similarity_threshold
        self.projection_frequency = projection_frequency
        self.batch_count = 0
        
    def should_project(self):
        """判断是否需要进行梯度投影"""
        self.batch_count += 1
        return self.batch_count % self.projection_frequency == 0
    
    def compute_cosine_similarity(self, grad1, grad2):
        """计算两个梯度的余弦相似度"""
        if grad1 is None or grad2 is None:
            return 1.0
        
        # 展平梯度
        g1_flat = grad1.flatten()
        g2_flat = grad2.flatten()
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(g1_flat.unsqueeze(0), g2_flat.unsqueeze(0))
        return cos_sim.item()
    
    def project_gradient(self, g_personal, g_global, alpha=0.5):
        """梯度投影：将个性化梯度投影到全局梯度方向"""
        if g_personal is None or g_global is None:
            return g_global if g_global is not None else g_personal
        
        # 展平梯度
        g_p_flat = g_personal.flatten()
        g_g_flat = g_global.flatten()
        
        # 计算投影
        dot_product = torch.dot(g_p_flat, g_g_flat)
        g_g_norm_sq = torch.dot(g_g_flat, g_g_flat)
        
        if g_g_norm_sq > 1e-8:
            projection = (dot_product / g_g_norm_sq) * g_g_flat
            projected_g_p = alpha * projection + (1 - alpha) * g_p_flat
            
            # 重塑为原始形状
            return projected_g_p.view_as(g_personal)
        else:
            return g_personal
    
    def process_shared_gradients(self, model, personal_loss, global_loss, alpha_stage=0.5):
        """处理共享层梯度冲突"""
        if not self.should_project():
            return False
        
        # 获取共享层参数
        shared_params = []
        for name, param in model.named_parameters():
            if 'shared_base' in name and param.requires_grad:
                shared_params.append((name, param))
        
        if not shared_params:
            return False
        
        # 计算个性化路径梯度
        personal_grads = torch.autograd.grad(
            personal_loss, [param for _, param in shared_params], 
            retain_graph=True, allow_unused=True
        )
        
        # 计算全局路径梯度  
        global_grads = torch.autograd.grad(
            global_loss, [param for _, param in shared_params],
            retain_graph=True, allow_unused=True
        )
        
        # 处理梯度冲突
        conflicts_resolved = 0
        for i, ((name, param), g_p, g_g) in enumerate(zip(shared_params, personal_grads, global_grads)):
            if g_p is not None and g_g is not None:
                # 计算相似度
                cos_sim = self.compute_cosine_similarity(g_p, g_g)
                
                # 如果存在冲突，进行投影
                if cos_sim < self.similarity_threshold:
                    projected_grad = self.project_gradient(g_p, g_g, alpha_stage)
                    param.grad = projected_grad
                    conflicts_resolved += 1
                else:
                    # 使用加权组合
                    param.grad = alpha_stage * g_p + (1 - alpha_stage) * g_g
        
        if conflicts_resolved > 0:
            logging.debug(f"解决了 {conflicts_resolved} 个梯度冲突")
        
        return True

class FeatureBalanceLoss(nn.Module):
    """特征平衡损失，确保共享层对两条路径都有用"""
    def __init__(self, temperature=1.0):
        super(FeatureBalanceLoss, self).__init__()
        self.temperature = temperature
        
    def compute_feature_importance(self, features, gradients):
        """计算特征重要性（使用梯度模长作为代理）"""
        if gradients is None:
            return torch.zeros(1, device=features.device)
        
        # 计算梯度模长
        grad_norm = torch.norm(gradients.flatten())
        return grad_norm
    
    def forward(self, shared_features, personal_gradients, global_gradients):
        """计算特征平衡损失"""
        # 计算两条路径的特征重要性
        personal_importance = self.compute_feature_importance(shared_features, personal_gradients)
        global_importance = self.compute_feature_importance(shared_features, global_gradients)
        
        # 计算平衡损失
        total_importance = personal_importance + global_importance + 1e-8
        personal_ratio = personal_importance / total_importance
        global_ratio = global_importance / total_importance
        
        # 目标是两者平衡（都接近0.5）
        balance_loss = torch.abs(personal_ratio - 0.5) + torch.abs(global_ratio - 0.5)
        
        return balance_loss

class EnhancedStagedLoss(nn.Module):
    """增强版分阶段损失函数"""
    def __init__(self):
        super(EnhancedStagedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.feature_balance_loss = FeatureBalanceLoss()
        self.gradient_projector = GradientProjector()
        
        # 损失权重
        self.lambda_balance = 0.1
        
    def stage1_loss(self, global_logits, targets, shared_features=None):
        """阶段1：纯全局特征学习"""
        global_loss = self.ce_loss(global_logits, targets)
        
        # 特征重要性损失（鼓励学习通用特征）
        feature_importance_loss = torch.tensor(0.0, device=global_logits.device)
        if shared_features is not None:
            # 鼓励特征的多样性（避免特征塌陷）
            features_flat = shared_features.flatten(1)
            feature_std = torch.std(features_flat, dim=1).mean()
            feature_importance_loss = -torch.log(feature_std + 1e-8)
        
        total_loss = global_loss + 0.05 * feature_importance_loss
        
        return total_loss, global_loss, feature_importance_loss
    
    def stage2_3_loss(self, local_logits, global_logits, targets, 
                     personal_gradients=None, global_gradients=None, 
                     shared_features=None, alpha=0.5):
        """阶段2&3：交替训练和精细调整"""
        local_loss = self.ce_loss(local_logits, targets)
        global_loss = self.ce_loss(global_logits, targets)
        
        # 特征平衡损失
        balance_loss = torch.tensor(0.0, device=global_logits.device)
        if shared_features is not None and personal_gradients is not None and global_gradients is not None:
            balance_loss = self.feature_balance_loss(shared_features, personal_gradients, global_gradients)
        
        # 组合损失
        total_loss = alpha * local_loss + (1 - alpha) * global_loss + self.lambda_balance * balance_loss
        
        return total_loss, local_loss, global_loss, balance_loss
    
    def apply_gradient_projection(self, model, personal_loss, global_loss, alpha_stage=0.5):
        """应用梯度投影"""
        return self.gradient_projector.process_shared_gradients(
            model, personal_loss, global_loss, alpha_stage
        )
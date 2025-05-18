import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedUnifiedLoss(nn.Module):
    """增强版统一损失函数，整合自适应控制器功能"""
    def __init__(self, init_alpha=0.5, init_beta=0.1):
        super(EnhancedUnifiedLoss, self).__init__()
        self.alpha = init_alpha  # 本地损失权重
        self.beta = init_beta    # 特征对齐权重
        self.ce = nn.CrossEntropyLoss()
        
        # 历史性能记录
        self.history = {
            'local_accuracy': [],
            'global_accuracy': [],
            'global_imbalance': []
        }
    
    def update_history(self, eval_results):
        """更新历史记录 - 从AdaptiveTrainingController借鉴"""
        if not eval_results:
            return
            
        local_accs = []
        global_accs = []
        imbalances = []
        
        for result in eval_results.values():
            if 'local_accuracy' in result:
                local_accs.append(result['local_accuracy'])
            if 'global_accuracy' in result:
                global_accs.append(result['global_accuracy'])
            if 'global_imbalance' in result:
                imbalances.append(result['global_imbalance'])
        
        # 计算平均值
        if local_accs:
            self.history['local_accuracy'].append(sum(local_accs) / len(local_accs))
        if global_accs:
            self.history['global_accuracy'].append(sum(global_accs) / len(global_accs))
        if imbalances:
            # 过滤掉infinity
            valid_imbalances = [i for i in imbalances if i != float('inf')]
            if valid_imbalances:
                self.history['global_imbalance'].append(sum(valid_imbalances) / len(valid_imbalances))
    
    def forward(self, global_logits, targets, local_loss, 
                personal_features=None, server_features=None, round_idx=0, total_rounds=100):
        """计算统一损失"""
        # 计算全局分类损失
        global_loss = self.ce(global_logits, targets)
        
        # 计算特征对齐损失
        feature_loss = torch.tensor(0.0, device=global_logits.device)
        if personal_features is not None and server_features is not None and self.beta > 0:
            feature_loss = self.compute_feature_alignment(personal_features, server_features)
        
        # 获取自适应权重
        alpha, beta = self.get_adaptive_weights(round_idx, total_rounds)
        
        # 计算总损失
        total_loss = (1 - alpha) * global_loss + alpha * local_loss + beta * feature_loss
        
        return total_loss, global_loss, feature_loss
    
    def compute_feature_alignment(self, personal_features, server_features):
        """计算特征对齐损失"""
        # 确保特征是二维
        if personal_features.dim() > 2:
            personal_features = F.adaptive_avg_pool2d(personal_features, (1, 1)).flatten(1)
        if server_features.dim() > 2:
            server_features = F.adaptive_avg_pool2d(server_features, (1, 1)).flatten(1)
        
        # 标准化特征
        personal_norm = F.normalize(personal_features, dim=1)
        server_norm = F.normalize(server_features, dim=1)
        
        # 计算余弦相似度，转换为损失
        cos_sim = torch.sum(personal_norm * server_norm, dim=1).mean()
        return 1.0 - cos_sim
    
    def get_adaptive_weights(self, round_idx, total_rounds):
        """计算自适应权重 - 整合AdaptiveTrainingController逻辑"""
        # 基本进度因子
        progress = round_idx / max(1, total_rounds)
        
        # 检查历史性能
        if len(self.history['local_accuracy']) >= 3 and len(self.history['global_accuracy']) >= 3:
            # 获取最近几轮的性能指标
            window_size = min(5, len(self.history['local_accuracy']))
            recent_local_acc = self.history['local_accuracy'][-window_size:]
            recent_global_acc = self.history['global_accuracy'][-window_size:]
            
            # 计算趋势
            local_trend = recent_local_acc[-1] - recent_local_acc[-3]
            global_trend = recent_global_acc[-1] - recent_global_acc[-3]
            
            # 当前性能
            current_local_acc = recent_local_acc[-1]
            current_global_acc = recent_global_acc[-1]
            
            # 计算不平衡度趋势
            if len(self.history['global_imbalance']) >= 3:
                recent_imbalance = self.history['global_imbalance'][-3:]
                imbalance_trend = recent_imbalance[-1] - recent_imbalance[0]
            else:
                imbalance_trend = 0
            
            # 根据性能差距和趋势调整alpha
            acc_gap = current_local_acc - current_global_acc
            
            # 高级自适应调整策略
            if global_trend < -1.0 and local_trend > 0:
                # 全局性能下降但本地性能上升，适度增加个性化权重
                alpha = min(0.7, self.alpha + 0.05)
            elif global_trend > 0.5 or (global_trend > 0 and local_trend < 0):
                # 更积极地降低alpha以促进全局学习
                alpha = max(0.2, self.alpha - 0.05)
            else:
                # 正常进度调整
                alpha = 0.3 + 0.4 * progress + 0.1 * np.tanh(acc_gap / 10)
            
            # 动态特征对齐权重
            if global_trend < 0 or imbalance_trend > 0.2:
                # 全局性能下降或不平衡度增加时，增强特征对齐
                beta = min(0.3, self.beta + 0.05)
            elif global_trend > 2.0 and imbalance_trend < 0:
                # 全局性能显著上升且不平衡度下降，适当减弱特征对齐
                beta = max(0.05, self.beta - 0.03)
            else:
                # 中期强，早期和后期弱
                beta = self.beta * (1 - abs(2 * progress - 1))
                
            # 更新内部状态
            self.alpha = alpha
            self.beta = beta
        else:
            # 简单进度调整
            alpha = 0.3 + 0.4 * progress
            beta = self.beta * (1 - abs(2 * progress - 1))
        
        return alpha, beta
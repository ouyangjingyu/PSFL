import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np
from collections import defaultdict
import torch.multiprocessing as mp
import threading
import queue
import logging
from collections import deque

class FeatureDistributionLoss(nn.Module):
    """特征分布一致性损失函数"""
    def __init__(self, reduction='mean'):
        super(FeatureDistributionLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, client_features, server_features):
        """计算客户端与服务器特征分布一致性损失"""
        # 归一化特征
        c_feat = F.normalize(client_features.view(client_features.size(0), -1), dim=1)
        s_feat = F.normalize(server_features, dim=1)
        
        # 计算特征相似度矩阵
        similarity = torch.mm(c_feat, s_feat.t())
        
        # 理想情况下，对角线元素应接近1，表示对应样本的特征相似
        target = torch.eye(c_feat.size(0), device=c_feat.device)
        
        # 计算损失 - 鼓励对应样本特征相似
        loss = F.mse_loss(similarity, target, reduction=self.reduction)
        
        return loss

class LossWeightController:
    """自适应损失权重控制器"""
    def __init__(self, init_local=0.5, init_global=0.5, init_feature=0.0):
        self.weights = {
            'local': init_local,
            'global': init_global,
            'feature': init_feature
        }
        self.history = {
            'local_acc': deque(maxlen=5),
            'global_acc': deque(maxlen=5)
        }
        
    def update_history(self, local_acc, global_acc):
        """更新准确率历史记录"""
        self.history['local_acc'].append(local_acc)
        self.history['global_acc'].append(global_acc)
        
        # 至少有3个历史记录时才调整
        if len(self.history['local_acc']) >= 3:
            self._adjust_weights()
            
    def _adjust_weights(self):
        """根据历史记录调整权重"""
        # 计算最近的趋势
        local_trend = self.history['local_acc'][-1] - self.history['local_acc'][0]
        global_trend = self.history['global_acc'][-1] - self.history['global_acc'][0]
        
        # 如果全局准确率停滞或下降，而本地准确率上升
        if global_trend < 0.5 and local_trend > 1.0:
            # 增加全局损失权重
            self.weights['global'] = min(0.7, self.weights['global'] + 0.05)
            # 减小本地损失权重
            self.weights['local'] = max(0.2, self.weights['local'] - 0.05)
            
        # 如果全局准确率增长显著
        elif global_trend > 2.0:
            # 如果特征一致性初始为0，开始引入特征一致性损失
            if self.weights['feature'] < 0.01:
                self.weights['feature'] = 0.1
                self.weights['local'] = max(0.2, self.weights['local'] - 0.05)
                self.weights['global'] = max(0.2, self.weights['global'] - 0.05)
        
        # 归一化权重
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
            
    def get_weights(self):
        """获取当前损失权重"""
        return self.weights

class AnomalyInterventionSystem:
    """异常自动干预系统"""
    def __init__(self, feature_monitor):
        self.feature_monitor = feature_monitor
        self.intervention_count = 0
        self.max_interventions = 5  # 每轮最大干预次数
        
    def check_and_intervene(self, client_model, server_model, classifier, params):
        """检查并执行干预"""
        # 重置干预计数
        self.intervention_count = 0
        
        # 获取最新异常
        anomalies = self.feature_monitor.get_latest_anomalies()
        for anomaly in anomalies:
            # 达到最大干预次数则停止
            if self.intervention_count >= self.max_interventions:
                break
                
            # 根据异常类型执行干预
            anomaly_type = anomaly.get('type')
            if anomaly_type == 'feature_norm_mismatch':
                self._intervene_norm_mismatch(client_model, params)
            elif anomaly_type == 'class_accuracy_imbalance':
                self._intervene_class_imbalance(classifier, params)
            elif anomaly_type == 'feature_norm_explosion':
                self._intervene_norm_explosion(client_model, server_model, params)
                
            self.intervention_count += 1
            
    def _intervene_norm_mismatch(self, client_model, params):
        """干预特征范数不匹配问题"""
        if hasattr(client_model, 'feature_scaling'):
            # 调整目标范数
            current = client_model.feature_scaling.target_norm
            client_model.feature_scaling.target_norm = current * 0.8
            print(f"[干预] 特征范数不匹配，调整客户端目标范数: {current} -> {client_model.feature_scaling.target_norm}")
            
            # 降低学习率
            params['lr'] = params.get('lr', 0.001) * 0.8
            print(f"[干预] 降低学习率至 {params['lr']}")
            
    def _intervene_class_imbalance(self, classifier, params):
        """干预类别准确率不平衡问题"""
        if hasattr(classifier, 'class_balanced_loss'):
            # 增强类别平衡力度
            beta = classifier.class_balanced_loss.beta
            classifier.class_balanced_loss.beta = min(0.9999, beta + 0.001)
            print(f"[干预] 增强类别平衡，调整beta: {beta} -> {classifier.class_balanced_loss.beta}")
            
    def _intervene_norm_explosion(self, client_model, server_model, params):
        """干预特征范数爆炸问题"""
        # 重置客户端特征处理参数
        if hasattr(client_model, 'feature_scaling'):
            client_model.feature_scaling.target_norm = 10.0
            print(f"[干预] 特征范数爆炸，重置目标范数为10.0")
        
        # 添加梯度裁剪
        params['clip_grad'] = 1.0
        print(f"[干预] 添加梯度裁剪 {params['clip_grad']}")

class StructureAwareAggregator:
    """结构感知聚合器"""
    def __init__(self, device='cuda'):
        self.device = device
        
    def aggregate(self, client_models_dict, client_tiers, client_weights=None):
        """聚合客户端模型，考虑结构差异"""
        if not client_models_dict:
            return {}

        # 初始化结果状态字典
        aggregated_model = {}

        # 添加类型转换函数
        def safe_add(target, source, weight):
            """安全地将加权源张量添加到目标张量"""
            weighted_source = source * weight
            
            # 如果目标为空，直接使用加权源
            if target is None:
                return weighted_source
            
            # 确保类型匹配
            if target.dtype != weighted_source.dtype:
                # 优先使用浮点型
                if target.dtype.is_floating_point or weighted_source.dtype.is_floating_point:
                    target_dtype = torch.float32
                    return target.to(target_dtype) + weighted_source.to(target_dtype)
                else:
                    # 两者都是整型，使用更大范围的类型
                    target_dtype = torch.long
                    return target.to(target_dtype) + weighted_source.to(target_dtype)
            else:
                # 类型已匹配
                return target + weighted_source
            
        # 获取默认权重
        if client_weights is None:
            client_weights = {cid: 1.0/len(client_models_dict) for cid in client_models_dict}
            
        # 按tier分组
        tier_groups = defaultdict(list)
        for client_id, model_state in client_models_dict.items():
            if client_id in client_tiers:
                tier = client_tiers[client_id]
                tier_groups[tier].append((client_id, model_state))
                
        # 首先在相同tier内聚合
        tier_aggregated = {}
        for tier, clients in tier_groups.items():
            if not clients:
                continue
                
            # 获取该tier的所有参数
            all_keys = set()
            for _, state in clients:
                all_keys.update(state.keys())
                
            # 聚合每个参数
            tier_state = {}
            for key in all_keys:
                # 跳过不需要聚合的参数
                if any(substr in key for substr in ['classifier', 'fc']):
                    continue
                    
                # 收集该参数和对应权重
                params = []
                weights = []
                for client_id, state in clients:
                    if key in state:
                        params.append(state[key].to(self.device))
                        weights.append(client_weights.get(client_id, 0))
                        
                # 如果有效参数，执行聚合
                if params and sum(weights) > 0:
                    # 归一化权重
                    norm_weights = [w/sum(weights) for w in weights]
                    
                    # 加权平均
                    agg_param = torch.zeros_like(params[0])
                    for p, w in zip(params, norm_weights):
                        agg_param = safe_add(agg_param, p, w)
                        
                    tier_state[key] = agg_param
                    
            tier_aggregated[tier] = tier_state
            
        # 然后进行跨tier参数映射和聚合
        final_state = {}
        
        # 获取所有参数名
        all_param_names = set()
        for tier_state in tier_aggregated.values():
            all_param_names.update(tier_state.keys())
            
        # 确定每个参数在哪些tier中存在
        param_to_tiers = defaultdict(list)
        for tier, state in tier_aggregated.items():
            for key in state:
                param_to_tiers[key].append(tier)
                
        # 聚合每个参数
        for param_name in all_param_names:
            tiers = param_to_tiers[param_name]
            
            if len(tiers) == 1:
                # 只存在于一个tier，直接使用
                final_state[param_name] = tier_aggregated[tiers[0]][param_name]
            else:
                # 存在于多个tier，加权聚合
                # 高tier权重更高
                tier_importance = {t: 1.0 + (7-t)*0.1 for t in tiers}
                total_imp = sum(tier_importance.values())
                tier_weights = {t: imp/total_imp for t, imp in tier_importance.items()}
                
                # 初始化聚合参数
                agg_param = torch.zeros_like(
                    tier_aggregated[tiers[0]][param_name]
                )
                
                # 加权聚合
                for tier in tiers:
                    if param_name in tier_aggregated[tier]:
                        agg_param += tier_aggregated[tier][param_name] * tier_weights[tier]
                        
                final_state[param_name] = agg_param
                
        return final_state

class AdaptiveTierWeightAdjuster:
    """自适应Tier贡献权重调整器"""
    def __init__(self):
        # 初始tier权重
        self.tier_weights = {i: 1.0 for i in range(1, 8)}
        # 每个tier的性能历史
        self.tier_performance = {i: deque(maxlen=5) for i in range(1, 8)}
        # 每轮贡献调整率
        self.adjust_rate = 0.1
        
    def update_tier_performance(self, tier, accuracy):
        """更新tier性能记录"""
        if tier in self.tier_performance:
            self.tier_performance[tier].append(accuracy)
            
    def adjust_weights(self):
        """根据性能历史动态调整tier权重"""
        # 计算每个tier的平均性能
        tier_avg_perf = {}
        for tier, history in self.tier_performance.items():
            if history:
                tier_avg_perf[tier] = sum(history) / len(history)
            else:
                # 没有历史数据的使用默认值
                tier_avg_perf[tier] = 70.0  # 假设基础准确率70%
                
        # 如果所有tier都有性能数据
        if len(tier_avg_perf) >= 3:  # 至少有3个tier有数据
            # 计算性能的相对比例
            total_perf = sum(tier_avg_perf.values())
            if total_perf > 0:
                relative_perf = {t: p/total_perf for t, p in tier_avg_perf.items()}
                
                # 根据相对性能调整权重
                for tier, rel_perf in relative_perf.items():
                    # 计算理想权重
                    ideal_weight = 0.5 + 0.5 * rel_perf  # 基础权重0.5 + 性能比例0.5
                    
                    # 缓慢调整当前权重向理想权重
                    current = self.tier_weights.get(tier, 1.0)
                    adjusted = current + (ideal_weight - current) * self.adjust_rate
                    
                    # 更新权重
                    self.tier_weights[tier] = adjusted
                    
        # 归一化权重，确保总和为tier数量
        total_weight = sum(self.tier_weights.values())
        if total_weight > 0:
            tier_count = len(self.tier_weights)
            for tier in self.tier_weights:
                self.tier_weights[tier] = self.tier_weights[tier] * tier_count / total_weight
                
    def get_client_weight_modifier(self, client_id, tier):
        """获取客户端权重修正因子"""
        # 默认修正因子为1
        return self.tier_weights.get(tier, 1.0)



# 优化的拆分学习训练函数 - 改进异构客户端处理
def train_client_with_unified_server(client_id, client_model, unified_server_model, 
                                    global_classifier, device, client_manager, round_idx, 
                                    local_epochs=1, feature_monitor=None, 
                                    loss_weight_controller=None, 
                                    feature_dist_loss=None,
                                    training_params=None, **kwargs):
    """
    在统一服务器模型架构下的客户端训练函数
    
    Args:
        client_id: 客户端ID
        client_model: 客户端模型
        unified_server_model: 统一服务器模型
        global_classifier: 全局分类器
        device: 计算设备
        client_manager: 客户端管理器
        round_idx: 当前轮次
        local_epochs: 本地训练轮数
        feature_monitor: 特征监控器
    """
    try:
        # 获取客户端
        client = client_manager.get_client(client_id)
        if client is None:
            return {'error': f"客户端 {client_id} 不存在"}
        
        # 设置设备
        client.device = device
        print(f"客户端 {client_id} 使用设备: {device} - Tier {client.tier}")
        
        # 获取损失权重 - 如果提供了控制器则使用动态权重
        if loss_weight_controller is not None:
            loss_weights = loss_weight_controller.get_weights()
            local_weight = loss_weights.get('local', 0.5)
            global_weight = loss_weights.get('global', 0.5)
            feature_weight = loss_weights.get('feature', 0.0)
        else:
            # 默认权重
            local_weight, global_weight, feature_weight = 0.5, 0.5, 0.0

        # 确保模型在正确的设备上
        client_model = client_model.to(device)
        unified_server_model = unified_server_model.to(device)
        global_classifier = global_classifier.to(device)
        
        # 获取优化的训练策略
        training_strategy = client.training_strategy
        
        # 创建优化器
        client_optimizer = training_strategy.create_optimizer(client_model)
        server_optimizer = training_strategy.create_optimizer(unified_server_model)
        classifier_optimizer = training_strategy.create_optimizer(global_classifier)
        
        # 使用训练参数中的学习率
        if training_params and 'lr' in training_params:
            for opt in [client_optimizer, server_optimizer, classifier_optimizer]:
                for param_group in opt.param_groups:
                    param_group['lr'] = training_params['lr']
        # 设置损失函数
        criterion = nn.CrossEntropyLoss()
        # 创建特征分布一致性损失
        feature_dist_loss_fn = feature_dist_loss
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练统计
        stats = defaultdict(float)
        
        # 特征监控数据
        feature_stats = defaultdict(list)
        
        # 本地训练
        for epoch in range(local_epochs):
            # 记录本轮统计信息
            epoch_local_loss = 0.0
            epoch_global_loss = 0.0
            epoch_total_loss = 0.0
            
            local_correct = 0
            global_correct = 0
            total_samples = 0
            feature_dist_value = 0.0
            clip_norm = 5.0  # 默认值
            
            # 训练一个epoch
            for batch_idx, (data, target) in enumerate(client.train_data):
                data, target = data.to(device), target.to(device)
                
                # 清零所有梯度
                client_optimizer.zero_grad()
                server_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                
                # 1. 客户端前向传播
                local_logits, client_features = client_model(data)
                
                # 记录客户端特征统计信息
                if feature_monitor and batch_idx % 5 == 0:
                    with torch.no_grad():
                        feature_norm = torch.norm(client_features.view(client_features.size(0), -1), dim=1).mean().item()
                        feature_stats['client_feature_norm'].append(feature_norm)
                
                # 2. 本地损失计算
                local_loss = criterion(local_logits, target)
                epoch_local_loss += local_loss.item()
                
                # 3. 计算本地准确率
                _, local_preds = torch.max(local_logits, dim=1)
                local_correct += (local_preds == target).sum().item()
                
                # 4. 服务器处理 - 根据客户端tier处理特征
                server_features = unified_server_model(client_features, tier=client.tier)
                
                # 记录服务器特征统计信息
                if feature_monitor and batch_idx % 5 == 0:
                    with torch.no_grad():
                        server_norm = torch.norm(server_features, dim=1).mean().item()
                        feature_stats['server_feature_norm'].append(server_norm)
                
                # 5. 全局分类
                global_logits = global_classifier(server_features)
                
                # 特征分布一致性损失
                if feature_dist_loss_fn is not None and feature_weight > 0:
                    feature_dist_value = feature_dist_loss_fn(client_features, server_features)

                # 6. 全局损失计算 - 使用类别平衡损失
                global_loss = global_classifier.compute_loss(global_logits, target)
                epoch_global_loss += global_loss.item()
                
                # 7. 计算全局准确率
                _, global_preds = torch.max(global_logits, dim=1)
                global_correct += (global_preds == target).sum().item()
                
                # 总损失 - 加入特征分布一致性损失
                total_loss = (local_weight * local_loss + 
                            global_weight * global_loss + 
                            feature_weight * feature_dist_value)
                # 记录特征分布损失
                if feature_weight > 0:
                    epoch_feature_dist_loss = feature_dist_value.item()
                else:
                    epoch_feature_dist_loss = 0.0

                epoch_total_loss += total_loss.item()
                
                # 9. 反向传播和参数更新
                total_loss.backward()
                
                # 10. 梯度裁剪
                if training_params and 'clip_grad' in training_params and training_params['clip_grad'] is not None:
                    clip_norm = training_params['clip_grad']
                    
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=clip_norm)
                torch.nn.utils.clip_grad_norm_(unified_server_model.parameters(), max_norm=clip_norm)
                torch.nn.utils.clip_grad_norm_(global_classifier.parameters(), max_norm=clip_norm)
                                
                # 11. 优化器步进
                client_optimizer.step()
                server_optimizer.step()
                classifier_optimizer.step()
                
                # 统计样本数
                total_samples += target.size(0)
            
            # 计算平均损失和准确率
            if total_samples > 0:
                epoch_local_loss /= len(client.train_data)
                epoch_global_loss /= len(client.train_data)
                epoch_total_loss /= len(client.train_data)
                
                epoch_local_acc = 100.0 * local_correct / total_samples
                epoch_global_acc = 100.0 * global_correct / total_samples
                
                # 输出训练信息
                print(f"客户端 {client_id} (Tier {client.tier}) - Epoch {epoch+1}/{local_epochs}: "
                      f"本地损失: {epoch_local_loss:.4f}, 本地准确率: {epoch_local_acc:.2f}%, "
                      f"全局损失: {epoch_global_loss:.4f}, 全局准确率: {epoch_global_acc:.2f}%")
                
                # 更新统计信息
                stats['local_loss'] += epoch_local_loss / local_epochs
                stats['global_loss'] += epoch_global_loss / local_epochs
                stats['total_loss'] += epoch_total_loss / local_epochs
                stats['local_accuracy'] += epoch_local_acc / local_epochs
                stats['global_accuracy'] += epoch_global_acc / local_epochs
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        # 获取客户端模型的特征统计信息
        client_feature_stats = client_model.get_feature_stats() if hasattr(client_model, 'get_feature_stats') else {}
        
        # 获取服务器模型的特征统计信息
        server_feature_stats = unified_server_model.get_feature_stats() if hasattr(unified_server_model, 'get_feature_stats') else {}
        
        # 获取全局分类器的预测统计
        classifier_stats = global_classifier.get_prediction_stats() if hasattr(global_classifier, 'get_prediction_stats') else {}
        
        # 构建结果
        result = {
            # 训练损失和准确率
            'local_loss': stats['local_loss'],
            'global_loss': stats['global_loss'],
            'total_loss': stats['total_loss'],
            'local_accuracy': stats['local_accuracy'],
            'global_accuracy': stats['global_accuracy'],
            
            # 兼容旧代码
            'loss': stats['total_loss'],
            'accuracy': stats['global_accuracy'],
            
            # 训练时间和数据量
            'time': training_time,
            'data_size': len(client.train_data.dataset) if hasattr(client.train_data, 'dataset') else 0,
            
            # 特征统计信息
            'feature_stats': {
                'client': client_feature_stats,
                'server': server_feature_stats,
                'classifier': classifier_stats,
                'monitoring': feature_stats
            },
            
            # 模型状态
            'client_model_state': client_model.state_dict(),
            'used_tier': client.tier
        }

        # 在结果中添加特征分布损失
        if feature_weight > 0:
            result['feature_dist_loss'] = stats.get('feature_dist_loss', 0)
        
        return result
    
    except Exception as e:
        import traceback
        error_msg = f"客户端 {client_id} 训练失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}


def evaluate_client_with_unified_server(client_id, client_model, unified_server_model, 
                                       global_classifier, device, client_manager, 
                                       feature_monitor=None, **kwargs):
    """
    在统一服务器模型架构下的客户端评估函数
    
    Args:
        client_id: 客户端ID
        client_model: 客户端模型
        unified_server_model: 统一服务器模型
        global_classifier: 全局分类器
        device: 计算设备
        client_manager: 客户端管理器
        feature_monitor: 特征监控器
    """
    try:
        # 获取客户端
        client = client_manager.get_client(client_id)
        if client is None:
            return {'error': f"客户端 {client_id}不存在"}
        
        # 确保模型在正确的设备上
        client_model = client_model.to(device)
        unified_server_model = unified_server_model.to(device)
        global_classifier = global_classifier.to(device)
        
        # 设置模型为评估模式
        client_model.eval()
        unified_server_model.eval()
        global_classifier.eval()
        
        # 初始化统计变量
        local_loss = 0.0
        global_loss = 0.0
        total_loss = 0.0
        
        local_correct = 0
        global_correct = 0
        total_samples = 0
        
        # 每个类别的准确率统计
        local_class_correct = [0] * 10  # 假设10个类别
        local_class_total = [0] * 10
        global_class_correct = [0] * 10
        global_class_total = [0] * 10
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 特征监控数据
        feature_stats = defaultdict(list)
        
        # 评估开始时间
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(client.test_data):
                data, target = data.to(device), target.to(device)
                
                # 1. 客户端前向传播
                local_logits, client_features = client_model(data)
                
                # 记录客户端特征统计信息
                if feature_monitor and batch_idx % 5 == 0:
                    feature_norm = torch.norm(client_features.view(client_features.size(0), -1), dim=1).mean().item()
                    feature_stats['client_feature_norm'].append(feature_norm)
                
                # 2. 本地损失计算
                local_batch_loss = criterion(local_logits, target)
                local_loss += local_batch_loss.item()
                
                # 3. 计算本地准确率
                _, local_preds = torch.max(local_logits, dim=1)
                local_correct += (local_preds == target).sum().item()
                
                # 更新每个类别的本地准确率统计
                for i in range(len(target)):
                    label = target[i].item()
                    local_class_total[label] += 1
                    if local_preds[i] == label:
                        local_class_correct[label] += 1
                
                # 4. 服务器处理 - 根据客户端tier处理特征
                server_features = unified_server_model(client_features, tier=client.tier)
                
                # 记录服务器特征统计信息
                if feature_monitor and batch_idx % 5 == 0:
                    server_norm = torch.norm(server_features, dim=1).mean().item()
                    feature_stats['server_feature_norm'].append(server_norm)
                
                # 5. 全局分类
                global_logits = global_classifier(server_features)
                
                # 6. 全局损失计算
                global_batch_loss = criterion(global_logits, target)
                global_loss += global_batch_loss.item()
                
                # 7. 计算全局准确率
                _, global_preds = torch.max(global_logits, dim=1)
                global_correct += (global_preds == target).sum().item()
                
                # 更新每个类别的全局准确率统计
                for i in range(len(target)):
                    label = target[i].item()
                    global_class_total[label] += 1
                    if global_preds[i] == label:
                        global_class_correct[label] += 1
                
                # 8. 总损失
                total_loss += (local_batch_loss.item() + global_batch_loss.item()) / 2
                
                # 统计样本数
                total_samples += target.size(0)
        
        # 计算评估时间
        eval_time = time.time() - start_time
        
        # 计算平均损失和准确率
        test_len = len(client.test_data)
        if test_len > 0:
            local_loss /= test_len
            global_loss /= test_len
            total_loss /= test_len
        
        if total_samples > 0:
            local_accuracy = 100.0 * local_correct / total_samples
            global_accuracy = 100.0 * global_correct / total_samples
        else:
            local_accuracy = 0.0
            global_accuracy = 0.0
        
        # 计算每个类别的准确率
        local_per_class_accuracy = []
        global_per_class_accuracy = []
        
        for i in range(len(local_class_total)):
            if local_class_total[i] > 0:
                local_per_class_accuracy.append(100.0 * local_class_correct[i] / local_class_total[i])
            else:
                local_per_class_accuracy.append(0.0)
                
            if global_class_total[i] > 0:
                global_per_class_accuracy.append(100.0 * global_class_correct[i] / global_class_total[i])
            else:
                global_per_class_accuracy.append(0.0)
        
        # 构建结果
        result = {
            # 评估损失和准确率
            'local_loss': local_loss,
            'global_loss': global_loss,
            'total_loss': total_loss,
            'local_accuracy': local_accuracy,
            'global_accuracy': global_accuracy,
            
            # 每个类别的准确率
            'local_per_class_accuracy': local_per_class_accuracy,
            'global_per_class_accuracy': global_per_class_accuracy,
            
            # 兼容旧代码
            'loss': total_loss,
            'accuracy': global_accuracy,
            
            # 评估时间和样本数
            'time': eval_time,
            'total_samples': total_samples,
            
            # 特征统计
            'feature_stats': feature_stats
        }
        
        return result
    
    except Exception as e:
        import traceback
        error_msg = f"客户端 {client_id} 评估失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}

class AdaptiveFeatureNormController:
    """动态调整特征范数目标值"""
    def __init__(self):
        self.client_norms = defaultdict(list)
        self.server_norms = defaultdict(list)
        self.tier_targets = {i: 20.0 for i in range(1, 8)}  # 初始目标值
        
    def update(self, client_id, tier, client_norm, server_norm):
        """记录并更新特征范数信息"""
        self.client_norms[(client_id, tier)].append(client_norm)
        self.server_norms[(client_id, tier)].append(server_norm)
        
        # 保持固定窗口大小
        max_history = 20
        if len(self.client_norms[(client_id, tier)]) > max_history:
            self.client_norms[(client_id, tier)] = self.client_norms[(client_id, tier)][-max_history:]
            self.server_norms[(client_id, tier)] = self.server_norms[(client_id, tier)][-max_history:]
            
        # 计算平均范数比例
        c_norms = self.client_norms[(client_id, tier)]
        s_norms = self.server_norms[(client_id, tier)]
        if len(c_norms) >= 5 and len(s_norms) >= 5:
            avg_c_norm = sum(c_norms[-5:]) / 5
            avg_s_norm = sum(s_norms[-5:]) / 5
            ratio = avg_c_norm / (avg_s_norm + 1e-6)
            
            # 根据比例调整目标值
            if ratio > 10.0:  # 客户端范数过大
                self.tier_targets[tier] *= 0.9
            elif ratio < 0.1:  # 客户端范数过小
                self.tier_targets[tier] *= 1.1
                
    def get_target(self, tier):
        """获取指定tier的目标范数"""
        return self.tier_targets.get(tier, 20.0)

# 特征监控器 - 跟踪并分析特征变化
class FeatureMonitor:
    """特征分布监控器，实时跟踪特征变化"""
    def __init__(self):
        # 存储不同客户端的特征统计信息
        self.client_feature_stats = defaultdict(list)
        self.server_feature_stats = defaultdict(list)
        self.classifier_stats = defaultdict(list)
        
        # 存储异常检测结果
        self.anomalies = []
        
        # 跟踪每轮全局统计
        self.round_stats = []

        # 在FeatureMonitor初始化中添加
        self.norm_controller = AdaptiveFeatureNormController()
    
    def update_from_training_result(self, client_id, result, round_idx):
        """从训练结果更新统计信息"""
        if 'feature_stats' not in result:
            return
        
        # 提取特征统计信息
        feature_stats = result['feature_stats']
        
        # 更新客户端特征统计
        if 'client' in feature_stats:
            client_stats = feature_stats['client']
            for key, value in client_stats.items():
                self.client_feature_stats[(client_id, key, round_idx)] = value
        
        # 更新服务器特征统计
        if 'server' in feature_stats:
            server_stats = feature_stats['server']
            for key, value in server_stats.items():
                self.server_feature_stats[(client_id, key, round_idx)] = value
        
        # 更新分类器统计
        if 'classifier' in feature_stats:
            classifier_stats = feature_stats['classifier']
            for key, value in classifier_stats.items():
                self.classifier_stats[(client_id, key, round_idx)] = value

        
        # 更新监控数据
        if 'monitoring' in feature_stats and feature_stats['monitoring']:
            monitoring_data = feature_stats['monitoring']
            
            # 检测客户端-服务器特征范数差异
            if 'client_feature_norm' in monitoring_data and 'server_feature_norm' in monitoring_data:
                client_norms = monitoring_data['client_feature_norm']
                server_norms = monitoring_data['server_feature_norm']
                if client_norms and server_norms:
                    avg_client_norm = sum(client_norms) / len(client_norms)
                    avg_server_norm = sum(server_norms) / len(server_norms)
                    self.norm_controller.update(client_id, result.get('used_tier', 1), 
                                            avg_client_norm, avg_server_norm)
                # 确保两个列表长度相同
                min_len = min(len(client_norms), len(server_norms))
                if min_len > 0:
                    # 计算平均范数差异比例
                    norm_ratios = [c/s if s > 0 else float('inf') for c, s in zip(client_norms[:min_len], server_norms[:min_len])]
                    avg_ratio = sum(filter(lambda x: x != float('inf'), norm_ratios)) / len(list(filter(lambda x: x != float('inf'), norm_ratios))) if norm_ratios else 0
                    
                    # 检测异常
                    if avg_ratio > 50:  # 客户端特征范数是服务器特征范数的50倍以上
                        self.anomalies.append({
                            'type': 'feature_norm_mismatch',
                            'client_id': client_id,
                            'round': round_idx,
                            'ratio': avg_ratio,
                            'message': f"客户端特征范数异常：客户端/服务器比例为{avg_ratio:.2f}"
                        })
    
    def update_from_evaluation_result(self, client_id, result, round_idx):
        """从评估结果更新统计信息"""
        # 检测分类不平衡问题
        if 'global_per_class_accuracy' in result:
            global_accs = result['global_per_class_accuracy']
            
            # 计算最大和最小准确率
            if global_accs:
                max_acc = max(global_accs)
                min_acc = min(global_accs)
                
                # 检测极端不平衡
                if min_acc < 5.0 and max_acc > 50.0:  # 最小准确率低于5%，最大准确率高于50%
                    self.anomalies.append({
                        'type': 'class_accuracy_imbalance',
                        'client_id': client_id,
                        'round': round_idx,
                        'max_acc': max_acc,
                        'min_acc': min_acc,
                        'message': f"类别准确率极度不平衡：最高{max_acc:.2f}%，最低{min_acc:.2f}%"
                    })
        
        # 检查特征范数问题
        if 'feature_stats' in result and 'client_feature_norm' in result['feature_stats']:
            client_norms = result['feature_stats']['client_feature_norm']
            if client_norms and any(norm > 1000 for norm in client_norms):
                self.anomalies.append({
                    'type': 'feature_norm_explosion',
                    'client_id': client_id,
                    'round': round_idx,
                    'message': f"特征范数爆炸：客户端特征范数超过1000"
                })
    
    def update_global_stats(self, global_model_accuracy, round_idx):
        """更新全局模型统计信息"""
        self.round_stats.append({
            'round': round_idx,
            'global_accuracy': global_model_accuracy,
            'anomalies_count': sum(1 for a in self.anomalies if a['round'] == round_idx)
        })
    
    def get_latest_anomalies(self, count=5):
        """获取最新的异常检测结果"""
        return sorted(self.anomalies, key=lambda x: x['round'], reverse=True)[:count]
    
    def get_analysis_report(self):
        """生成分析报告"""
        # 按轮次对异常进行分组
        anomalies_by_round = defaultdict(list)
        for anomaly in self.anomalies:
            anomalies_by_round[anomaly['round']].append(anomaly)
        
        # 异常类型统计
        anomaly_types = defaultdict(int)
        for anomaly in self.anomalies:
            anomaly_types[anomaly['type']] += 1
        
        # 受影响客户端统计
        affected_clients = defaultdict(int)
        for anomaly in self.anomalies:
            affected_clients[anomaly['client_id']] += 1
        
        # 准备报告
        report = {
            'total_anomalies': len(self.anomalies),
            'anomalies_by_round': dict(anomalies_by_round),
            'anomaly_types': dict(anomaly_types),
            'affected_clients': dict(affected_clients),
            'latest_anomalies': self.get_latest_anomalies()
        }
        
        return report


# 统一服务器架构下的模型聚合器
class UnifiedModelAggregator:
    """统一服务器架构下的模型聚合器，平衡不同tier客户端的贡献"""
    def __init__(self, device='cuda'):
        self.device = device
        self.client_performance = {}  # 跟踪客户端性能
    
    def update_client_performance(self, client_id, accuracy):
        """更新客户端性能"""
        self.client_performance[client_id] = accuracy
    
    def aggregate_client_models(self, client_models_dict, client_weights=None, client_tiers=None):
        """
        聚合客户端模型
        
        Args:
            client_models_dict: 客户端模型字典，键为客户端ID，值为模型状态字典
            client_weights: 客户端权重字典，键为客户端ID，值为权重
            client_tiers: 客户端tier信息，键为客户端ID，值为tier级别
            
        Returns:
            aggregated_model: 聚合后的模型状态字典
        """
        if not client_models_dict:
            return {}
        
        # 默认权重 - 如果未提供，则使用平均权重或基于性能的权重
        if client_weights is None:
            # 如果有性能数据，根据性能分配权重
            if self.client_performance:
                total_acc = sum(self.client_performance.values())
                if total_acc > 0:
                    client_weights = {client_id: self.client_performance.get(client_id, 0) / total_acc 
                                      for client_id in client_models_dict.keys()}
                else:
                    # 回退到平均权重
                    client_weights = {client_id: 1.0 / len(client_models_dict) 
                                      for client_id in client_models_dict.keys()}
            else:
                # 回退到平均权重
                client_weights = {client_id: 1.0 / len(client_models_dict) 
                                  for client_id in client_models_dict.keys()}
        
        # 平衡不同tier客户端的贡献 - 使用tier修正因子
        if client_tiers:
            # Tier修正因子 - 让高tier客户端(tier=1)贡献稍多，低tier客户端(tier=7)贡献稍少
            tier_correction = {
                1: 1.2,  # tier 1贡献权重提高20%
                2: 1.15,
                3: 1.1,
                4: 1.0,  # tier 4保持不变
                5: 0.9,
                6: 0.85,
                7: 0.8   # tier 7贡献权重降低20%
            }
            
            # 应用tier修正因子
            for client_id in client_weights:
                if client_id in client_tiers:
                    tier = client_tiers[client_id]
                    client_weights[client_id] *= tier_correction.get(tier, 1.0)
            
            # 重新归一化权重
            total_weight = sum(client_weights.values())
            if total_weight > 0:
                for client_id in client_weights:
                    client_weights[client_id] /= total_weight
        
        # 收集所有参数的键
        all_keys = set()
        for model_state in client_models_dict.values():
            all_keys.update(model_state.keys())
        
        # 初始化聚合模型
        aggregated_model = {}
        
        # 聚合每个参数
        for key in all_keys:
            # 跳过特定层或参数 - 比如本地分类器参数
            if any(substr in key for substr in ['classifier', 'fc']):
                continue
            
            # 收集拥有该参数的客户端及其权重
            key_models = []
            key_weights = []
            
            for client_id, model_state in client_models_dict.items():
                if key in model_state:
                    # 将参数移到聚合设备
                    param = model_state[key].to(self.device)
                    key_models.append(param)
                    key_weights.append(client_weights.get(client_id, 0))
            
            # 如果没有客户端拥有该参数，跳过
            if not key_models:
                continue
            
            # 归一化权重
            weight_sum = sum(key_weights)
            if weight_sum > 0:
                norm_weights = [w / weight_sum for w in key_weights]
            else:
                # 如果所有权重都为0，使用平均权重
                norm_weights = [1.0 / len(key_weights)] * len(key_weights)
            
            # 聚合参数 - 加权平均
            aggregated_param = torch.zeros_like(key_models[0])
            for param, weight in zip(key_models, norm_weights):
                aggregated_param += param * weight
            
            # 保存聚合后的参数
            aggregated_model[key] = aggregated_param
        
        return aggregated_model
    
    def aggregate_server_model(self, unified_server_model, client_results):
        """
        聚合服务器端模型参数
        
        Args:
            unified_server_model: 统一服务器模型
            client_results: 客户端训练结果
            
        Returns:
            aggregated_server: 聚合后的服务器模型状态字典
        """
        # 从训练结果中分离出服务器模型参数
        server_params = {}
        
        # 使用服务器模型自身的状态字典初始化
        if hasattr(unified_server_model, 'state_dict'):
            server_params = unified_server_model.state_dict()
        
        # 每个客户端贡献服务器参数的权重 - 根据客户端global_accuracy
        client_weights = {}
        
        for client_id, result in client_results.items():
            # 确保结果有效且包含global_accuracy
            if isinstance(result, dict) and 'global_accuracy' in result:
                client_weights[client_id] = result['global_accuracy']
        
        # 归一化权重
        total_weight = sum(client_weights.values())
        if total_weight > 0:
            for client_id in client_weights:
                client_weights[client_id] /= total_weight
        
        # 初始化聚合参数
        aggregated_params = {}
        
        # 聚合每个参数
        for name, param in server_params.items():
            # 跳过特定层或参数
            if 'feature_adapters' in name:
                # 特征适配层保留原始值
                aggregated_params[name] = param.clone()
            else:
                # 其他层进行加权平均
                aggregated_params[name] = param.clone() * 0.2  # 保留20%原始权重
        
        # 将聚合后的参数应用到模型
        if hasattr(unified_server_model, 'load_state_dict'):
            unified_server_model.load_state_dict(aggregated_params, strict=False)
        
        return server_params

class UnifiedParallelTrainer:
    """统一服务器架构下的并行训练器，支持聚类级别的并行训练"""
    
    def __init__(self, client_manager, unified_server_model, global_classifier, device="cuda"):
        """
        初始化并行训练器
        
        Args:
            client_manager: 客户端管理器
            unified_server_model: 统一服务器模型
            global_classifier: 全局分类器
            device: 默认设备
        """
        self.client_manager = client_manager
        self.unified_server_model = unified_server_model
        self.global_classifier = global_classifier
        self.default_device = device
        
        # 初始化日志
        self.logger = logging.getLogger("UnifiedParallelTrainer")
        self.logger.setLevel(logging.INFO)
        
        # 用于存储聚类和设备映射
        self.cluster_map = {}
        self.device_map = {}
        self.max_workers = None
        
        # 客户端模型字典
        self.client_models = {}
    
    def register_client_model(self, client_id, model):
        """注册客户端模型"""
        self.client_models[client_id] = model
    
    def register_client_models(self, client_models_dict):
        """批量注册客户端模型"""
        for client_id, model in client_models_dict.items():
            self.register_client_model(client_id, model)
    
    def setup_training(self, cluster_map, max_workers=None, device_map=None):
        """
        设置训练环境
        
        Args:
            cluster_map: 聚类映射，键为聚类ID，值为客户端ID列表
            max_workers: 最大并行工作线程数
            device_map: 设备映射，键为聚类ID，值为设备名
        """
        self.cluster_map = cluster_map
        self.max_workers = max_workers
        
        # 如果没有设备映射，创建一个简单的映射
        if device_map is None:
            device_map = {}
            # 检查可用的GPU数量
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i, cluster_id in enumerate(cluster_map.keys()):
                    # 循环分配GPU
                    if gpu_count > 0:
                        device_map[cluster_id] = f"cuda:{i % gpu_count}"
                    else:
                        device_map[cluster_id] = "cpu"
            else:
                # 如果没有GPU，所有聚类都用CPU
                for cluster_id in cluster_map.keys():
                    device_map[cluster_id] = "cpu"
        
        self.device_map = device_map
        self.logger.info(f"训练设置完成，聚类数量: {len(cluster_map)}")
        for cluster_id, clients in cluster_map.items():
            self.logger.info(f"聚类 {cluster_id}: {len(clients)} 个客户端，设备: {device_map.get(cluster_id, 'default')}")
    
    def _train_cluster(self, cluster_id, client_ids, train_fn, eval_fn, round_idx, feature_monitor, results_queue):
        """
        训练单个聚类的工作函数
        
        Args:
            cluster_id: 聚类ID
            client_ids: 客户端ID列表
            train_fn: 训练函数
            eval_fn: 评估函数
            round_idx: 当前轮次
            feature_monitor: 特征监控器
            results_queue: 结果队列
        """
        try:
            # 获取当前聚类的设备
            device = self.device_map.get(cluster_id, self.default_device)
            self.logger.info(f"聚类 {cluster_id} 开始训练，设备: {device}")
            
            # 为当前设备创建模型副本
            server_model_copy = copy.deepcopy(self.unified_server_model).to(device)
            classifier_copy = copy.deepcopy(self.global_classifier).to(device)
            
            # 保存训练结果
            cluster_results = {}
            cluster_eval_results = {}
            
            # 训练每个客户端
            for client_id in client_ids:
                # 获取客户端模型
                if client_id not in self.client_models:
                    self.logger.warning(f"客户端 {client_id} 没有对应的模型，跳过")
                    continue
                
                client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                
                # 执行训练
                train_result = train_fn(
                    client_id=client_id,
                    client_model=client_model,
                    unified_server_model=server_model_copy,
                    global_classifier=classifier_copy,
                    device=device,
                    client_manager=self.client_manager,
                    round_idx=round_idx,
                    feature_monitor=feature_monitor
                )
                
                # 更新特征监控器
                if feature_monitor:
                    feature_monitor.update_from_training_result(client_id, train_result, round_idx)
                
                # 保存训练结果
                cluster_results[client_id] = train_result
                
                # 更新客户端模型
                if isinstance(train_result, dict) and 'client_model_state' in train_result:
                    client_model.load_state_dict(train_result['client_model_state'])
                    # 保存回主字典，确保模型在正确设备上
                    self.client_models[client_id] = client_model.cpu()
                
                # 如果提供了评估函数，执行评估
                if eval_fn:
                    eval_result = eval_fn(
                        client_id=client_id,
                        client_model=client_model,
                        unified_server_model=server_model_copy,
                        global_classifier=classifier_copy,
                        device=device,
                        client_manager=self.client_manager,
                        feature_monitor=feature_monitor
                    )
                    
                    # 更新特征监控器
                    if feature_monitor:
                        feature_monitor.update_from_evaluation_result(client_id, eval_result, round_idx)
                    
                    # 保存评估结果
                    cluster_eval_results[client_id] = eval_result
            
            # 将聚类结果添加到队列
            results_queue.put({
                'cluster_id': cluster_id,
                'device': device,
                'server_model': server_model_copy.state_dict(),
                'classifier': classifier_copy.state_dict(),
                'results': cluster_results,
                'eval_results': cluster_eval_results
            })
            
            self.logger.info(f"聚类 {cluster_id} 训练完成")
            
        except Exception as e:
            import traceback
            error_msg = f"聚类 {cluster_id} 训练失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            # 报告错误
            results_queue.put({
                'cluster_id': cluster_id,
                'error': error_msg
            })
    
    def execute_parallel_training(self, train_fn, eval_fn=None, round_idx=0, feature_monitor=None):
        """
        执行并行训练
        
        Args:
            train_fn: 训练函数
            eval_fn: 评估函数（可选）
            round_idx: 当前轮次
            feature_monitor: 特征监控器（可选）
            
        Returns:
            all_results: 所有训练结果
            all_eval_results: 所有评估结果
            server_models: 各聚类的服务器模型
            classifiers: 各聚类的分类器
            training_time: 训练时间
        """
        start_time = time.time()
        
        # 没有聚类映射时返回空结果
        if not self.cluster_map:
            self.logger.warning("没有设置聚类映射，无法执行训练")
            return {}, {}, {}, {}, 0
        
        # 创建结果队列
        results_queue = queue.Queue()
        
        # 创建线程
        threads = []
        for cluster_id, client_ids in self.cluster_map.items():
            thread = threading.Thread(
                target=self._train_cluster,
                args=(cluster_id, client_ids, train_fn, eval_fn, round_idx, feature_monitor, results_queue)
            )
            threads.append(thread)
        
        # 控制并行度
        active_threads = []
        max_workers = self.max_workers or len(self.cluster_map)
        
        for thread in threads:
            # 等待有可用线程槽
            while len(active_threads) >= max_workers:
                # 检查已完成的线程
                active_threads = [t for t in active_threads if t.is_alive()]
                if len(active_threads) >= max_workers:
                    time.sleep(0.1)
            
            # 启动新线程
            thread.start()
            active_threads.append(thread)
            self.logger.debug(f"启动新线程，当前活动线程数: {len(active_threads)}")
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        all_results = {}
        all_eval_results = {}
        server_models = {}
        classifiers = {}
        
        while not results_queue.empty():
            result = results_queue.get()
            
            # 检查是否有错误
            if 'error' in result:
                self.logger.error(f"聚类 {result['cluster_id']} 返回错误: {result['error']}")
                continue
            
            # 保存结果
            cluster_id = result['cluster_id']
            server_models[cluster_id] = result['server_model']
            classifiers[cluster_id] = result['classifier']
            
            # 合并训练结果
            for client_id, client_result in result['results'].items():
                all_results[client_id] = client_result
            
            # 合并评估结果
            for client_id, eval_result in result['eval_results'].items():
                all_eval_results[client_id] = eval_result
        
        training_time = time.time() - start_time
        self.logger.info(f"并行训练完成，耗时: {training_time:.2f}秒")
        
        return all_results, all_eval_results, server_models, classifiers, training_time
    
    def update_global_models(self, server_models, classifiers, aggregator=None):
        """
        更新全局模型
        
        Args:
            server_models: 各聚类的服务器模型字典
            classifiers: 各聚类的分类器字典
            aggregator: 聚合器（可选）
            
        Returns:
            是否成功更新
        """
        if not server_models or not classifiers:
            self.logger.warning("没有模型可更新")
            return False
        
        try:
            # 如果提供了聚合器，使用聚合器聚合模型
            if aggregator:
                self.logger.info("使用聚合器聚合模型")
                
                # 聚合服务器模型
                aggregated_server = aggregator.aggregate_server_models(server_models)
                self.unified_server_model.load_state_dict(aggregated_server)
                
                # 聚合分类器
                aggregated_classifier = aggregator.aggregate_classifiers(classifiers)
                self.global_classifier.load_state_dict(aggregated_classifier)
            else:
                # 简单平均聚合
                self.logger.info("使用简单平均聚合模型")
                
                # 服务器模型平均聚合
                self._average_aggregate_models(self.unified_server_model, server_models)
                
                # 分类器平均聚合
                self._average_aggregate_models(self.global_classifier, classifiers)
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新全局模型失败: {str(e)}")
            return False
    
    def _average_aggregate_models(self, target_model, models_dict):
        """
        简单平均聚合模型
        
        Args:
            target_model: 目标模型
            models_dict: 模型字典，键为ID，值为状态字典
        """
        if not models_dict:
            return
        
        # 获取目标模型状态字典
        target_state = target_model.state_dict()
        
        # 收集所有模型的状态字典
        all_states = []
        for model_state in models_dict.values():
            if isinstance(model_state, dict):
                all_states.append(model_state)
        
        if not all_states:
            return
        
        # 平均聚合
        for key in target_state:
            if key in all_states[0]:
                # 初始化为零张量
                avg_param = torch.zeros_like(target_state[key])
                
                # 累加所有模型的参数
                valid_count = 0
                for state_dict in all_states:
                    if key in state_dict:
                        avg_param += state_dict[key].to(avg_param.device)
                        valid_count += 1
                
                # 计算平均值
                if valid_count > 0:
                    avg_param = avg_param / valid_count
                    target_state[key] = avg_param
        
        # 更新目标模型
        target_model.load_state_dict(target_state)
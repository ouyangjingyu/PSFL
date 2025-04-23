import torch
import torch.nn as nn
import numpy as np
import time
import copy
from collections import defaultdict

class TierHFLServerGroup:
    """TierHFL服务器组 - 修改版，移除经验回放缓存"""
    def __init__(self, group_id, server_model, global_classifier, device='cuda'):
        self.group_id = group_id
        self.server_model = server_model
        self.global_classifier = global_classifier
        self.device = device
        self.client_ids = []  # 该组管理的客户端ID
        
        # 优化器
        self.server_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=0.0005, weight_decay=1e-4)
        self.classifier_optimizer = torch.optim.Adam(
            self.global_classifier.parameters(), lr=0.001, weight_decay=1e-3)
        
        # 学习率调度器
        self.server_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.server_optimizer, mode='max', factor=0.7, patience=5, min_lr=0.00001)
        self.classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.classifier_optimizer, mode='max', factor=0.7, patience=3, min_lr=0.0001)
        
        # 训练统计
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
        }
    
    def add_client(self, client_id):
        """添加客户端到组"""
        if client_id not in self.client_ids:
            self.client_ids.append(client_id)
    
    def remove_client(self, client_id):
        """从组中移除客户端"""
        if client_id in self.client_ids:
            self.client_ids.remove(client_id)
    
    def update_clients(self, client_ids):
        """更新客户端列表"""
        self.client_ids = client_ids.copy()
    
    def train_batch(self, features, targets, contrastive_features=None, contrastive_labels=None):
        """训练一个批次
        
        Args:
            features: 特征 [B, C, H, W]
            targets: 目标 [B]
            contrastive_features: 对比学习特征 (可选)
            contrastive_labels: 对比学习标签 (可选)
            
        Returns:
            损失和准确率
        """
        # 确保在正确设备上
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        # 清除梯度
        self.server_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        
        # 前向传播
        server_features = self.server_model(features)
        logits = self.global_classifier(server_features)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        
        # 添加对比损失(如果提供)
        if contrastive_features is not None and contrastive_labels is not None:
            contrast_loss_fn = ContrastiveLearningLoss()
            contrast_loss = contrast_loss_fn(
                server_features, targets, contrastive_features, contrastive_labels)
            loss += 0.1 * contrast_loss  # 添加对比损失，权重为0.1
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        self.server_optimizer.step()
        self.classifier_optimizer.step()
        
        # 计算准确率
        _, predicted = logits.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = 100.0 * correct / targets.size(0)
        
        return loss.item(), accuracy
    
    def evaluate_batch(self, features, targets):
        """评估一个批次
        
        Args:
            features: 特征 [B, C, H, W]
            targets: 目标 [B]
            
        Returns:
            损失和准确率
        """
        # 确保在正确设备上
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        # 设置评估模式
        self.server_model.eval()
        self.global_classifier.eval()
        
        with torch.no_grad():
            # 前向传播
            server_features = self.server_model(features)
            logits = self.global_classifier(server_features)
            
            # 计算损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, targets)
            
            # 计算准确率
            _, predicted = logits.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100.0 * correct / targets.size(0)
        
        # 恢复训练模式
        self.server_model.train()
        self.global_classifier.train()
        
        return loss.item(), accuracy
    
    def process_client_features(self, features, return_gradients=False):
        """处理客户端上传的特征
        
        Args:
            features: 特征 [B, C, H, W]
            return_gradients: 是否返回梯度
            
        Returns:
            处理后的特征，及梯度(如果需要)
        """
        # 确保在正确设备上
        features = features.to(self.device)
        
        # 如果需要梯度，设置requires_grad
        if return_gradients:
            features.requires_grad_(True)
        
        # 前向传播
        server_features = self.server_model(features)
        
        # 如果需要梯度，返回特征和梯度计算函数
        if return_gradients:
            def get_gradients(targets):
                # 清除梯度
                if features.grad is not None:
                    features.grad.zero_()
                
                # 计算分类结果
                logits = self.global_classifier(server_features)
                
                # 计算损失
                targets = targets.to(self.device)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, targets)
                
                # 反向传播
                loss.backward()
                
                # 返回梯度
                return features.grad.clone()
            
            return server_features, get_gradients
        
        return server_features
    
    def update_learning_rate(self, accuracy):
        """根据性能更新学习率"""
        self.server_scheduler.step(accuracy)
        self.classifier_scheduler.step(accuracy)

class TierHFLCentralServer:
    """TierHFL中央服务器，管理所有服务器组"""
    def __init__(self, device='cuda'):
        self.device = device
        self.server_groups = {}  # 服务器组
        self.client_to_group = {}  # 客户端到组的映射
        
        # 全局统计
        self.stats = {
            'global_accuracy': [],
            'group_accuracy': defaultdict(list),
            'cross_client_accuracy': []
        }
        
        # 评估历史
        self.evaluation_history = []
    
    def create_server_group(self, group_id, server_model, global_classifier):
        """创建服务器组"""
        server_group = TierHFLServerGroup(
            group_id, server_model, global_classifier, self.device)
        self.server_groups[group_id] = server_group
        return server_group
    
    def get_server_group(self, group_id):
        """获取服务器组"""
        return self.server_groups.get(group_id)
    
    def assign_client_to_group(self, client_id, group_id):
        """将客户端分配到服务器组"""
        # 如果客户端已在某个组中，先移除
        if client_id in self.client_to_group:
            old_group_id = self.client_to_group[client_id]
            old_group = self.get_server_group(old_group_id)
            if old_group:
                old_group.remove_client(client_id)
        
        # 分配到新组
        self.client_to_group[client_id] = group_id
        group = self.get_server_group(group_id)
        if group:
            group.add_client(client_id)
        else:
            # 如果指定的组不存在，记录错误并尝试分配到第一个可用组
            print(f"错误：组 {group_id} 不存在，尝试将客户端 {client_id} 分配到其他组")
            if self.server_groups:
                first_group_id = next(iter(self.server_groups.keys()))
                self.client_to_group[client_id] = first_group_id
                self.server_groups[first_group_id].add_client(client_id)
    
    def get_client_group(self, client_id):
        """获取客户端所在的组"""
        group_id = self.client_to_group.get(client_id)
        if group_id is not None and group_id in self.server_groups:
            return self.server_groups[group_id]
        
        # 如果找不到分配或组不存在，返回第一个可用的组
        if self.server_groups:
            print(f"客户端 {client_id} 无有效组分配，使用默认组")
            default_group_id = next(iter(self.server_groups.keys()))
            # 自动分配到默认组
            self.assign_client_to_group(client_id, default_group_id)
            return self.server_groups[default_group_id]
        
        return None
    
    def process_client_features(self, client_id, features, return_gradients=False):
        """处理客户端特征"""
        # 获取客户端所在服务器组
        group = self.get_client_group(client_id)
        if group:
            return group.process_client_features(features, return_gradients)
        
        # 如果客户端未分配组，使用第一个组
        if self.server_groups:
            first_group = next(iter(self.server_groups.values()))
            return first_group.process_client_features(features, return_gradients)
        
        # 没有组可用，返回空结果
        return None if not return_gradients else (None, None)
    
    def update_evaluation_history(self, round_idx, results):
        """更新评估历史"""
        self.evaluation_history.append({
            'round': round_idx,
            'results': results,
            'timestamp': time.time()
        })
        
        # 更新统计信息
        if 'global_accuracy' in results:
            self.stats['global_accuracy'].append(results['global_accuracy'])
        
        for group_id, acc in results.get('group_accuracy', {}).items():
            self.stats['group_accuracy'][group_id].append(acc)
        
        if 'cross_client_accuracy' in results:
            self.stats['cross_client_accuracy'].append(results['cross_client_accuracy'])
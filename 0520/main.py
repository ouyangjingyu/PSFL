import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import random
import argparse
import logging
import wandb
import copy
import warnings
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms
import math

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入自定义模块
from model.resnet import EnhancedServerModel, TierAwareClientModel, ImprovedGlobalClassifier
from utils.tierhfl_aggregator import StabilizedAggregator
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_loss import EnhancedUnifiedLoss

from analyze.tierhfl_analyze import validate_server_effectiveness

# 导入验证器
from analyze.tierhfl_analyze import GlobalClassifierVerifier

# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
# 导入新的Fashion-MNIST数据加载器
from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist
# 导入监控器
from analyze.diagnostic_monitor import EnhancedTierHFLDiagnosticMonitor
# 串行训练器实现
class SimpleSerialTrainer:
    """简化的串行训练器，用于实现TierHFL的训练流程 - 修复版"""
    def __init__(self, client_manager, server_model, global_classifier, 
                loss_fn=None, device="cuda"):
        self.client_manager = client_manager
        self.server_model = server_model
        self.global_classifier = global_classifier
        self.device = device
        
        # 处理损失函数
        if loss_fn is None:
            self.loss_fn = EnhancedUnifiedLoss()
        else:
            self.loss_fn = loss_fn
        
        # 客户端模型字典
        self.client_models = {}
        
        # 聚类映射和服务器模型状态
        self.cluster_map = {}
        self.cluster_server_models = {}
        # 添加全局分类器副本字典
        self.cluster_global_classifiers = {}
        
    def register_client_models(self, client_models_dict):
        """注册客户端模型"""
        self.client_models.update(client_models_dict)
    
    def setup_training(self, cluster_map):
        """设置训练环境"""
        self.cluster_map = cluster_map
        # 为每个聚类初始化服务器模型和全局分类器
        self.cluster_server_models = {}
        self.cluster_global_classifiers = {}
        for cluster_id in cluster_map.keys():
            self.cluster_server_models[cluster_id] = copy.deepcopy(self.server_model.state_dict())
            self.cluster_global_classifiers[cluster_id] = copy.deepcopy(self.global_classifier.state_dict())
    
    def execute_round(self, round_idx, total_rounds, diagnostic_monitor=None):
        """执行一轮训练 - 集成增强版监控和综合分析"""
        start_time = time.time()
        
        # 监控学习率
        if diagnostic_monitor is not None:
            client_lrs = {}
            for client_id in range(len(self.client_models)):
                client = self.client_manager.get_client(client_id)
                if client:
                    client_lrs[client_id] = client.lr
            
            lr_analysis = diagnostic_monitor.monitor_learning_rates(client_lrs, round_idx)

        
        # 结果容器
        train_results = {}
        eval_results = {}
        shared_states = {}
        
        # 确定当前训练阶段
        if round_idx < 10:
            training_phase = "initial"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 初始特征学习阶段")
        elif round_idx < 80:
            training_phase = "alternating"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 交替训练阶段")
        else:
            training_phase = "fine_tuning"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 精细调整阶段")
        
        # 依次处理每个聚类
        for cluster_id, client_ids in self.cluster_map.items():
            logging.info(f"处理聚类 {cluster_id}, 包含 {len(client_ids)} 个客户端")
            
            # 创建聚类特定的模型
            cluster_server = copy.deepcopy(self.server_model).to(self.device)
            cluster_server.load_state_dict(self.cluster_server_models[cluster_id])
            
            cluster_classifier = copy.deepcopy(self.global_classifier).to(self.device)
            cluster_classifier.load_state_dict(self.cluster_global_classifiers[cluster_id])
            
            # === 监控聚类模型稳定性 ===
            if diagnostic_monitor is not None:
                diagnostic_monitor.monitor_model_stability_fixed(
                    cluster_server.state_dict(), round_idx, f"server_cluster_{cluster_id}"
                )
                diagnostic_monitor.monitor_model_stability_fixed(
                    cluster_classifier.state_dict(), round_idx, f"classifier_cluster_{cluster_id}"
                )
            
            # 处理聚类中的每个客户端
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if not client or client_id not in self.client_models:
                    continue
                
                logging.info(f"训练客户端 {client_id} (Tier: {client.tier})")
                
                client_model = self.client_models[client_id].to(self.device)
                client.model = client_model
                
                # === 监控客户端模型稳定性 ===
                if diagnostic_monitor is not None:
                    diagnostic_monitor.monitor_model_stability_fixed(
                        client_model.state_dict(), round_idx, f"client_{client_id}"
                    )
                
                # 根据训练阶段执行训练
                if training_phase == "initial":
                    train_result = self._train_initial_phase(
                        client, client_model, cluster_server, cluster_classifier, 
                        round_idx, total_rounds, diagnostic_monitor)
                    train_result['local_loss'] = 0.0
                    train_result['local_accuracy'] = 0.0
                    
                elif training_phase == "alternating":
                    personal_result = self._train_personal_path(
                        client, client_model, client.lr * 0.05, round_idx, total_rounds, diagnostic_monitor)
                    
                    train_result = self._train_alternating_phase(
                        client, client_model, cluster_server, cluster_classifier,
                        personal_result, round_idx, total_rounds, diagnostic_monitor)
                    
                else:
                    train_result = self._train_fine_tuning_phase(
                        client, client_model, cluster_server, cluster_classifier,
                        round_idx, total_rounds, diagnostic_monitor)
                
                # 保存结果
                train_results[client_id] = train_result
                
                # 评估客户端
                eval_result = self._evaluate_client(
                    client, client_model, cluster_server, cluster_classifier)
                eval_results[client_id] = eval_result
                
                # 保存共享层状态
                shared_state = {}
                for name, param in client_model.named_parameters():
                    if 'shared_base' in name:
                        shared_state[name] = param.data.clone().cpu()
                shared_states[client_id] = shared_state
                
                # 更新客户端模型
                self.client_models[client_id] = client_model.cpu()
                
                torch.cuda.empty_cache()
            
            # 保存聚类模型
            self.cluster_server_models[cluster_id] = cluster_server.cpu().state_dict()
            self.cluster_global_classifiers[cluster_id] = cluster_classifier.cpu().state_dict()
        
        # 计算总训练时间
        training_time = time.time() - start_time
        
        # === 生成综合诊断报告 ===
        if diagnostic_monitor is not None:
            comprehensive_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            
            # 记录关键发现
            if comprehensive_report['overall_health'] == 'critical':
                logging.error(f"轮次{round_idx+1}: 检测到严重问题!")
                for issue in comprehensive_report['critical_issues']:
                    logging.error(f"  - {issue}")
            
            if comprehensive_report['recommendations']:
                logging.info(f"轮次{round_idx+1} 优化建议:")
                for rec in comprehensive_report['recommendations']:
                    logging.info(f"  - {rec}")
        
        return train_results, eval_results, shared_states, training_time
        

    def _train_initial_phase(self, client, client_model, server_model, classifier, 
                        round_idx, total_rounds, diagnostic_monitor=None):
        """阶段1: 初始特征学习阶段 - 集成增强版监控"""
        start_time = time.time()
        
        # 冻结个性化路径
        for name, param in client_model.named_parameters():
            if 'personalized_path' in name or 'local_classifier' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        classifier.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=client.lr * 0.5
        )
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        
        # 统计信息
        stats = {
            'global_loss': 0.0,
            'total_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 训练循环
        for batch_idx, (data, target) in enumerate(client.train_data):
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除梯度
            shared_optimizer.zero_grad()
            server_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            # 前向传播
            local_logits, shared_features, personal_features = client_model(data)
            server_features = server_model(shared_features)
            global_logits = classifier(server_features)
            
            # 计算损失
            global_loss = F.cross_entropy(global_logits, target)
            # 本地损失用于监控（不参与反向传播）
            with torch.no_grad():
                local_loss = F.cross_entropy(local_logits, target)
            total_loss = global_loss
            
            # === 增强版诊断监控 ===
            if diagnostic_monitor is not None and batch_idx == 0:
                try:
                    # 1. 修复版梯度冲突分析
                    gradient_analysis = diagnostic_monitor.analyze_gradient_conflict_fixed(
                        client_model, global_loss, local_loss, round_idx, client.client_id
                    )
                    
                    # 2. 分类器崩溃检测
                    collapse_analysis = diagnostic_monitor.detect_classifier_collapse(
                        global_logits, target, round_idx, client.client_id, "global"
                    )
                    
                    # 3. 共享层质量分析
                    quality_analysis = diagnostic_monitor.analyze_shared_layer_quality(
                        shared_features, round_idx, client.client_id
                    )
                    
                    # 4. 分类器权重分析
                    if round_idx % 5 == 0:  # 每5轮分析一次权重
                        weight_analysis = diagnostic_monitor.analyze_classifier_weights(
                            classifier, round_idx
                        )
                    
                except Exception as e:
                    logging.error(f"初始阶段增强监控出错: {str(e)}")
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                max_norm=1.0
            )
            
            # 更新参数
            shared_optimizer.step()
            server_optimizer.step()
            classifier_optimizer.step()
            
            # 更新统计信息
            stats['global_loss'] += global_loss.item()
            stats['total_loss'] += total_loss.item()
            stats['batch_count'] += 1
            
            _, pred = global_logits.max(1)
            stats['correct'] += pred.eq(target).sum().item()
            stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'total_loss']:
            if stats['batch_count'] > 0:
                stats[key] /= stats['batch_count']
        
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        # 解冻所有层
        for name, param in client_model.named_parameters():
            param.requires_grad = True
        
        return {
            'global_loss': stats['global_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time
        }

    def _train_alternating_phase(self, client, client_model, server_model, classifier, 
                            personal_result, round_idx, total_rounds, diagnostic_monitor=None):
        """阶段2: 交替训练阶段 - 先训练全局路径，再微调个性化路径"""
        progress = round_idx / total_rounds
        
        # --- 第一部分: 训练全局路径 ---
        # 冻结个性化路径
        for name, param in client_model.named_parameters():
            if 'personalized_path' in name or 'local_classifier' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 全局路径训练
        shared_lr = client.lr * 0.1 * (1 - progress * 0.5)  # 动态调整共享层学习率
        global_result = self._train_global_path(client, client_model, server_model, classifier, 
                                            shared_lr, round_idx, total_rounds)
        
        # --- 第二部分: 训练个性化路径 ---
        # 启用所有层
        for name, param in client_model.named_parameters():
            param.requires_grad = True
        
        # 个性化路径训练 - 共享层使用极小学习率
        shared_lr = client.lr * 0.05 * (1 - progress * 0.7)  # 更小的学习率
        personal_result = self._train_personal_path(client, client_model, shared_lr, 
                                                round_idx, total_rounds)
        
        # 组合结果
        result = {
            'global_loss': global_result['global_loss'],
            'total_loss': (global_result['total_loss'] + personal_result['local_loss']) / 2,
            'global_accuracy': global_result['global_accuracy'],
            'local_accuracy': personal_result['local_accuracy'],
            'time_cost': global_result['time_cost'] + personal_result['time_cost']
        }
        
        return result

    def _train_fine_tuning_phase(self, client, client_model, server_model, classifier, 
                            round_idx, total_rounds, diagnostic_monitor=None):
        """阶段3: 精细调整阶段 - 分别微调两条路径，共享层使用极小学习率"""
        # 全局路径训练 - 极小共享层学习率
        shared_lr = client.lr * 0.02
        global_result = self._train_global_path(client, client_model, server_model, classifier, 
                                            shared_lr, round_idx, total_rounds)
        
        # 个性化路径训练 - 更小共享层学习率
        shared_lr = client.lr * 0.01
        personal_result = self._train_personal_path(client, client_model, shared_lr, 
                                                round_idx, total_rounds)
        
        # 每3轮执行一次特征对齐
        if round_idx % 3 == 0:
            self._align_features(client, client_model, server_model)
        
        # 组合结果
        result = {
            'global_loss': global_result['global_loss'],
            'total_loss': (global_result['total_loss'] + personal_result['local_loss']) / 2,
            'global_accuracy': global_result['global_accuracy'],
            'local_accuracy': personal_result['local_accuracy'],
            'time_cost': global_result['time_cost'] + personal_result['time_cost']
        }
        
        return result

    def _train_global_path(self, client, client_model, server_model, classifier, 
                    shared_lr, round_idx, total_rounds, diagnostic_monitor=None):
        """训练全局路径 - 共享层和服务器模型，添加诊断监控"""
        start_time = time.time()
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        classifier.train()
        
        # 创建针对客户端共享层的优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=shared_lr
        )
        
        # 创建针对服务器模型的优化器
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        
        # 创建针对全局分类器的优化器
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        
        # 统计信息
        stats = {
            'global_loss': 0.0,
            'total_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 训练循环
        for batch_idx, (data, target) in enumerate(client.train_data):
            # 移至设备
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除所有梯度
            shared_optimizer.zero_grad()
            server_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            # 前向传播
            local_logits, shared_features, personal_features = client_model(data)
            server_features = server_model(shared_features)
            global_logits = classifier(server_features)
            
            # 计算损失
            global_loss = F.cross_entropy(global_logits, target)
            local_loss = F.cross_entropy(local_logits, target)  # 用于监控
            total_loss = global_loss
            
            # === 添加诊断监控 ===
            if diagnostic_monitor is not None and batch_idx == 0:  # 只在第一个batch监控，避免过多开销
                try:
                    # 1. 梯度冲突分析
                    gradient_analysis = diagnostic_monitor.analyze_gradient_conflict(
                        client_model, global_loss, local_loss, round_idx, client.client_id
                    )
                    
                    # 2. 特征质量监控
                    feature_analysis = diagnostic_monitor.monitor_feature_quality(
                        personal_features, server_features, shared_features, 
                        round_idx, client.client_id
                    )
                    
                    # 3. 分类置信度分析
                    confidence_analysis = diagnostic_monitor.analyze_classification_confidence(
                        global_logits, target, round_idx, client.client_id, "global"
                    )
                    
                    # 4. 损失组件分析
                    loss_analysis = diagnostic_monitor.analyze_loss_components(
                        total_loss, global_loss, local_loss, torch.tensor(0.0),  # feature_loss为0
                        0.0, 0.0, round_idx, client.client_id  # alpha, beta为0
                    )
                    
                except Exception as e:
                    logging.error(f"诊断监控出错: {str(e)}")
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                max_norm=0.5
            )
            
            # 更新参数
            shared_optimizer.step()
            server_optimizer.step()
            classifier_optimizer.step()
            
            # 更新统计信息
            stats['global_loss'] += global_loss.item()
            stats['total_loss'] += total_loss.item()
            stats['batch_count'] += 1
            
            # 计算准确率
            _, pred = global_logits.max(1)
            stats['correct'] += pred.eq(target).sum().item()
            stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'total_loss']:
            if stats['batch_count'] > 0:
                stats[key] /= stats['batch_count']
        
        # 计算全局准确率
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        return {
            'global_loss': stats['global_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time
        }

    def _train_personal_path(self, client, client_model, shared_lr, round_idx, total_rounds, diagnostic_monitor=None):
        """训练个性化路径 - 集成增强版监控"""
        start_time = time.time()
        
        client_model.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n],
            lr=shared_lr
        )
        personal_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() 
            if 'shared_base' not in n and p.requires_grad],
            lr=client.lr
        )
        
        # 统计信息
        stats = {
            'local_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 训练循环
        for epoch in range(client.local_epochs):
            for batch_idx, (data, target) in enumerate(client.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                shared_optimizer.zero_grad()
                personal_optimizer.zero_grad()
                
                # 前向传播
                local_logits, shared_features, personal_features = client_model(data)
                local_loss = F.cross_entropy(local_logits, target)
                
                # === 增强版诊断监控 ===
                if diagnostic_monitor is not None and batch_idx == 0 and epoch == 0:
                    try:
                        # 1. 分类器崩溃检测
                        collapse_analysis = diagnostic_monitor.detect_classifier_collapse(
                            local_logits, target, round_idx, client.client_id, "local"
                        )
                        
                        # 2. 共享层质量分析
                        quality_analysis = diagnostic_monitor.analyze_shared_layer_quality(
                            shared_features, round_idx, client.client_id
                        )
                        
                    except Exception as e:
                        logging.error(f"个性化路径增强监控出错: {str(e)}")
                
                # 反向传播
                local_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                    max_norm=0.3
                )
                
                # 更新参数
                shared_optimizer.step()
                personal_optimizer.step()
                
                # 更新统计信息
                stats['local_loss'] += local_loss.item()
                stats['batch_count'] += 1
                
                _, pred = local_logits.max(1)
                stats['correct'] += pred.eq(target).sum().item()
                stats['total'] += target.size(0)
        
        # 计算结果
        avg_local_loss = stats['local_loss'] / max(1, stats['batch_count'])
        local_accuracy = 100.0 * stats['correct'] / max(1, stats['total'])
        
        return {
            'local_loss': avg_local_loss,
            'local_accuracy': local_accuracy,
            'time_cost': time.time() - start_time
        }

    def _align_features(self, client, client_model, server_model):
        """特征对齐 - 仅更新共享层，使其特征对两条路径都有用"""
        # 设置训练模式
        client_model.train()
        server_model.train()
        
        # 冻结除共享层外的所有层
        for name, param in client_model.named_parameters():
            if 'shared_base' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        for param in server_model.parameters():
            param.requires_grad = False
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n],
            lr=client.lr * 0.03
        )
        
        # 获取一个批次数据
        try:
            data, target = next(iter(client.train_data))
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            _, shared_features, personal_features = client_model(data)
            server_features = server_model(shared_features)
            
            # 处理特征
            if personal_features.dim() > 2:
                personal_features = F.adaptive_avg_pool2d(personal_features, (1, 1)).flatten(1)
            if server_features.dim() > 2:
                server_features = server_features.flatten(1)
            
            # 特征标准化
            personal_norm = F.normalize(personal_features, dim=1)
            server_norm = F.normalize(server_features, dim=1)
            
            # 计算特征对齐损失
            alignment_loss = 1.0 - torch.mean(torch.sum(personal_norm * server_norm, dim=1))
            
            # 反向传播
            optimizer.zero_grad()
            alignment_loss.backward()
            optimizer.step()
            
            # 解冻所有层
            for name, param in client_model.named_parameters():
                param.requires_grad = True
            
            for param in server_model.parameters():
                param.requires_grad = True
                
        except (StopIteration, RuntimeError) as e:
            print(f"特征对齐过程发生错误: {str(e)}")
            
            # 确保解冻所有层
            for name, param in client_model.named_parameters():
                param.requires_grad = True
            
            for param in server_model.parameters():
                param.requires_grad = True
    
    def _evaluate_client(self, client, client_model, server_model, classifier):
        """评估客户端模型"""
        # 设置评估模式
        client_model.eval()
        server_model.eval()
        classifier.eval()
        
        # 统计信息
        local_correct = 0
        global_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in client.test_data:
                # 移至设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                local_logits, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                global_logits = classifier(server_features)
                
                # 计算准确率
                _, local_pred = local_logits.max(1)
                _, global_pred = global_logits.max(1)
                
                local_correct += local_pred.eq(target).sum().item()
                global_correct += global_pred.eq(target).sum().item()
                total += target.size(0)
        
        # 计算准确率
        local_accuracy = 100.0 * local_correct / max(1, total)
        global_accuracy = 100.0 * global_correct / max(1, total)
        
        return {
            'local_accuracy': local_accuracy,
            'global_accuracy': global_accuracy,
            'total_samples': total
        }
    
    def aggregate_client_shared_layers(self, shared_states):
        """聚合客户端共享层参数"""
        if not shared_states:
            return {}
        
        # 创建权重字典 - 默认平均权重
        weights = {client_id: 1.0 / len(shared_states) for client_id in shared_states}
        
        # 聚合
        aggregated_model = {}
        
        for key in next(iter(shared_states.values())).keys():
            # 计算加权和
            weighted_sum = None
            total_weight = 0.0
            
            for client_id, state in shared_states.items():
                if key in state:
                    weight = weights.get(client_id, 0.0)
                    total_weight += weight
                    
                    # 累加
                    if weighted_sum is None:
                        weighted_sum = weight * state[key].clone()
                    else:
                        weighted_sum += weight * state[key].clone()
            
            # 计算加权平均
            if weighted_sum is not None and total_weight > 0:
                aggregated_model[key] = weighted_sum / total_weight
        
        return aggregated_model
    
    def aggregate_server_models(self, eval_results=None):
        """聚合所有聚类的服务器模型"""
        if not self.cluster_server_models:
            return None
        
        # 计算聚类权重 - 默认平均权重
        if eval_results:
            # 基于性能的权重
            cluster_weights = {}
            cluster_performance = {}
            
            # 计算每个聚类的平均性能
            for cluster_id, client_ids in self.cluster_map.items():
                cluster_accuracies = []
                for client_id in client_ids:
                    if client_id in eval_results:
                        cluster_accuracies.append(eval_results[client_id].get('global_accuracy', 0))
                
                if cluster_accuracies:
                    cluster_performance[cluster_id] = sum(cluster_accuracies) / len(cluster_accuracies)
                else:
                    cluster_performance[cluster_id] = 0.0
            
            # 将性能转换为权重
            total_performance = sum(cluster_performance.values())
            if total_performance > 0:
                for cluster_id, perf in cluster_performance.items():
                    cluster_weights[cluster_id] = perf / total_performance
            else:
                # 平均权重
                cluster_weights = {c_id: 1.0/len(self.cluster_map) for c_id in self.cluster_map}
        else:
            # 平均权重
            cluster_weights = {c_id: 1.0/len(self.cluster_map) for c_id in self.cluster_map}
        
        # 创建临时模型用于聚合
        temp_model = copy.deepcopy(self.server_model)
        
        # 参数字典
        aggregated_params = {}
        
        # 聚合参数
        for name, param in temp_model.named_parameters():
            weighted_sum = None
            total_weight = 0.0
            
            for cluster_id, weight in cluster_weights.items():
                if cluster_id in self.cluster_server_models:
                    cluster_state = self.cluster_server_models[cluster_id]
                    if name in cluster_state:
                        total_weight += weight
                        if weighted_sum is None:
                            weighted_sum = weight * cluster_state[name].clone()
                        else:
                            weighted_sum += weight * cluster_state[name].clone()
            
            if weighted_sum is not None and total_weight > 0:
                aggregated_params[name] = weighted_sum / total_weight
        
        return aggregated_params

    def aggregate_global_classifiers(self, eval_results=None):
        """聚合所有聚类的全局分类器"""
        if not self.cluster_global_classifiers:
            return None
        
        # 计算聚类权重
        if eval_results:
            # 基于性能的权重
            cluster_weights = {}
            for cluster_id, client_ids in self.cluster_map.items():
                accuracies = [eval_results.get(client_id, {}).get('global_accuracy', 0) 
                            for client_id in client_ids if client_id in eval_results]
                
                # 避免极端权重
                if accuracies:
                    avg_acc = sum(accuracies) / len(accuracies)
                    # 线性映射而非二次方，减小权重差异
                    weight = 0.5 + 0.5 * (avg_acc / 100.0)  # 限制权重范围在[0.5, 1.0]
                else:
                    weight = 0.5
                    
                cluster_weights[cluster_id] = weight
        else:
            # 平均权重
            cluster_weights = {c_id: 1.0/len(self.cluster_map) for c_id in self.cluster_map}
        
        # 创建临时模型用于聚合
        temp_classifier = copy.deepcopy(self.global_classifier)
        
        # 参数字典
        aggregated_params = {}
        
        # 聚合参数
        for name, param in temp_classifier.named_parameters():
            weighted_sum = None
            total_weight = 0.0
            
            for cluster_id, weight in cluster_weights.items():
                if cluster_id in self.cluster_global_classifiers:
                    cluster_state = self.cluster_global_classifiers[cluster_id]
                    if name in cluster_state:
                        total_weight += weight
                        if weighted_sum is None:
                            weighted_sum = weight * cluster_state[name].clone()
                        else:
                            weighted_sum += weight * cluster_state[name].clone()
            
            if weighted_sum is not None and total_weight > 0:
                aggregated_params[name] = weighted_sum / total_weight
        
        return aggregated_params
    
    def update_client_shared_layers(self, aggregated_shared_state):
        """更新所有客户端的共享层参数"""
        if not aggregated_shared_state:
            return False
        
        for client_id, model in self.client_models.items():
            for name, param in model.named_parameters():
                if 'shared_base' in name and name in aggregated_shared_state:
                    param.data.copy_(aggregated_shared_state[name])
        
        return True
    
    def update_server_models(self, aggregated_server_model, aggregated_global_classifier=None):
        """更新所有聚类的服务器模型和全局分类器"""
        updated = False
        
        # 更新服务器模型
        if aggregated_server_model:
            # 更新主服务器模型
            for name, param in self.server_model.named_parameters():
                if name in aggregated_server_model:
                    param.data.copy_(aggregated_server_model[name])
            
            # 更新所有聚类的服务器模型
            for cluster_id in self.cluster_server_models:
                self.cluster_server_models[cluster_id] = copy.deepcopy(self.server_model.state_dict())
            
            updated = True
        
        # 更新全局分类器
        if aggregated_global_classifier:
            # 更新主全局分类器
            for name, param in self.global_classifier.named_parameters():
                if name in aggregated_global_classifier:
                    param.data.copy_(aggregated_global_classifier[name])
            
            # 更新所有聚类的全局分类器
            for cluster_id in self.cluster_global_classifiers:
                self.cluster_global_classifiers[cluster_id] = copy.deepcopy(self.global_classifier.state_dict())
            
            updated = True
        
        return updated

# 设置随机种子函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='TierHFL: 分层异构联邦学习框架 (串行版本)')
    
    # 实验标识
    parser.add_argument('--running_name', default="TierHFL_Serial", type=str, help='实验名称')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.005, type=float, help='初始学习率')
    parser.add_argument('--lr_factor', default=0.9, type=float, help='学习率衰减因子')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet56', help='使用的神经网络 (resnet56 或 resnet110)')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='fashion_mnist', 
                       help='训练数据集 (cifar10, cifar100, fashion_mnist, cinic10)')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='本地工作节点上数据集的划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=5, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=10, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=256, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    parser.add_argument('--n_clusters', default=3, type=int, help='客户端聚类数量')
    
    # TierHFL特有参数
    parser.add_argument('--init_alpha', default=0.6, type=float, help='初始本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.15, type=float, help='初始特征对齐损失权重')
    parser.add_argument('--beta', default=0.3, type=float, help='聚合动量因子')
    
    args = parser.parse_args()
    return args

# 设置日志和wandb
def setup_logging(args):
    # 配置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.running_name}.log")
        ]
    )
    logger = logging.getLogger("TierHFL")
    
    # 初始化wandb
    try:
        wandb.init(
            mode="online",
            project="TierHFL",
            name=args.running_name,
            config=args,
            tags=[f"model_{args.model}", f"dataset_{args.dataset}", 
                  f"clients_{args.client_number}", f"partition_{args.partition_method}"],
            group=f"{args.model}_{args.dataset}"
        )
        
        # 设置自定义面板
        wandb.define_metric("round")
        wandb.define_metric("global/*", step_metric="round")
        wandb.define_metric("local/*", step_metric="round")
        wandb.define_metric("client/*", step_metric="round")
        wandb.define_metric("time/*", step_metric="round")
        wandb.define_metric("params/*", step_metric="round") 
        
    except Exception as e:
        print(f"警告: wandb初始化失败: {e}")
        # 使用离线模式
        try:
            wandb.init(mode="offline", project="TierHFL", name=args.running_name)
        except:
            print("完全禁用wandb")
            
    return logger

# 加载数据集
def load_dataset(args):
    if args.dataset == "cifar10":
        data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_cifar100
    elif args.dataset == "fashion_mnist":
        data_loader = load_partition_data_fashion_mnist
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    else:
        data_loader = load_partition_data_cifar10

    if args.dataset == "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
        
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset

# 为客户端分配设备资源
def allocate_device_resources(client_number):
    resources = {}
    
    # 随机分配tier (1-4)
    tier_weights = [0.2, 0.3, 0.3, 0.2]  # tier 1-4的分布概率
    tiers = random.choices(range(1, 5), weights=tier_weights, k=client_number)
    
    # 为每个客户端分配资源
    for client_id in range(client_number):
        tier = tiers[client_id]
        
        # 根据tier分配计算能力
        if tier == 1:  # 高性能设备
            compute_power = random.uniform(0.8, 1.0)
            network_speed = random.choice([50, 100, 200])
            storage_capacity = random.choice([256, 512, 1024])
        elif tier == 2:  # 中高性能设备
            compute_power = random.uniform(0.6, 0.8)
            network_speed = random.choice([30, 50, 100])
            storage_capacity = random.choice([128, 256, 512])
        elif tier == 3:  # 中低性能设备
            compute_power = random.uniform(0.3, 0.6)
            network_speed = random.choice([20, 30, 50])
            storage_capacity = random.choice([64, 128, 256])
        else:  # tier 4, 低性能设备
            compute_power = random.uniform(0.1, 0.3)
            network_speed = random.choice([5, 10, 20])
            storage_capacity = random.choice([16, 32, 64])
        
        # 存储资源信息
        resources[client_id] = {
            "tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed,
            "storage_capacity": storage_capacity
        }
    
    return resources

def print_cluster_info(cluster_map, client_resources, logger):
    """打印聚类信息详情"""
    logger.info("===== 聚类分布情况 =====")
    for cluster_id, client_ids in cluster_map.items():
        client_tiers = [client_resources[client_id]['tier'] for client_id in client_ids]
        avg_tier = sum(client_tiers) / len(client_tiers) if client_tiers else 0
        tier_distribution = {}
        for tier in client_tiers:
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            
        logger.info(f"聚类 {cluster_id}: {len(client_ids)}个客户端")
        logger.info(f"  - 客户端ID: {client_ids}")
        logger.info(f"  - 平均Tier: {avg_tier:.2f}")
        logger.info(f"  - Tier分布: {tier_distribution}")
        
        # 计算客户端资源异质性
        if client_ids:
            compute_powers = [client_resources[cid]['compute_power'] for cid in client_ids]
            network_speeds = [client_resources[cid]['network_speed'] for cid in client_ids]
            
            logger.info(f"  - 计算能力: 平均={sum(compute_powers)/len(compute_powers):.2f}, "
                       f"最小={min(compute_powers):.2f}, 最大={max(compute_powers):.2f}")
            logger.info(f"  - 网络速度: 平均={sum(network_speeds)/len(network_speeds):.2f}, "
                       f"最小={min(network_speeds)}, 最大={max(network_speeds)}")
    
    # 计算全局聚类指标
    all_clients = sum(len(clients) for clients in cluster_map.values())
    logger.info(f"总计: {len(cluster_map)}个聚类, {all_clients}个客户端")

def load_global_test_set(args):
    """创建全局IID测试集用于评估泛化性能"""
    if args.dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                [0.2675, 0.2565, 0.2761])
        ])
        
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "fashion_mnist":
        # 新增Fashion-MNIST支持
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530])
        ])
        
        testset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "cinic10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                [0.24205776, 0.23828046, 0.25874835])
        ])
        
        # 使用存储在args.data_dir/cinic10/test目录下的CINIC10测试集
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'cinic10', 'test'),
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    else:
        # 默认返回CIFAR10
        return load_global_test_set_cifar10(args)

def evaluate_global_model(client_model, server_model, global_classifier, global_test_loader, device):
    """评估全局模型在全局测试集上的性能 - 修复版"""
    # 确保所有模型都在正确的设备上
    client_model = client_model.to(device)
    server_model = server_model.to(device)
    global_classifier = global_classifier.to(device)
    
    # 设置为评估模式
    client_model.eval()
    server_model.eval()
    global_classifier.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in global_test_loader:
            # 移到设备
            data, target = data.to(device), target.to(device)
            
            try:
                # 完整的前向传播
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                logits = global_classifier(server_features)
                
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 记录预测和目标，用于后续分析
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            except Exception as e:
                print(f"评估中出现错误: {str(e)}")
                continue
    
    accuracy = 100.0 * correct / max(1, total)
    
    # 记录额外的调试信息
    logging.info(f"全局模型评估 - 样本总数: {total}, 正确预测: {correct}")
    if len(all_predictions) >= 100:
        # 打印预测分布
        from collections import Counter
        pred_counter = Counter(all_predictions)
        target_counter = Counter(all_targets)
        logging.info(f"预测分布: {dict(pred_counter)}")
        logging.info(f"目标分布: {dict(target_counter)}")
    
    return accuracy

class ModelFeatureClusterer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clustering_history = []
    
    def cluster_clients(self, client_models, client_ids, eval_dataset=None, device='cuda'):
        """基于模型特征的聚类方法，考虑服务器不能获取原始数据的限制"""
        clusters = {i: [] for i in range(self.num_clusters)}
        
        # 提取模型特征 - 使用共享层参数而非原始数据
        client_features = {}
        feature_dims = []
        
        # 第一步：提取特征
        for client_id in client_ids:
            if client_id in client_models:
                model = client_models[client_id]
                features = []
                
                # 只提取共享层参数作为特征
                for name, param in model.named_parameters():
                    if 'shared_base' in name and 'weight' in name:
                        # 提取统计信息而非原始参数
                        param_data = param.detach().cpu()
                        # 只收集标量特征，避免形状不一致
                        features.extend([
                            param_data.mean().item(),
                            param_data.std().item(),
                            param_data.abs().max().item(),
                            (param_data > 0).float().mean().item()  # 正值比例
                        ])
                
                if features:
                    # 确保features是一维数组
                    features_array = np.array(features, dtype=np.float32)
                    client_features[client_id] = features_array
                    feature_dims.append(len(features_array))
        
        # 检查所有特征向量的维度是否一致
        if feature_dims and len(set(feature_dims)) > 1:
            # 如果维度不一致，找出最常见的维度
            from collections import Counter
            dim_counter = Counter(feature_dims)
            common_dim = dim_counter.most_common(1)[0][0]
            
            print(f"发现不同维度的特征向量: {dict(dim_counter)}，使用最常见维度: {common_dim}")
            
            # 处理维度不一致的特征向量
            for client_id in list(client_features.keys()):
                feat = client_features[client_id]
                if len(feat) != common_dim:
                    if len(feat) < common_dim:
                        # 如果特征太短，使用填充
                        client_features[client_id] = np.pad(feat, (0, common_dim - len(feat)), 'constant')
                    else:
                        # 如果特征太长，进行裁剪
                        client_features[client_id] = feat[:common_dim]
        
        # 尝试K-means聚类
        if len(client_features) >= self.num_clusters:
            try:
                from sklearn.cluster import KMeans
                # 转换为矩阵
                feature_client_ids = list(client_features.keys())
                features_matrix = np.vstack([client_features[cid] for cid in feature_client_ids])
                
                # 标准化特征
                mean = np.mean(features_matrix, axis=0)
                std = np.std(features_matrix, axis=0) + 1e-8
                features_matrix = (features_matrix - mean) / std
                
                # 执行K-means
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
                kmeans.fit(features_matrix)
                
                # 构建聚类映射
                for i, label in enumerate(kmeans.labels_):
                    client_id = feature_client_ids[i]
                    clusters[label].append(client_id)
                
                # 处理没有特征的客户端 - 平均分配
                remaining_clients = [cid for cid in client_ids if cid not in client_features]
                for i, client_id in enumerate(remaining_clients):
                    target_cluster = i % self.num_clusters
                    clusters[target_cluster].append(client_id)
                    
            except Exception as e:
                print(f"K-means聚类失败: {str(e)}，使用备选方案")
                # 备用方案 - 均匀分配
                for i, client_id in enumerate(client_ids):
                    cluster_idx = i % self.num_clusters
                    clusters[cluster_idx].append(client_id)
        else:
            # 备用方案：均匀分配
            for i, client_id in enumerate(client_ids):
                cluster_idx = i % self.num_clusters
                clusters[cluster_idx].append(client_id)
        
        # 记录聚类结果
        self.clustering_history.append({
            'timestamp': time.time(),
            'clusters': copy.deepcopy(clusters),
            'num_clients': len(client_ids)
        })
            
        return clusters

# 主函数
def main():
    """主函数，串行版TierHFL实现 - 增强版"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 添加新参数
    args.initial_phase_rounds = 10  # 初始阶段轮数
    args.alternating_phase_rounds = 70  # 交替训练阶段轮数
    args.fine_tuning_phase_rounds = 20  # 精细调整阶段轮数
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logging(args)
    logger.info("初始化TierHFL: 串行版本 - 增强版训练策略")
    
    # 设置默认设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"默认设备: {device}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args)
    
    # 获取数据集信息
    if args.dataset != "cinic10":
        train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
    else:
        train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, _ = dataset
    
    # 加载全局测试集
    logger.info("加载全局IID测试集用于评估泛化性能...")
    global_test_loader = load_global_test_set(args)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配异构资源...")
    client_resources = allocate_device_resources(args.client_number)
    
    # 创建客户端管理器
    logger.info("创建客户端管理器...")
    client_manager = TierHFLClientManager()
    
    # 注册客户端
    for client_id in range(args.client_number):
        resource = client_resources[client_id]
        tier = resource["tier"]
        
        # 创建客户端
        client = client_manager.add_client(
            client_id=client_id,
            tier=tier,
            train_data=train_data_local_dict[client_id],
            test_data=test_data_local_dict[client_id],
            device=device,
            lr=args.lr,
            local_epochs=args.client_epoch
        )

    # 确定输入通道数
    input_channels = 1 if args.dataset == "fashion_mnist" else 3

    # 创建客户端模型
    logger.info(f"创建双路径客户端模型...")
    client_models = {}
    for client_id, resource in client_resources.items():
        tier = resource["tier"]
        client_models[client_id] = TierAwareClientModel(
            num_classes=class_num, 
            tier=tier,
            model_type=args.model,
            input_channels=input_channels  # 添加输入通道参数
        )

    # 创建服务器特征提取模型
    logger.info("创建服务器特征提取模型...")
    server_model = EnhancedServerModel(
        model_type=args.model,
        feature_dim=128,
        input_channels=input_channels  # 添加输入通道参数
    ).to(device)
    
    # 创建全局分类器
    logger.info("创建全局分类器...")
    global_classifier = ImprovedGlobalClassifier(
        feature_dim=128, 
        num_classes=class_num
    ).to(device)
    
    # 创建统一损失函数
    logger.info("创建统一损失函数...")
    loss_fn = EnhancedUnifiedLoss(init_alpha=args.init_alpha, init_beta=args.init_lambda)
    
    # 创建稳定化聚合器
    logger.info("创建稳定化聚合器...")
    aggregator = StabilizedAggregator(beta=args.beta, device=device)
    
    # 创建客户端聚类器
    logger.info("创建数据分布聚类器...")
    clusterer = ModelFeatureClusterer(num_clusters=args.n_clusters)
    
    # 对客户端进行初始聚类
    logger.info("执行初始客户端聚类...")
    client_ids = list(range(args.client_number))
    cluster_map = clusterer.cluster_clients(
        client_models=client_models,
        client_ids=client_ids
    )
    
    # 打印初始聚类信息
    print_cluster_info(cluster_map, client_resources, logger)
    
    # 创建串行训练器
    logger.info("创建增强版串行训练器...")
    trainer = SimpleSerialTrainer(
        client_manager=client_manager,
        server_model=server_model,
        global_classifier=global_classifier,
        loss_fn=loss_fn,
        device=device
    )
    
    # 注册客户端模型
    trainer.register_client_models(client_models)
    
    # 设置训练环境
    trainer.setup_training(cluster_map=cluster_map)

    # 创建诊断监控器
    logger.info("创建诊断监控器...")
    diagnostic_monitor = EnhancedTierHFLDiagnosticMonitor(device=device)
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    prev_global_acc = 0.0
    
    # 在训练开始前进行初始验证
    initial_validation = validate_server_effectiveness(
        args, 
        client_models,
        server_model, 
        global_classifier,
        global_test_loader, 
        test_data_local_dict
    )

    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 添加训练阶段信息
        if round_idx < args.initial_phase_rounds:
            logger.info("当前处于初始特征学习阶段")
        elif round_idx < args.initial_phase_rounds + args.alternating_phase_rounds:
            logger.info("当前处于交替训练阶段")
        else:
            logger.info("当前处于精细调整阶段")
        
        # 执行训练 - 传递增强版诊断监控器
        train_results, eval_results, shared_states, training_time = trainer.execute_round(
            round_idx=round_idx, 
            total_rounds=args.rounds,
            diagnostic_monitor=diagnostic_monitor
        )
        
        # 聚合过程
        logger.info("聚合客户端共享层...")
        aggregation_start_time = time.time()
        
        aggregated_shared_state = trainer.aggregate_client_shared_layers(shared_states)
        
        # 分析聚合权重
        if aggregated_shared_state and diagnostic_monitor:
            # 创建聚合权重分析（简化版）
            aggregation_weights = {}
            for client_id in shared_states.keys():
                aggregation_weights[client_id] = 1.0 / len(shared_states)  # 均等权重
            
            diagnostic_monitor.analyze_aggregation_weights(aggregation_weights, round_idx)
        
        # 聚合服务器模型
        logger.info("聚合服务器模型...")
        aggregated_server_model = trainer.aggregate_server_models(eval_results)

        logger.info("聚合全局分类器...")
        aggregated_global_classifier = trainer.aggregate_global_classifiers(eval_results)
        
        # 更新模型
        logger.info("更新客户端共享层...")
        trainer.update_client_shared_layers(aggregated_shared_state)
        
        logger.info("更新服务器模型...")
        trainer.update_server_models(aggregated_server_model, aggregated_global_classifier)
        
        aggregation_time = time.time() - aggregation_start_time
        
        # 评估全局模型
        tier1_clients = [cid for cid, resource in client_resources.items() if resource['tier'] == 1]
        if tier1_clients:
            sample_client_id = tier1_clients[0]
        else:
            sample_client_id = list(client_models.keys())[0]
            
        global_model_accuracy = evaluate_global_model(
            client_models[sample_client_id], server_model, global_classifier, 
            global_test_loader, device)
        
        # 计算平均准确率
        avg_local_acc = np.mean([result.get('local_accuracy', 0) for result in eval_results.values()])
        avg_global_acc = np.mean([result.get('global_accuracy', 0) for result in eval_results.values()])
        
        # 更新最佳准确率
        is_best = global_model_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_model_accuracy
            try:
                torch.save({
                    'client_model': client_models[sample_client_id].state_dict(),
                    'server_model': server_model.state_dict(),
                    'global_classifier': global_classifier.state_dict(),
                    'round': round_idx,
                    'accuracy': best_accuracy
                }, f"{args.running_name}_best_model.pth")
                logger.info(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"保存模型失败: {str(e)}")
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"本地平均准确率: {avg_local_acc:.2f}%, 全局平均准确率: {avg_global_acc:.2f}%")
        logger.info(f"全局模型在独立测试集上的准确率: {global_model_accuracy:.2f}%")
        logger.info(f"最佳准确率: {best_accuracy:.2f}%")
        logger.info(f"轮次总时间: {round_time:.2f}秒, 训练: {training_time:.2f}秒, 聚合: {aggregation_time:.2f}秒")
        
        # 记录准确率变化
        acc_change = global_model_accuracy - prev_global_acc
        prev_global_acc = global_model_accuracy
        logger.info(f"全局准确率变化: {acc_change:.2f}%")
        
        # === 增强版诊断报告 ===
        if round_idx % 3 == 0:  # 每3轮生成一次详细报告
            logger.info("=== 增强版诊断报告 ===")
            comprehensive_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            
            logger.info(f"系统健康状态: {comprehensive_report['overall_health'].upper()}")
            
            if comprehensive_report['critical_issues']:
                logger.error("严重问题:")
                for issue in comprehensive_report['critical_issues']:
                    logger.error(f"  🚨 {issue}")
            
            if comprehensive_report['warnings']:
                logger.warning("警告:")
                for warning in comprehensive_report['warnings']:
                    logger.warning(f"  ⚠️ {warning}")
            
            if comprehensive_report['recommendations']:
                logger.info("优化建议:")
                for rec in comprehensive_report['recommendations']:
                    logger.info(f"  💡 {rec}")
            
            # 如果检测到分类器崩溃，提供紧急建议
            if diagnostic_monitor.collapse_detected:
                logger.error("🚨 检测到分类器崩溃! 紧急处理建议:")
                logger.error("  1. 立即降低学习率至当前值的10%")
                logger.error("  2. 检查损失权重平衡设置")
                logger.error("  3. 考虑重新初始化全局分类器")
                logger.error("  4. 增加批量大小以提高训练稳定性")
        
        # 导出增强版诊断指标到wandb
        try:
            diagnostic_monitor.export_metrics_to_wandb(wandb)
        except Exception as e:
            logger.error(f"导出增强诊断指标失败: {str(e)}")
        
        # 记录到wandb（添加健康状态指标）
        try:
            # 获取当前健康状态
            current_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            health_score = {'good': 1.0, 'warning': 0.7, 'poor': 0.4, 'critical': 0.0, 'error': 0.0}.get(
                current_report['overall_health'], 0.5)
            
            metrics = {
                "round": round_idx + 1,
                "global/test_accuracy": global_model_accuracy,
                "global/best_accuracy": best_accuracy,
                "local/avg_accuracy": avg_local_acc,
                "global/avg_accuracy": avg_global_acc,
                "time/round_seconds": round_time,
                "time/training_seconds": training_time,
                "time/aggregation_seconds": aggregation_time,
                "global/accuracy_change": acc_change,
                "diagnostic/system_health_score": health_score,
                "diagnostic/classifier_collapsed": 1 if diagnostic_monitor.collapse_detected else 0,
                "training/phase": 1 if round_idx < args.initial_phase_rounds else 
                                (2 if round_idx < args.initial_phase_rounds + args.alternating_phase_rounds else 3)
            }
            
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"记录wandb指标失败: {str(e)}")
        
        # 每10轮重新聚类一次
        if (round_idx + 1) % 10 == 0 and round_idx >= args.initial_phase_rounds:
            logger.info("重新进行客户端聚类...")
            try:
                cluster_map = clusterer.cluster_clients(
                    client_models=client_models,
                    client_ids=client_ids,
                    eval_dataset=global_test_loader
                )
                trainer.setup_training(cluster_map=cluster_map)
                print_cluster_info(cluster_map, client_resources, logger)
            except Exception as e:
                logger.error(f"重新聚类失败: {str(e)}")
        
        # 动态学习率调整 - 基于健康状态
        if round_idx > 0 and round_idx % 10 == 0:
            logger.info("基于系统健康状态调整学习率...")
            
            # 获取当前健康状态
            current_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            
            for client_id in range(args.client_number):
                client = client_manager.get_client(client_id)
                if client:
                    # 根据健康状态调整学习率衰减程度
                    if current_report['overall_health'] == 'critical':
                        # 严重问题：大幅降低学习率
                        client.lr *= 0.5
                        logger.info(f"客户端 {client_id} 学习率紧急调整为: {client.lr:.6f}")
                    elif current_report['overall_health'] == 'poor':
                        # 较多问题：加大衰减
                        client.lr *= args.lr_factor * 0.8
                        logger.info(f"客户端 {client_id} 学习率加大衰减为: {client.lr:.6f}")
                    else:
                        # 正常衰减
                        if round_idx < args.initial_phase_rounds:
                            client.lr *= args.lr_factor
                        elif round_idx < args.initial_phase_rounds + args.alternating_phase_rounds:
                            client.lr *= args.lr_factor * 0.9
                        else:
                            client.lr *= args.lr_factor * 0.8
                        logger.info(f"客户端 {client_id} 学习率正常更新为: {client.lr:.6f}")

        # 每隔5轮进行一次服务器有效性验证
        if (round_idx + 1) % 5 == 0:
            round_validation = validate_server_effectiveness(
                args, 
                client_models,
                server_model, 
                global_classifier,
                global_test_loader, 
                test_data_local_dict
            )
            
            # 记录验证结果
            try:
                wandb.log({
                    "round": round_idx + 1,
                    "validation/feature_quality": round_validation['feature_quality'],
                    "validation/heterogeneity_adaptation": round_validation['heterogeneity_adaptation'],
                    "validation/simple_classifier_acc": round_validation['simple_classifier_acc']
                })
            except Exception as e:
                logger.error(f"记录wandb验证指标失败: {str(e)}")
    
    # 训练完成后的最终报告
    logger.info("=== 最终诊断报告 ===")
    final_report = diagnostic_monitor.comprehensive_diagnostic_report(args.rounds - 1)
    
    logger.info(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")
    logger.info(f"最终系统健康状态: {final_report['overall_health'].upper()}")
    
    if final_report['critical_issues']:
        logger.error("遗留的严重问题:")
        for issue in final_report['critical_issues']:
            logger.error(f"  - {issue}")
    
    if final_report['recommendations']:
        logger.info("后续优化建议:")
        for rec in final_report['recommendations']:
            logger.info(f"  - {rec}")
    
    # 关闭wandb
    try:
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()
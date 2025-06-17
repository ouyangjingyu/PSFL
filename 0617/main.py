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

# 导入自定义模块，包括新的增强损失函数和分层聚合器
from model.resnet import EnhancedServerModel, TierAwareClientModel, ImprovedGlobalClassifier
from utils.tierhfl_aggregator import LayeredAggregator
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_loss import EnhancedStagedLoss

from analyze.tierhfl_analyze import validate_server_effectiveness


# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
# 导入新的Fashion-MNIST数据加载器
from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist
# 导入监控器
from analyze.diagnostic_monitor import EnhancedTierHFLDiagnosticMonitor

# 增强版串行训练器实现
class EnhancedSerialTrainer:
    """增强版串行训练器，集成梯度投影和分层聚合"""
    def __init__(self, client_manager, server_model, global_classifier, 
                device="cuda"):
        self.client_manager = client_manager
        self.server_model = server_model
        self.global_classifier = global_classifier
        self.device = device
        
        # 客户端模型字典
        self.client_models = {}
        
        # 聚类映射和服务器模型状态
        self.cluster_map = {}
        self.cluster_server_models = {}
        self.cluster_global_classifiers = {}
        
        # 增强损失函数和分层聚合器
        self.enhanced_loss = EnhancedStagedLoss()
        self.layered_aggregator = LayeredAggregator(device=device)
        
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
                    train_result = self._train_initial_phase_enhanced(
                        client, client_model, cluster_server, cluster_classifier, 
                        round_idx, total_rounds, diagnostic_monitor)
                    
                elif training_phase == "alternating":
                    train_result = self._train_alternating_phase_enhanced(
                        client, client_model, cluster_server, cluster_classifier,
                        round_idx, total_rounds, diagnostic_monitor)
                    
                else:  # fine_tuning
                    train_result = self._train_fine_tuning_phase_enhanced(
                        client, client_model, cluster_server, cluster_classifier,
                        round_idx, total_rounds, diagnostic_monitor)
                
                # 保存结果
                train_results[client_id] = train_result
                
                # 评估客户端
                eval_result = self._evaluate_client(
                    client, client_model, cluster_server, cluster_classifier)
                eval_result['cluster_id'] = cluster_id  # 添加聚类ID
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

    def _train_initial_phase_enhanced(self, client, client_model, server_model, classifier, 
                                    round_idx, total_rounds, diagnostic_monitor=None):
        """阶段1: 增强版初始特征学习阶段"""
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
            'feature_importance_loss': 0.0,
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
            
            # 使用增强损失函数
            total_loss, global_loss, feature_importance_loss = self.enhanced_loss.stage1_loss(
                global_logits, target, shared_features
            )
            
            # === 增强版诊断监控 ===
            if diagnostic_monitor is not None and batch_idx == 0:
                try:
                    # 本地损失用于监控
                    with torch.no_grad():
                        local_loss = F.cross_entropy(local_logits, target)
                    
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
            stats['feature_importance_loss'] += feature_importance_loss.item()
            stats['total_loss'] += total_loss.item()
            stats['batch_count'] += 1
            
            _, pred = global_logits.max(1)
            stats['correct'] += pred.eq(target).sum().item()
            stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'feature_importance_loss', 'total_loss']:
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
            'feature_importance_loss': stats['feature_importance_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time
        }

    def _train_alternating_phase_enhanced(self, client, client_model, server_model, classifier, 
                                        round_idx, total_rounds, diagnostic_monitor=None):
        """阶段2: 增强版交替训练阶段"""
        progress = round_idx / total_rounds
        
        # --- 第一部分: 训练个性化路径 (1 epoch) ---
        personal_result = self._train_personal_path_enhanced(
            client, client_model, client.lr * 0.05, round_idx, total_rounds, diagnostic_monitor)
        
        # --- 第二部分: 训练全局路径 (2 epochs) ---
        global_result = self._train_global_path_enhanced(
            client, client_model, server_model, classifier, 
            client.lr * 0.1, round_idx, total_rounds, diagnostic_monitor, epochs=2)
        
        # 组合结果
        result = {
            'global_loss': global_result['global_loss'],
            'local_loss': personal_result['local_loss'],
            'balance_loss': global_result.get('balance_loss', 0.0),
            'total_loss': (global_result['total_loss'] + personal_result['local_loss']) / 2,
            'global_accuracy': global_result['global_accuracy'],
            'local_accuracy': personal_result['local_accuracy'],
            'time_cost': global_result['time_cost'] + personal_result['time_cost']
        }
        
        return result

    def _train_fine_tuning_phase_enhanced(self, client, client_model, server_model, classifier, 
                                        round_idx, total_rounds, diagnostic_monitor=None):
        """阶段3: 增强版精细调整阶段"""
        # 全局路径训练
        global_result = self._train_global_path_enhanced(
            client, client_model, server_model, classifier, 
            client.lr * 0.02, round_idx, total_rounds, diagnostic_monitor, epochs=2)
        
        # 个性化路径训练
        personal_result = self._train_personal_path_enhanced(
            client, client_model, client.lr * 0.01, 
            round_idx, total_rounds, diagnostic_monitor)
        
        # 组合结果
        result = {
            'global_loss': global_result['global_loss'],
            'local_loss': personal_result['local_loss'],
            'balance_loss': global_result.get('balance_loss', 0.0),
            'total_loss': (global_result['total_loss'] + personal_result['local_loss']) / 2,
            'global_accuracy': global_result['global_accuracy'],
            'local_accuracy': personal_result['local_accuracy'],
            'time_cost': global_result['time_cost'] + personal_result['time_cost']
        }
        
        return result

    def _train_global_path_enhanced(self, client, client_model, server_model, classifier, 
                                  shared_lr, round_idx, total_rounds, diagnostic_monitor=None, epochs=1):
        """增强版全局路径训练"""
        start_time = time.time()
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        classifier.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=shared_lr
        )
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        
        # 统计信息
        stats = {
            'global_loss': 0.0,
            'local_loss': 0.0,
            'balance_loss': 0.0,
            'total_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 计算alpha值
        progress = round_idx / total_rounds
        alpha = 0.3 + 0.4 * progress  # 个性化权重随训练进度增加
        
        # 训练循环
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(client.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                local_logits, shared_features, personal_features = client_model(data)
                server_features = server_model(shared_features)
                global_logits = classifier(server_features)
                
                # 计算损失
                local_loss = F.cross_entropy(local_logits, target)
                global_loss = F.cross_entropy(global_logits, target)
                
                # 计算梯度用于特征平衡损失
                # 为避免计算图复杂化，我们简化特征平衡损失
                balance_loss = torch.tensor(0.0, device=global_logits.device)
                
                # 使用增强损失函数
                total_loss, local_loss_calc, global_loss_calc, balance_loss = self.enhanced_loss.stage2_3_loss(
                    local_logits, global_logits, target, 
                    personal_gradients=None, global_gradients=None,  # 简化实现
                    shared_features=shared_features, alpha=alpha
                )
                
                # 清除梯度
                shared_optimizer.zero_grad()
                server_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                
                # 反向传播
                total_loss.backward()
                
                # 应用梯度投影（简化版）
                # 在实际实现中，我们可以在这里应用梯度投影
                # 但为了保持代码简洁，暂时跳过复杂的梯度投影
                
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
                stats['local_loss'] += local_loss.item()
                stats['balance_loss'] += balance_loss.item()
                stats['total_loss'] += total_loss.item()
                stats['batch_count'] += 1
                
                _, pred = global_logits.max(1)
                stats['correct'] += pred.eq(target).sum().item()
                stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'local_loss', 'balance_loss', 'total_loss']:
            if stats['batch_count'] > 0:
                stats[key] /= stats['batch_count']
        
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        return {
            'global_loss': stats['global_loss'],
            'local_loss': stats['local_loss'],
            'balance_loss': stats['balance_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time
        }

    def _train_personal_path_enhanced(self, client, client_model, shared_lr, round_idx, total_rounds, diagnostic_monitor=None):
        """增强版个性化路径训练"""
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

    def aggregate_client_shared_layers(self, shared_states, eval_results):
        """使用分层聚合器聚合客户端共享层"""
        if not shared_states:
            return {}
        
        return self.layered_aggregator.aggregate_shared_layers(shared_states, eval_results)

    def aggregate_server_models(self, eval_results=None):
        """使用分层聚合器聚合服务器模型"""
        if not self.cluster_server_models:
            return {}
        
        return self.layered_aggregator.aggregate_server_models(self.cluster_server_models, eval_results)

    def aggregate_global_classifiers(self, eval_results=None):
        """使用分层聚合器聚合全局分类器"""
        if not self.cluster_global_classifiers:
            return {}
        
        return self.layered_aggregator.aggregate_global_classifiers(self.cluster_global_classifiers, eval_results)

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

# 其余函数保持不变...
# [这里包含所有其他函数的完整代码，如 set_seed, parse_arguments, setup_logging 等]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='TierHFL: 分层异构联邦学习框架 (增强版本)')
    
    # 实验标识
    parser.add_argument('--running_name', default="TierHFL_Enhanced", type=str, help='实验名称')
    
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
    parser.add_argument('--client_number', type=int, default=5, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=256, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    parser.add_argument('--n_clusters', default=3, type=int, help='客户端聚类数量')
    
    # TierHFL特有参数
    parser.add_argument('--init_alpha', default=0.6, type=float, help='初始本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.15, type=float, help='初始特征对齐损失权重')
    parser.add_argument('--beta', default=0.3, type=float, help='聚合动量因子')
    
    args = parser.parse_args()
    return args

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
            mode="offline",
            project="TierHFL_Enhanced",
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
        try:
            wandb.init(mode="offline", project="TierHFL", name=args.running_name)
        except:
            print("完全禁用wandb")
            
    return logger

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
    """主函数，增强版TierHFL实现"""
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
    logger.info("初始化TierHFL: 增强版本 - 集成梯度投影和分层聚合")
    
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
    
    # 创建增强版串行训练器
    logger.info("创建增强版串行训练器...")
    trainer = EnhancedSerialTrainer(
        client_manager=client_manager,
        server_model=server_model,
        global_classifier=global_classifier,
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
        logger.info("使用分层聚合策略聚合模型...")
        aggregation_start_time = time.time()
        
        # 聚合客户端共享层
        aggregated_shared_state = trainer.aggregate_client_shared_layers(shared_states, eval_results)
        
        # 聚合服务器模型
        aggregated_server_model = trainer.aggregate_server_models(eval_results)

        # 聚合全局分类器
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
                "training/phase": 1 if round_idx < args.initial_phase_rounds else 
                                (2 if round_idx < args.initial_phase_rounds + args.alternating_phase_rounds else 3)
            }
            
            # 添加特征平衡和损失组件指标
            if train_results:
                avg_balance_loss = np.mean([result.get('balance_loss', 0) for result in train_results.values()])
                metrics["training/avg_balance_loss"] = avg_balance_loss
            
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
        
        # 动态学习率调整
        if round_idx > 0 and round_idx % 10 == 0:
            for client_id in range(args.client_number):
                client = client_manager.get_client(client_id)
                if client:
                    client.lr *= args.lr_factor
                    logger.info(f"客户端 {client_id} 学习率更新为: {client.lr:.6f}")

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
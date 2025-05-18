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
    
    def execute_round(self, round_idx, total_rounds):
        """执行一轮训练 - 修复版"""
        start_time = time.time()
        
        # 结果容器
        train_results = {}
        eval_results = {}
        shared_states = {}
        
        # 依次处理每个聚类
        for cluster_id, client_ids in self.cluster_map.items():
            logging.info(f"处理聚类 {cluster_id}, 包含 {len(client_ids)} 个客户端")
            
            # 创建聚类特定的服务器模型和全局分类器
            cluster_server = copy.deepcopy(self.server_model).to(self.device)
            cluster_server.load_state_dict(self.cluster_server_models[cluster_id])
            
            # 创建聚类特定的全局分类器
            cluster_classifier = copy.deepcopy(self.global_classifier).to(self.device)
            cluster_classifier.load_state_dict(self.cluster_global_classifiers[cluster_id])
            
            # 依次处理聚类中的每个客户端
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if not client or client_id not in self.client_models:
                    continue
                
                logging.info(f"训练客户端 {client_id} (Tier: {client.tier})")
                
                # 准备客户端模型
                client_model = self.client_models[client_id].to(self.device)
                client.model = client_model
                
                # 阶段一：训练客户端个性化层
                personal_result = self._train_personal_layers(
                    client, client_model, round_idx, total_rounds)
                
                # 阶段二：训练服务器模型和共享层 - 使用聚类特定的全局分类器
                server_result = self._train_server_model(
                    client, client_model, cluster_server, cluster_classifier, 
                    personal_result, round_idx, total_rounds)
                
                # 保存训练结果
                train_results[client_id] = {
                    'local_loss': personal_result['local_loss'],
                    'global_loss': server_result['global_loss'],
                    'total_loss': server_result['total_loss'],
                    'local_accuracy': personal_result['local_accuracy'],
                    'global_accuracy': server_result['global_accuracy'],
                    'training_time': personal_result['time_cost'] + server_result['time_cost']
                }
                
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
                
                # 释放GPU内存
                torch.cuda.empty_cache()
            
            # 保存聚类服务器模型和全局分类器
            self.cluster_server_models[cluster_id] = cluster_server.cpu().state_dict()
            self.cluster_global_classifiers[cluster_id] = cluster_classifier.cpu().state_dict()
        
        # 计算总训练时间
        training_time = time.time() - start_time
        
        return train_results, eval_results, shared_states, training_time
    
    def _train_personal_layers(self, client, client_model, round_idx, total_rounds):
        """阶段一：训练客户端个性化层"""
        start_time = time.time()
        
        # 设置模式
        client_model.train()
        
        # 冻结共享层
        for name, param in client_model.named_parameters():
            if 'shared_base' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' not in n],
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
                # 移至设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向传播
                local_logits, shared_features, _ = client_model(data)
                
                # 计算本地损失
                local_loss = F.cross_entropy(local_logits, target)
                
                # 反向传播
                local_loss.backward()
                
                # 更新参数
                optimizer.step()
                
                # 更新统计信息
                stats['local_loss'] += local_loss.item()
                stats['batch_count'] += 1
                
                # 计算准确率
                _, pred = local_logits.max(1)
                batch_correct = pred.eq(target).sum().item()
                batch_total = target.size(0)
                
                stats['correct'] += batch_correct
                stats['total'] += batch_total
        
        # 计算平均值和总体准确率
        avg_local_loss = stats['local_loss'] / max(1, stats['batch_count'])
        local_accuracy = 100.0 * stats['correct'] / max(1, stats['total'])
        
        # 解冻共享层，为第二阶段训练准备
        for name, param in client_model.named_parameters():
            param.requires_grad = True
        
        return {
            'local_loss': avg_local_loss,
            'local_accuracy': local_accuracy,
            'time_cost': time.time() - start_time
        }
    
    def _train_server_model(self, client, client_model, server_model, classifier,
                           personal_result, round_idx, total_rounds):
        """阶段二：训练服务器模型和共享层 - 修复版，确保完整计算图"""
        start_time = time.time()
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        classifier.train()
        
        # 创建针对客户端共享层的优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=0.0005
        )
        
        # 创建针对服务器模型的优化器
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        
        # 创建针对全局分类器的优化器 - 添加学习率预热
        base_lr = 0.0005
        warmup_factor = min(1.0, 0.2 + (round_idx * 0.1))  # 更缓慢的预热
        classifier_lr = base_lr * warmup_factor
        # 创建针对全局分类器的优化器 - 添加L2正则化
        classifier_optimizer = torch.optim.Adam(
            classifier.parameters(), 
            lr=classifier_lr,
            weight_decay=0.01  # 添加L2正则化
        )

        # 统计信息
        stats = {
            'total_loss': 0.0,
            'global_loss': 0.0,
            'feature_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 训练循环
        for data, target in client.train_data:
            # 移至设备
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除所有梯度
            shared_optimizer.zero_grad()
            server_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            # 前向传播 - 修复：确保完整的计算图
            local_logits, shared_features, personal_features = client_model(data)
            
            # 服务器前向传播
            server_features = server_model(shared_features)
            global_logits = classifier(server_features)
            
            # 计算损失
            progress = round_idx / total_rounds
            alpha = 0.3 + 0.4 * progress  # 动态调整本地与全局损失平衡
            beta = 0.1 * (1 - abs(2 * progress - 1))  # 动态调整特征对齐损失权重
            
            global_loss = F.cross_entropy(global_logits, target)
            local_loss = personal_result['local_loss']  # 从阶段一获取
            
            # 计算特征对齐损失
            feature_loss = torch.tensor(0.0, device=self.device)
            if personal_features is not None:
                personal_features_pooled = F.adaptive_avg_pool2d(personal_features, (1, 1)).flatten(1)
                server_features_flat = server_features.flatten(1) if server_features.dim() > 2 else server_features
                
                # 确保维度匹配
                min_dim = min(personal_features_pooled.size(1), server_features_flat.size(1))
                personal_features_pooled = personal_features_pooled[:, :min_dim]
                server_features_flat = server_features_flat[:, :min_dim]
                
                # 计算特征对齐损失
                personal_norm = F.normalize(personal_features_pooled, dim=1)
                server_norm = F.normalize(server_features_flat, dim=1)
                
                cos_sim = torch.sum(personal_norm * server_norm, dim=1).mean()
                feature_loss = 1.0 - cos_sim
            
            # 总损失
            total_loss = (1 - alpha) * global_loss + alpha * local_loss + beta * feature_loss
            
            # 反向传播 - 确保梯度流向所有相关参数
            total_loss.backward()
            
            # 添加梯度裁剪以提高训练稳定性
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
            shared_params = [p for n, p in client_model.named_parameters() if 'shared_base' in n]
            torch.nn.utils.clip_grad_norm_(shared_params, max_norm=1.0)

            # 更新参数 - 使用标准优化器步骤
            server_optimizer.step()
            classifier_optimizer.step()
            shared_optimizer.step()
            
            # 更新统计信息
            stats['total_loss'] += total_loss.item()
            stats['global_loss'] += global_loss.item()
            stats['feature_loss'] += feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss
            stats['batch_count'] += 1
            
            # 计算准确率
            _, pred = global_logits.max(1)
            stats['correct'] += pred.eq(target).sum().item()
            stats['total'] += target.size(0)
        
        # 计算平均值
        if stats['batch_count'] > 0:
            for key in ['total_loss', 'global_loss', 'feature_loss']:
                stats[key] /= stats['batch_count']
        
        # 计算全局准确率
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        return {
            'total_loss': stats['total_loss'],
            'global_loss': stats['global_loss'],
            'feature_loss': stats['feature_loss'],
            'global_accuracy': stats['global_accuracy'],
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
    parser.add_argument('--dataset', type=str, default='cifar10', help='训练数据集')
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
    """主函数，串行版TierHFL实现"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logging(args)
    logger.info("初始化TierHFL: 串行版本")
    
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
    
    # 创建客户端模型
    logger.info(f"创建双路径客户端模型...")
    client_models = {}
    for client_id, resource in client_resources.items():
        tier = resource["tier"]
        client_models[client_id] = TierAwareClientModel(
            num_classes=class_num, 
            tier=tier,
            model_type=args.model
        )

    # 创建服务器特征提取模型
    logger.info("创建服务器特征提取模型...")
    server_model = EnhancedServerModel(
        model_type=args.model,
        feature_dim=128
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
    logger.info("创建串行训练器...")
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
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    prev_global_acc = 0.0
    
    # 在训练开始前进行初始验证
    initial_validation = validate_server_effectiveness(
        args, 
        client_models,  # 传递整个client_models字典
        server_model, 
        global_classifier,
        global_test_loader, 
        test_data_local_dict
    )


    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 执行训练
        train_results, eval_results, shared_states, training_time = trainer.execute_round(
            round_idx=round_idx, 
            total_rounds=args.rounds
        )
        
        # 1. 聚合客户端共享层
        logger.info("聚合客户端共享层...")
        aggregation_start_time = time.time()
        
        aggregated_shared_state = trainer.aggregate_client_shared_layers(shared_states)
        
        # 2. 聚合服务器模型
        logger.info("聚合服务器模型...")
        aggregated_server_model = trainer.aggregate_server_models(eval_results)

        logger.info("聚合全局分类器...")
        aggregated_global_classifier = trainer.aggregate_global_classifiers(eval_results)
        
        # 3. 更新客户端共享层
        logger.info("更新客户端共享层...")
        trainer.update_client_shared_layers(aggregated_shared_state)
        
        # 4. 更新服务器模型
        logger.info("更新服务器模型...")
        trainer.update_server_models(aggregated_server_model)
        
        aggregation_time = time.time() - aggregation_start_time
        
        # 评估全局模型
        # 选择一个tier 1的客户端进行评估
        tier1_clients = [cid for cid, resource in client_resources.items() if resource['tier'] == 1]
        if tier1_clients:
            sample_client_id = tier1_clients[0]
        else:
            sample_client_id = list(client_models.keys())[0]
            
        # 评估全局模型 - 使用修复版本
        global_model_accuracy = evaluate_global_model(
            client_models[sample_client_id], server_model, global_classifier, 
            global_test_loader, device)
        
        # 计算平均准确率
        avg_local_acc = np.mean([result.get('local_accuracy', 0) for result in eval_results.values()])
        avg_global_acc = np.mean([result.get('global_accuracy', 0) for result in eval_results.values()])
        
        # 打印个别客户端性能，用于调试
        for client_id in range(min(3, args.client_number)):  # 只打印前3个客户端的数据作为样本
            if client_id in eval_results:
                logger.info(f"客户端 {client_id} - 本地准确率: {eval_results[client_id]['local_accuracy']:.2f}%, "
                           f"全局准确率: {eval_results[client_id]['global_accuracy']:.2f}%")
        
        # 更新最佳准确率
        is_best = global_model_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_model_accuracy
            # 保存最佳模型
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
        
        # 记录到wandb
        try:
            metrics = {
                "round": round_idx + 1,
                "global/test_accuracy": global_model_accuracy,
                "global/best_accuracy": best_accuracy,
                "local/avg_accuracy": avg_local_acc,
                "global/avg_accuracy": avg_global_acc,
                "time/round_seconds": round_time,
                "time/training_seconds": training_time,
                "time/aggregation_seconds": aggregation_time,
                "global/accuracy_change": acc_change
            }
            
            # 记录每个客户端的性能
            for client_id in range(args.client_number):
                if client_id in eval_results:
                    metrics[f"client/{client_id}/local_accuracy"] = eval_results[client_id].get('local_accuracy', 0)
                    metrics[f"client/{client_id}/global_accuracy"] = eval_results[client_id].get('global_accuracy', 0)
            
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"记录wandb指标失败: {str(e)}")
        
        # 每10轮重新聚类一次
        if (round_idx + 1) % 10 == 0 and round_idx < args.rounds - 10:
            logger.info("重新进行客户端聚类...")
            try:
                cluster_map = clusterer.cluster_clients(
                    client_models=client_models,
                    client_ids=client_ids
                )
                trainer.setup_training(cluster_map=cluster_map)
                print_cluster_info(cluster_map, client_resources, logger)
            except Exception as e:
                logger.error(f"重新聚类失败: {str(e)}")
        
        # 每10轮更新学习率
        if round_idx > 0 and round_idx % 10 == 0:
            logger.info("更新客户端学习率...")
            for client_id in range(args.client_number):
                client = client_manager.get_client(client_id)
                if client:
                    client.lr *= args.lr_factor
                    logger.info(f"客户端 {client_id} 学习率更新为: {client.lr:.6f}")

        # 每隔3轮进行一次验证
        if (round_idx + 1) % 3 == 0:
            round_validation = validate_server_effectiveness(
                args, 
                client_models,  # 传递整个client_models字典
                server_model, 
                global_classifier,
                global_test_loader, 
                test_data_local_dict
            )

        # 每隔3轮进行一次全局分类器验证
        if (round_idx + 1) % 3 == 0:
            logger.info("验证全局分类器...")
            try:
                verifier = GlobalClassifierVerifier(
                    server_model, 
                    global_classifier,
                    client_models,
                    global_test_loader, 
                    test_data_local_dict,
                    device=device
                )
                verifier.run_all_tests()
            except Exception as e:
                logger.error(f"全局分类器验证失败: {str(e)}")
            
            # 记录验证结果
            try:
                wandb.log({
                    "round": round_idx + 1,
                    "validation/feature_quality": round_validation['feature_quality'],
                    "validation/heterogeneity_adaptation": round_validation['heterogeneity_adaptation'],
                    "validation/identity_leakage": round_validation['identity_leakage']
                })
            except Exception as e:
                logger.error(f"记录wandb指标失败: {str(e)}")
    
    # 训练完成
    logger.info(f"TierHFL串行训练完成! 最佳准确率: {best_accuracy:.2f}%")
    
    # 关闭wandb
    try:
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()
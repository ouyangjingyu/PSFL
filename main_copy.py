# ============================================================================
# Deployment Environment and Resource Profiles:
# The DTFL and the baselines are deployed on a server with the following specifications:
# - Dual-sockets Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz 
# - Four NVIDIA GeForce GTX 1080 Ti GPUs
# - 64 GB of memory

# Each client in the simulation is assigned a distinct simulated CPU and communication resource
# to replicate heterogeneous resources, simulating varying training times based on CPU/network profiles.
# We simulate a heterogeneous environment with varying client capacity in both cross-solo and cross-device FL settings.

# We consider 5 resource profiles:
# 1. 4 CPUs with 100 Mbps
# 2. 2 CPUs with 30 Mbps
# 3. 1 CPU with 30 Mbps
# 4. 0.2 CPU with 30 Mbps
# 5. 0.1 CPU with 10 Mbps communication speed to the server.

# In this implementaion number of tiers is 6 (M=6)
# ============================================================================

# ============================================================================
# Enhanced Heterogeneous Federated Learning Framework
# ============================================================================


import torch
import torch.nn as nn
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

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入优化后的模型
from model.resnet import Bottleneck
from model.resnet import BasicBlock
from model.resnet import ResNet
from model.resnet import UnifiedServerModel
from model.resnet import EnhancedGlobalClassifier
from model.resnet import create_tier_client_model
from model.resnet import create_all_tier_client_models
from model.resnet import create_unified_server_model
from model.resnet import create_unified_classifier

# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from data.cifar10_eval_dataset import get_cifar10_proxy_dataset

# 导入优化后的工具模块
from utils.unified_client_module import ClientManager, EnhancedClient
from utils.parallel_training_framework import create_training_framework
from utils.client_clustering import adaptive_cluster_assignment
from utils.training_strategy import create_training_strategy_for_client

# 导入新增的模块
from utils.unified_training_framework import train_client_with_unified_server
from utils.unified_training_framework import evaluate_client_with_unified_server
from utils.unified_training_framework import FeatureMonitor
from utils.unified_training_framework import UnifiedModelAggregator, UnifiedParallelTrainer

# 设置随机种子，确保实验可复现
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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一服务器架构的异构联邦学习框架')
    
    # 实验标识
    parser.add_argument('--running_name', default="HPFL-Unified", type=str, help='实验名称')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.00075, type=float, help='学习率')
    parser.add_argument('--lr_factor', default=0.85, type=float, help='学习率衰减因子')
    parser.add_argument('--lr_patience', default=10, type=float, help='学习率调整耐心值')
    parser.add_argument('--lr_min', default=1e-6, type=float, help='最小学习率')
    parser.add_argument('--optimizer', default="Adam", type=str, help='优化器: SGD, Adam等')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet110', help='训练使用的神经网络 (resnet110 或 resnet56)')
    parser.add_argument('--groups_per_channel', type=int, default=32, help='GroupNorm的每通道分组数，默认32')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='cifar10', help='训练数据集')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='本地工作节点上数据集的划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=1, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=10, help='分布式集群中的工作节点数量')
    parser.add_argument('--batch_size', type=int, default=200, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    parser.add_argument('--n_clusters', default=3, type=int, help='客户端聚类数量')
    parser.add_argument('--max_workers', default=None, type=int, help='最大并行工作线程数')
    
    args = parser.parse_args()
    return args


def setup_logging(args):
    """设置日志记录"""
    # 配置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.running_name}.log")
        ]
    )
    logger = logging.getLogger("HPFL-Unified")
    
    # 配置wandb
    wandb.init(
        mode="online",
        project="HeterogeneousFL-Unified",
        name=args.running_name,
        config=args,
        tags=[f"model_{args.model}", f"dataset_{args.dataset}", f"clients_{args.client_number}", "UnifiedServer"]
    )
    
    return logger


def load_dataset(args):
    """加载并分割数据集"""
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


def allocate_device_resources(client_number):
    """为客户端分配设备资源"""
    resources = {}
    
    # 在1-7之间随机分配tier
    tier_weights = [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]  # tier 1-7的分布概率
    tiers = random.choices(range(1, 8), weights=tier_weights, k=client_number)
    
    # 为每个客户端分配资源
    for client_id in range(client_number):
        tier = tiers[client_id]
        
        # 根据tier分配计算能力
        if tier <= 2:  # 高能力设备
            compute_power = random.uniform(0.8, 1.0)
        elif tier <= 5:  # 中等能力设备
            compute_power = random.uniform(0.4, 0.8)
        else:  # 低能力设备
            compute_power = random.uniform(0.1, 0.4)
        
        # 分配网络速度 (MB/s)
        if tier <= 2:
            network_speed = random.choice([50, 100, 200])
        elif tier <= 5:
            network_speed = random.choice([20, 30, 50])
        else:
            network_speed = random.choice([5, 10, 20])
        
        # 分配存储容量 (GB)
        if tier <= 2:
            storage_capacity = random.choice([256, 512, 1024])
        elif tier <= 5:
            storage_capacity = random.choice([64, 128, 256])
        else:
            storage_capacity = random.choice([16, 32, 64])
        
        # 存储资源信息
        resources[client_id] = {
            "storage_tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed,
            "storage_capacity": storage_capacity
        }
    
    return resources


def setup_clients(args, dataset, client_resources):
    """设置客户端管理器和客户端"""
    # 提取数据集信息
    if args.dataset != "cinic10":
        _, _, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
    else:
        _, _, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, _ = dataset
    
    # 创建客户端管理器
    client_manager = ClientManager()
    
    # 设置默认设备
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 为每个客户端创建训练策略
    for client_idx in range(args.client_number):
        # 获取客户端资源信息
        resources = client_resources[client_idx]
        tier = resources["storage_tier"]
        
        # 创建优化的学习率
        optimized_lr = args.lr * (1.2 if tier > 5 else (0.8 if tier < 3 else 1.0))
        
        # 添加客户端
        client = client_manager.add_client(
            client_id=client_idx,
            tier=tier,
            train_dataset=train_data_local_dict[client_idx],
            test_dataset=test_data_local_dict[client_idx],
            device=default_device,
            lr=optimized_lr,
            local_epochs=args.client_epoch,
            resources=resources
        )
        
        # 为客户端设置优化的训练策略
        training_strategy = create_training_strategy_for_client(
            tier=tier,
            is_low_resource=(resources["compute_power"] < 0.5)
        )
        client.training_strategy = training_strategy
    
    return client_manager, class_num


def setup_models(args, class_num):
    """设置模型"""
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 选择模型类型
    if args.model == 'resnet110':
        layers = [6, 6, 6, 6, 6, 6]  # ResNet-110的层配置
        block = Bottleneck
    else:  # 默认使用ResNet-56
        layers = [3, 3, 3, 3, 3, 3]  # ResNet-56的层配置
        block = Bottleneck
    
    # 创建所有tier的客户端模型
    print("创建不同tier的客户端模型...")
    client_models = create_all_tier_client_models(
        block=block,
        layers=layers,
        num_classes=class_num,
        groups_per_channel=args.groups_per_channel
    )
    
    # 创建统一的服务器模型和全局分类器
    print("创建统一的服务器模型和全局分类器...")
    unified_server_model, global_classifier = create_unified_server_model(
        block=block,
        layers=layers,
        num_classes=class_num,
        groups_per_channel=args.groups_per_channel
    )
    
    # 将模型移至目标设备
    unified_server_model = unified_server_model.to(device)
    global_classifier = global_classifier.to(device)
    
    print(f"模型类型: {args.model}, 分类数: {class_num}")
    
    return client_models, unified_server_model, global_classifier


def pretrain_and_cluster(client_manager, client_models, args, device):
    """客户端预训练和数据分布感知聚类"""
    print("开始客户端预训练...")
    
    # 创建评估数据集
    eval_dataset = get_cifar10_proxy_dataset(
        option='balanced_test',
        num_samples=1000,
        seed=42
    )
    
    # 为每个客户端分配对应tier的模型
    client_tier_models = {}
    for client_id in range(args.client_number):
        client = client_manager.get_client(client_id)
        if client:
            tier = client.tier
            client_tier_models[client_id] = copy.deepcopy(client_models[tier])
    
    # 执行预训练
    client_ids = list(range(args.client_number))
    start_time = time.time()
    pretrain_results = client_manager.pre_train_clients(client_ids, client_tier_models, epochs=5)
    pretrain_time = time.time() - start_time
    
    # 记录预训练统计信息
    pretrain_stats = {
        "pretrain/total_time": pretrain_time,
        "pretrain/avg_time_per_client": pretrain_time / len(client_ids) if len(client_ids) > 0 else 0
    }
    for client_id, result in pretrain_results.items():
        pretrain_stats[f"pretrain/client_{client_id}/time"] = result['time']
        pretrain_stats[f"pretrain/client_{client_id}/model_size_mb"] = result['model_size_mb']
    
    # 记录到wandb
    wandb.log(pretrain_stats)
    
    # 收集预训练后的模型
    pretrained_models = {}
    for client_id, result in pretrain_results.items():
        model = copy.deepcopy(client_tier_models[client_id])
        model.load_state_dict(result['model_state'])
        pretrained_models[client_id] = model
    
    # 数据分布感知聚类
    print("执行数据分布感知聚类...")
    cluster_start_time = time.time()
    client_models_list = [pretrained_models[idx] for idx in sorted(pretrained_models.keys())]
    client_indices = sorted(pretrained_models.keys())
    
    # 执行聚类
    client_clusters = adaptive_cluster_assignment(
        client_models_list, 
        client_indices,
        eval_dataset,
        device,
        n_clusters=args.n_clusters
    )
    cluster_time = time.time() - cluster_start_time
    
    # 设置客户端聚类信息
    client_manager.set_client_clusters(client_clusters)
    
    # 记录聚类结果
    cluster_stats = {
        "clustering/time": cluster_time,
        "clustering/num_clusters": len(client_clusters)
    }
    
    for cluster_id, clients in client_clusters.items():
        cluster_stats[f"clustering/cluster_{cluster_id}/size"] = len(clients)
        wandb.log(cluster_stats)
        
        # 记录每个聚类的客户端tier分布
        tier_distribution = {}
        for c_id in clients:
            tier = client_manager.get_client(c_id).tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        print(f"聚类 {cluster_id}: {len(clients)} 个客户端 - {clients}")
        print(f"聚类 {cluster_id} Tier分布: {tier_distribution}")
        
        # 记录到wandb
        for tier, count in tier_distribution.items():
            wandb.log({f"clustering/cluster_{cluster_id}/tier_{tier}_count": count})
    
    return client_clusters, pretrained_models


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logging(args)
    logger.info("初始化统一服务器架构的异构联邦学习框架...")
    
    # 设置默认设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"默认设备: {device}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args)
    
    # 获取类别数
    if args.dataset != "cinic10":
        class_num = dataset[7]
    else:
        class_num = dataset[7]
    
    # 记录数据集统计信息
    if args.dataset != "cinic10":
        train_data_num, test_data_num = dataset[0], dataset[1]
    else:
        train_data_num, test_data_num = dataset[0], dataset[1]
    
    dataset_stats = {
        "dataset/name": args.dataset,
        "dataset/total_train_samples": train_data_num,
        "dataset/total_test_samples": test_data_num,
        "dataset/num_classes": class_num
    }
    
    if args.dataset != "cinic10":
        train_data_local_num_dict = dataset[4]
    else:
        train_data_local_num_dict = dataset[4]
    
    # 记录每个客户端的样本数量
    for client_idx, num_samples in train_data_local_num_dict.items():
        dataset_stats[f"dataset/client_{client_idx}_samples"] = num_samples
    
    wandb.log(dataset_stats)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配异构资源...")
    client_resources = allocate_device_resources(args.client_number)
    
    # 记录客户端资源分配
    resource_stats = {}
    for client_idx, resources in client_resources.items():
        resource_stats[f"client_{client_idx}/tier"] = resources["storage_tier"]
        resource_stats[f"client_{client_idx}/storage_gb"] = resources["storage_capacity"]
        resource_stats[f"client_{client_idx}/network_speed"] = resources["network_speed"]
        resource_stats[f"client_{client_idx}/compute_power"] = resources["compute_power"]
    
    wandb.log(resource_stats)
    
    # 设置客户端
    logger.info("初始化客户端...")
    client_manager, class_num = setup_clients(args, dataset, client_resources)
    
    # 设置模型
    logger.info(f"创建 {args.model} 模型架构...")
    client_models, unified_server_model, global_classifier = setup_models(args, class_num)
    
    # 创建特征监控器
    logger.info("初始化特征监控系统...")
    feature_monitor = FeatureMonitor()
    
    # 创建模型聚合器
    logger.info("初始化模型聚合器...")
    model_aggregator = UnifiedModelAggregator(device=device)
    
    # 创建评估数据集
    eval_dataset = get_cifar10_proxy_dataset(
        option='balanced_test',
        num_samples=2000,
        seed=100
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, shuffle=False
    )
    
    # 预训练和聚类
    logger.info("开始客户端预训练和聚类...")
    client_clusters, pretrained_models = pretrain_and_cluster(
        client_manager, client_models, args, device
    )
    
    # 为每个客户端创建对应tier的模型
    client_tier_models = {}
    for client_id in range(args.client_number):
        client = client_manager.get_client(client_id)
        if client:
            tier = client.tier
            # 使用预训练模型或创建新模型
            if client_id in pretrained_models:
                client_tier_models[client_id] = pretrained_models[client_id]
            else:
                client_tier_models[client_id] = copy.deepcopy(client_models[tier])
    
    # 创建并行训练器
    logger.info("创建并行训练框架...")
    parallel_trainer = UnifiedParallelTrainer(
        client_manager=client_manager,
        unified_server_model=unified_server_model,
        global_classifier=global_classifier,
        device=device
    )
    
    # 注册客户端模型
    parallel_trainer.register_client_models(client_tier_models)
    
    # 设置训练环境
    parallel_trainer.setup_training(
        cluster_map=client_clusters,
        max_workers=args.max_workers
    )
    
    # 定义更多的度量指标
    wandb.define_metric("round")
    wandb.define_metric("global/test_accuracy", step_metric="round")
    wandb.define_metric("global/test_loss", step_metric="round")
    wandb.define_metric("global/class_balance", step_metric="round")
    
    wandb.define_metric("client/local_accuracy", step_metric="round")
    wandb.define_metric("client/global_accuracy", step_metric="round")
    wandb.define_metric("client/feature_norm_ratio", step_metric="round")
    
    wandb.define_metric("anomalies/count", step_metric="round")
    wandb.define_metric("anomalies/feature_norm_mismatch", step_metric="round")
    wandb.define_metric("anomalies/class_accuracy_imbalance", step_metric="round")
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    
    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 执行并行训练
        train_results, eval_results, server_models, classifiers, training_time = parallel_trainer.execute_parallel_training(
            train_fn=train_client_with_unified_server,
            eval_fn=evaluate_client_with_unified_server,
            round_idx=round_idx,
            feature_monitor=feature_monitor
        )
        
        # 更新全局模型
        logger.info("更新全局模型...")
        parallel_trainer.update_global_models(
            server_models=server_models,
            classifiers=classifiers,
            aggregator=model_aggregator
        )
        
        # 全局模型评估
        logger.info("评估全局模型性能...")
        global_correct = 0
        global_total = 0
        global_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        # 记录每个类别的准确率
        class_correct = [0] * class_num
        class_total = [0] * class_num
        
        # 全局评估使用统一服务器模型和全局分类器
        unified_server_model.eval()
        global_classifier.eval()
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(device), target.to(device)
                
                # 前向传播 - 模拟tier=1的客户端(特征最丰富)
                # 1. 提取特征 - 使用基础模型(conv1+gn1+relu)
                features = server_features = None
                
                # 创建临时tier=1客户端模型进行特征提取
                temp_model = client_models[1].to(device)
                temp_model.eval()
                _, features = temp_model(data)
                
                # 2. 服务器处理
                server_features = unified_server_model(features, tier=1)
                
                # 3. 全局分类
                output = global_classifier(server_features)
                
                # 计算损失和准确率
                loss = criterion(output, target)
                global_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                global_total += target.size(0)
                global_correct += (predicted == target).sum().item()
                
                # 记录每个类别的准确率
                for i in range(len(target)):
                    label = target[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        # 计算平均损失和准确率
        global_loss /= len(eval_loader)
        global_accuracy = 100.0 * global_correct / global_total if global_total > 0 else 0
        
        # 计算每个类别的准确率
        per_class_acc = []
        for i in range(class_num):
            if class_total[i] > 0:
                per_class_acc.append(100.0 * class_correct[i] / class_total[i])
            else:
                per_class_acc.append(0.0)
        
        # 计算类别平衡度 - 最高准确率与最低准确率的比值
        if min(per_class_acc) > 0:
            class_balance = max(per_class_acc) / min(per_class_acc)
        else:
            class_balance = float('inf')
        
        # 更新特征监控器的全局统计信息
        feature_monitor.update_global_stats(global_accuracy, round_idx)
        
        # 更新最佳准确率
        is_best = global_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_accuracy
            # 保存最佳模型
            best_model_path = f"{args.running_name}_best_model.pth"
            torch.save({
                'server_model': unified_server_model.state_dict(),
                'global_classifier': global_classifier.state_dict(),
                'round': round_idx,
                'accuracy': best_accuracy
            }, best_model_path)
            logger.info(f"保存最佳模型到 {best_model_path}")
        
        # 计算客户端统计信息
        client_local_acc = []
        client_global_acc = []
        client_feature_norm_ratios = []
        
        for client_id, result in train_results.items():
            if isinstance(result, dict):
                # 收集本地和全局准确率
                if 'local_accuracy' in result:
                    client_local_acc.append(result['local_accuracy'])
                if 'global_accuracy' in result:
                    client_global_acc.append(result['global_accuracy'])
                
                # 收集特征范数比例
                if 'feature_stats' in result and 'monitoring' in result['feature_stats']:
                    monitoring = result['feature_stats']['monitoring']
                    if 'client_feature_norm' in monitoring and 'server_feature_norm' in monitoring:
                        client_norms = monitoring['client_feature_norm']
                        server_norms = monitoring['server_feature_norm']
                        
                        # 计算平均范数比例
                        if client_norms and server_norms:
                            min_len = min(len(client_norms), len(server_norms))
                            if min_len > 0:
                                ratios = []
                                for i in range(min_len):
                                    if server_norms[i] > 0:
                                        ratios.append(client_norms[i] / server_norms[i])
                                
                                if ratios:
                                    avg_ratio = sum(ratios) / len(ratios)
                                    client_feature_norm_ratios.append(avg_ratio)
        
        # 计算平均值
        avg_local_acc = sum(client_local_acc) / len(client_local_acc) if client_local_acc else 0
        avg_global_acc = sum(client_global_acc) / len(client_global_acc) if client_global_acc else 0
        avg_feature_norm_ratio = sum(client_feature_norm_ratios) / len(client_feature_norm_ratios) if client_feature_norm_ratios else 0
        
        # 获取异常统计信息
        anomalies = feature_monitor.get_latest_anomalies()
        anomalies_count = len([a for a in anomalies if a['round'] == round_idx])
        
        # 异常类型统计
        feature_norm_mismatch = len([a for a in anomalies if a['round'] == round_idx and a['type'] == 'feature_norm_mismatch'])
        class_accuracy_imbalance = len([a for a in anomalies if a['round'] == round_idx and a['type'] == 'class_accuracy_imbalance'])
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"全局准确率: {global_accuracy:.2f}%, 最佳: {best_accuracy:.2f}%")
        logger.info(f"平均本地准确率: {avg_local_acc:.2f}%, 平均全局准确率: {avg_global_acc:.2f}%")
        logger.info(f"类别平衡度: {class_balance:.2f}")
        logger.info(f"特征范数比例: {avg_feature_norm_ratio:.2f}")
        logger.info(f"异常数量: {anomalies_count}")
        logger.info(f"耗时: {round_time:.2f}秒")
        
        # 记录到wandb
        metrics = {
            "round": round_idx + 1,
            
            # 全局模型性能
            "global/test_accuracy": global_accuracy,
            "global/test_loss": global_loss,
            "global/best_accuracy": best_accuracy,
            "global/is_best_model": 1 if is_best else 0,
            "global/class_balance": class_balance,
            
            # 客户端性能
            "client/local_accuracy": avg_local_acc,
            "client/global_accuracy": avg_global_acc,
            "client/feature_norm_ratio": avg_feature_norm_ratio,
            
            # 异常统计
            "anomalies/count": anomalies_count,
            "anomalies/feature_norm_mismatch": feature_norm_mismatch,
            "anomalies/class_accuracy_imbalance": class_accuracy_imbalance,
            
            # 时间统计
            "time/round_seconds": round_time,
            "time/training_seconds": training_time
        }
        
        # 记录每个类别的准确率
        for i, acc in enumerate(per_class_acc):
            metrics[f"global/class_{i}_accuracy"] = acc
        
        wandb.log(metrics)
    
    # 训练完成，记录最终统计信息和分析报告
    analysis_report = feature_monitor.get_analysis_report()
    
    final_stats = {
        "final/best_accuracy": best_accuracy,
        "final/rounds": args.rounds,
        "final/client_number": args.client_number,
        "final/clusters": len(client_clusters),
        "final/total_anomalies": analysis_report['total_anomalies']
    }
    wandb.log(final_stats)
    
    logger.info(f"统一服务器架构的异构联邦学习训练完成! 最佳准确率: {best_accuracy:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()
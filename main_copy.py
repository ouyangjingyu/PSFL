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


# 主函数中导入新模块
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
import math
from collections import defaultdict
import pynvml

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入模型架构 - 已修改为使用GroupNorm
from model.resnet import resnet56_SFL_local_tier_7
from model.resnet import resnet110_SFL_local_tier_7
from model.resnet import resnet110_SFL_fedavg_base

# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from data.cifar10_eval_dataset import get_cifar10_proxy_dataset

# 导入自定义模块 - 包括新增的设备管理器和优化训练策略
from utils.enhanced_model_architecture import create_enhanced_client_model, create_enhanced_server_model, UnifiedClassifier
from utils.client_clustering import adaptive_cluster_assignment, extract_model_predictions
from utils.model_diagnosis_repair import ModelDiagnosticTracker, comprehensive_model_repair
from utils.enhanced_aggregation import EnhancedAggregator, enhanced_hierarchical_aggregation_no_projection
from utils.parallel_training_framework import create_training_framework, create_training_framework_with_global_classifier
from utils.unified_client_module import EnhancedClient, ClientManager, train_client_with_global_classifier
from utils.global_model_utils import create_models_by_splitting, create_global_model, combine_to_global_model, split_global_model
from utils.evaluate import comprehensive_evaluation, print_evaluation_results
from utils.device_manager import DeviceManager
from utils.training_strategy import OptimizedTrainingStrategy, create_training_strategy_for_client

# 设置随机种子，确保实验可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='异构联邦学习框架')
    
    # 实验标识
    parser.add_argument('--running_name', default="HPFL-GN", type=str, help='实验名称')
    
    # 优化相关参数 - 修改默认参数以适应GroupNorm
    parser.add_argument('--lr', default=0.00075, type=float, help='学习率 (默认为0.001的0.75倍，适应GroupNorm)')
    parser.add_argument('--lr_factor', default=0.85, type=float, help='学习率衰减因子')
    parser.add_argument('--lr_patience', default=10, type=float, help='学习率调整耐心值')
    parser.add_argument('--lr_min', default=1e-6, type=float, help='最小学习率')
    parser.add_argument('--optimizer', default="Adam", type=str, help='优化器: SGD, Adam等 (默认改为Adam)')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数 - 添加GroupNorm相关参数
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
    parser.add_argument('--whether_local_loss', default=True, type=bool, help='是否使用本地损失')
    parser.add_argument('--tier', default=5, type=int, help='默认tier级别')
    
    # 集群和并行训练参数
    parser.add_argument('--n_clusters', default=3, type=int, help='客户端聚类数量')
    parser.add_argument('--max_workers', default=None, type=int, help='最大并行工作线程数')
    
    # 网络模拟参数
    parser.add_argument('--net_speed_list', type=str, default=[100, 30, 30, 30, 10], 
                        help='网络速度列表(MB)')
    parser.add_argument('--delay_coefficient_list', type=str, default=[16, 20, 34, 130, 250],
                        help='延迟系数列表')
    
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
    logger = logging.getLogger("HPFL-GN")
    
    # 配置wandb
    wandb.init(
        mode="online",
        project="HeterogeneousFL-GN",  # 更新项目名称以反映使用GroupNorm
        name=args.running_name,
        config=args,
        tags=[f"model_{args.model}", f"dataset_{args.dataset}", f"clients_{args.client_number}", "GroupNorm"]
    )
    
    return logger

def safe_load_state_dict(model, state_dict, verbose=False):
    """安全地加载状态字典，处理架构不匹配的情况"""
    # 过滤状态字典，只保留模型中存在的键
    model_keys = set(model.state_dict().keys())
    dict_keys = set(state_dict.keys())
    
    # 找出不匹配的键
    missing_keys = model_keys - dict_keys
    unexpected_keys = dict_keys - model_keys
    
    if verbose:
        if missing_keys:
            print(f"模型中存在但状态字典中缺少的键: {missing_keys}")
        if unexpected_keys:
            print(f"状态字典中存在但模型中缺少的键: {unexpected_keys}")
    
    # 创建过滤后的状态字典
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    # 加载过滤后的状态字典
    model.load_state_dict(filtered_dict, strict=False)
    
    return model

def log_system_resources():
    """记录系统资源使用情况"""
    import psutil
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_used_gb = memory_info.used / (1024 ** 3)
    memory_percent = memory_info.percent
    
    resource_metrics = {
        "system/cpu_percent": cpu_percent,
        "system/memory_used_gb": memory_used_gb,
        "system/memory_percent": memory_percent
    }
    
    # 如果有GPU，记录GPU使用情况
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        resource_metrics.update({
            "system/gpu_mem_alloc_gb": gpu_mem_alloc,
            "system/gpu_mem_reserved_gb": gpu_mem_reserved
        })
        
        # 尝试获取GPU利用率（需要pynvml支持）
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            resource_metrics["system/gpu_utilization"] = gpu_utilization
        except:
            pass
    
    # 记录到wandb
    wandb.log(resource_metrics)
    
    return resource_metrics


def allocate_resources_to_clients(num_clients):
    """分配存储和设备资源模拟异构环境"""
    num_storage_tiers = 7
    client_resource_allocation = {}

    # 定义每个tier的概率（tier 3-5概率更高）
    storage_tier_weights = [0.05, 0.05, 0.25, 0.25, 0.25, 0.1, 0.05]  # tier 1-7的概率

    # 为每个客户端分配存储tier和资源
    for client_id in range(num_clients):
        # 随机选择存储tier
        storage_tier = random.choices(range(1, num_storage_tiers + 1), weights=storage_tier_weights, k=1)[0]
        
        # 分配随机存储容量
        storage_capacity = random.choice([64, 128, 256, 512, 1024])  # 存储容量（GB）
        
        # 分配网络速度 (MB/s)
        network_speed = random.choice([10, 30, 50, 100, 200])
        
        # 分配计算能力 (相对值)
        if storage_tier <= 2:  # 高级设备
            compute_power = random.uniform(0.8, 1.0)
        elif storage_tier <= 5:  # 中级设备
            compute_power = random.uniform(0.4, 0.8)
        else:  # 低级设备
            compute_power = random.uniform(0.1, 0.4)
            
        # 创建资源字典
        client_resource_allocation[client_id] = {
            "storage_tier": storage_tier,
            "storage_capacity": storage_capacity,
            "network_speed": network_speed,
            "compute_power": compute_power
        }
    
    return client_resource_allocation

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

def setup_clients(args, dataset, client_resources, client_models, server_models):
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
    
    # 为每个客户端分配模型和数据
    for client_idx in range(args.client_number):
        # 获取客户端资源信息
        resources = client_resources[client_idx]
        tier = resources["storage_tier"]
        
        # 创建优化的学习率，适应GroupNorm
        optimized_lr = args.lr * (1.0 if tier > 3 else 0.8)  # 高tier使用稍低的学习率
        
        # 添加客户端
        client_manager.add_client(
            client_id=client_idx,
            tier=tier,
            train_dataset=train_data_local_dict[client_idx],
            test_dataset=test_data_local_dict[client_idx],
            device=default_device,
            lr=optimized_lr,  # 使用优化的学习率
            local_epochs=args.client_epoch,
            resources=resources
        )
    
    # 创建客户端模型和服务器模型字典
    client_models_dict = {}
    server_models_dict = {}
    
    for client_idx in range(args.client_number):
        # 获取客户端tier
        tier = client_resources[client_idx]["storage_tier"]
        
        # 分配相应tier的模型
        client_models_dict[client_idx] = copy.deepcopy(client_models[tier])
        server_models_dict[client_idx] = copy.deepcopy(server_models[tier])
    
    return client_manager, client_models_dict, server_models_dict

def pretrain_and_cluster(client_manager, client_models_dict, args, device):
    """客户端预训练和数据分布感知聚类"""
    print("开始客户端预训练...")
    
    # 创建评估数据集
    eval_dataset = get_cifar10_proxy_dataset(
        option='balanced_test',
        num_samples=1000,
        seed=42
    )
    
    # 执行预训练
    client_ids = list(range(args.client_number))
    start_time = time.time()
    pretrain_results = client_manager.pre_train_clients(client_ids, client_models_dict, epochs=5)
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
        model = copy.deepcopy(client_models_dict[client_id])
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
        
        # 记录到wandb（每个tier的计数）
        for tier, count in tier_distribution.items():
            wandb.log({f"clustering/cluster_{cluster_id}/tier_{tier}_count": count})
    
    return client_clusters, pretrained_models

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging(args)
    logger.info("初始化GroupNorm版异构联邦学习框架...")
    
    # 创建设备管理器
    device_manager = DeviceManager()
    
    # 选择最佳聚合设备
    aggregation_device = device_manager.select_aggregation_device()
    logger.info(f"聚合过程使用设备: {aggregation_device}")
    
    # 设置默认设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"默认设备: {device}")
    
    # 记录初始系统资源
    resource_info = log_system_resources()
    logger.info(f"系统资源: CPU利用率 {resource_info['system/cpu_percent']}%, 内存使用 {resource_info['system/memory_used_gb']:.2f}GB ({resource_info['system/memory_percent']}%)")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args)
    if args.dataset != "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, _ = dataset
    
    # 记录数据集统计信息
    dataset_stats = {
        "dataset/name": args.dataset,
        "dataset/total_train_samples": train_data_num,
        "dataset/total_test_samples": test_data_num,
        "dataset/num_classes": class_num
    }
    
    # 记录每个客户端的样本数量
    for client_idx, num_samples in train_data_local_num_dict.items():
        dataset_stats[f"dataset/client_{client_idx}_samples"] = num_samples
    
    wandb.log(dataset_stats)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配异构资源...")
    client_resources = allocate_resources_to_clients(args.client_number)
    
    # 记录客户端资源分配
    resource_stats = {}
    for client_idx, resources in client_resources.items():
        resource_stats[f"client_{client_idx}/tier"] = resources["storage_tier"]
        resource_stats[f"client_{client_idx}/storage_gb"] = resources["storage_capacity"]
        resource_stats[f"client_{client_idx}/network_speed"] = resources["network_speed"]
        resource_stats[f"client_{client_idx}/compute_power"] = resources["compute_power"]
    
    wandb.log(resource_stats)
    
    # 创建模型 - 使用GroupNorm版本
    logger.info(f"创建GroupNorm版 {args.model} 模型架构...")
    # 传入groups_per_channel参数
    client_models, server_models, unified_classifier, init_glob_model, num_tiers = create_models(
        args, class_num, groups_per_channel=args.groups_per_channel
    )
    
    # 设置客户端
    logger.info("初始化客户端...")
    client_manager, client_models_dict, server_models_dict = setup_clients(
        args, dataset, client_resources, client_models, server_models
    )
    
    # 创建聚合器
    logger.info("初始化增强聚合器...")
    aggregator = EnhancedAggregator(device_manager)
    
    # 创建诊断追踪器
    logger.info("初始化模型诊断系统...")
    diagnostic_tracker = ModelDiagnosticTracker()
    
    # 创建评估数据集（用于诊断和评估）
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
        client_manager, client_models_dict, args, device
    )
    
    # 更新客户端模型（使用预训练结果）
    for client_id, model in pretrained_models.items():
        client_models_dict[client_id] = model
    
    # 创建训练框架
    logger.info("创建并行训练框架...")
    training_framework = create_training_framework_with_global_classifier(
        client_models_dict, server_models_dict, client_resources
    )
        
    # 设置共享分类器
    training_framework.set_shared_classifier(unified_classifier)

    # 定义更多的度量指标
    wandb.define_metric("round")
    wandb.define_metric("global/test_accuracy", step_metric="round")
    wandb.define_metric("global/test_loss", step_metric="round")

    # 添加本地分类器指标
    wandb.define_metric("client/local_train_accuracy", step_metric="round")
    wandb.define_metric("client/local_train_loss", step_metric="round")
    wandb.define_metric("client/local_test_accuracy", step_metric="round")
    wandb.define_metric("client/local_test_loss", step_metric="round")

    # 添加全局分类器指标
    wandb.define_metric("client/global_train_accuracy", step_metric="round")
    wandb.define_metric("client/global_train_loss", step_metric="round")
    wandb.define_metric("client/global_test_accuracy", step_metric="round")
    wandb.define_metric("client/global_test_loss", step_metric="round")

    # 原有指标（兼容旧代码）
    wandb.define_metric("server/avg_train_accuracy", step_metric="round")
    wandb.define_metric("server/avg_test_accuracy", step_metric="round")
    wandb.define_metric("server/training_time", step_metric="round")

    # 为每个聚类创建表格
    for cluster_id in client_clusters.keys():
        wandb.define_metric(f"cluster_{cluster_id}/local_train_accuracy", step_metric="round")
        wandb.define_metric(f"cluster_{cluster_id}/local_test_accuracy", step_metric="round")
        wandb.define_metric(f"cluster_{cluster_id}/global_train_accuracy", step_metric="round")
        wandb.define_metric(f"cluster_{cluster_id}/global_test_accuracy", step_metric="round")
        # 兼容原有指标
        wandb.define_metric(f"cluster_{cluster_id}/avg_train_accuracy", step_metric="round")
        wandb.define_metric(f"cluster_{cluster_id}/avg_test_accuracy", step_metric="round")
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 设置训练环境
        training_framework.setup_training(
            client_clusters, 
            max_workers=args.max_workers
        )
        
        # 设置共享分类器
        training_framework.set_shared_classifier(unified_classifier)

        # 执行并行训练
        cluster_results, client_stats, training_time = training_framework.execute_training(
            train_client_with_global_classifier,  # 使用优化的训练函数
            evaluate_client_function,
            client_manager=client_manager,
            round_idx=round_idx,
            coordinator=training_framework,  # 传入协调器
            local_epochs=args.client_epoch,
            split_rounds=1
        )
        
        # 执行分层聚合前获取客户端tier信息
        client_tiers = {}
        for client_id in client_models_dict.keys():
            client = client_manager.get_client(client_id)
            if client is not None:
                client_tiers[client_id] = client.tier

        # 收集训练后的模型
        client_models_params, client_weights = training_framework.collect_trained_models(cluster_results)
        
        # 执行增强的分层聚合
        logger.info("执行基于设备管理的双层聚合...")
        try:
            # 使用设备管理器进行聚合
            global_model, cluster_models, agg_log = aggregator.enhanced_hierarchical_aggregation(
                client_models_params, 
                client_weights, 
                client_clusters,
                client_tiers=client_tiers,
                global_model_template=init_glob_model.state_dict() if init_glob_model else None,
                num_classes=class_num
            )
            
        except Exception as e:
            import traceback
            print(f"聚合过程出错: {str(e)}")
            print(traceback.format_exc())
            global_model = {} if init_glob_model is None else init_glob_model.state_dict()
            cluster_models = {}
            agg_log = {'error': str(e)}
                
        # 更新全局模型和聚类模型
        training_framework.set_global_model(global_model)
        training_framework.set_cluster_models(cluster_models)
        
        # 创建全局模型进行评估
        global_eval_model = copy.deepcopy(init_glob_model)
        global_eval_model = global_eval_model.to(device)

        # 打印全局模型状态字典键
        print("\n==== 全局聚合模型状态字典信息 ====")
        global_model_keys = set(global_model.keys())
        print(f"键的数量: {len(global_model_keys)}")
        print(f"示例键: {list(global_model_keys)[:5]}")
        
        # 加载聚合后的模型参数 - 使用设备管理器恢复到原始设备
        global_eval_model = safe_load_state_dict(global_eval_model, global_model, verbose=True)

        # 进行全面评估
        logger.info("执行全面模型评估...")
        eval_results = comprehensive_evaluation(
            global_eval_model,  # 全局模型
            client_models_dict,  # 客户端模型 
            server_models_dict,  # 服务器模型
            unified_classifier,  # 全局分类器
            eval_dataset,
            client_clusters,
            client_resources,  # 传入客户端资源信息（包含tier级别）
            device,
            class_num
        )

        # 打印并记录结果
        print_evaluation_results(eval_results, round_idx)

        # 记录到wandb
        comp_eval_metrics = {
            f"comprehensive/global_accuracy": eval_results['global']['accuracy'],
            f"comprehensive/avg_local_accuracy": eval_results['averages']['local_accuracy'],
            f"comprehensive/avg_server_global_accuracy": eval_results['averages']['server_global_accuracy']
        }

        # 添加客户端-全局评估结果
        if isinstance(eval_results['averages']['client_global_accuracy'], (int, float)):
            comp_eval_metrics[f"comprehensive/avg_client_global_accuracy"] = eval_results['averages']['client_global_accuracy']

        # 记录每个聚类的评估结果
        for key, value in eval_results.items():
            if key.startswith('cluster_'):
                cluster_id = key.split('_')[1]
                comp_eval_metrics[f"comprehensive/cluster_{cluster_id}/local_accuracy"] = value['local_accuracy']
                comp_eval_metrics[f"comprehensive/cluster_{cluster_id}/server_global_accuracy"] = value['server_global_accuracy']
                if isinstance(value['client_global_accuracy'], (int, float)):
                    comp_eval_metrics[f"comprehensive/cluster_{cluster_id}/client_global_accuracy"] = value['client_global_accuracy']

        wandb.log(comp_eval_metrics)
        
        # 全局模型评估
        logger.info("评估全局模型性能...")
        global_eval_model.eval()
        correct = 0
        total = 0
        test_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        # 记录每个类别的准确率
        class_correct = [0] * class_num
        class_total = [0] * class_num
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(device), target.to(device)
                output = global_eval_model(data)
                
                # 处理可能的元组输出
                if isinstance(output, tuple):
                    output = output[0]
                
                # 计算损失和准确率
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 记录每个类别的准确率
                for i in range(len(target)):
                    label = target[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        # 计算平均损失和准确率
        test_loss /= len(eval_loader)
        accuracy = 100.0 * correct / total
        
        # 计算每个类别的准确率
        per_class_acc = [100.0 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
        
        # 更新最佳准确率
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            # 保存最佳模型
            torch.save(global_eval_model.state_dict(), f"{args.running_name}_best_model.pth")

        # 计算训练统计信息
        # 1. 收集所有客户端的训练/测试数据
        all_client_local_train_acc = []     # 本地分类器训练准确率
        all_client_local_train_loss = []    # 本地分类器训练损失
        all_client_global_train_acc = []    # 全局分类器训练准确率
        all_client_global_train_loss = []   # 全局分类器训练损失

        all_client_local_test_acc = []      # 本地分类器测试准确率
        all_client_local_test_loss = []     # 本地分类器测试损失
        all_client_global_test_acc = []     # 全局分类器测试准确率
        all_client_global_test_loss = []    # 全局分类器测试损失

        # 兼容原有代码
        all_client_train_acc = []
        all_client_train_loss = []
        all_client_test_acc = []
        all_client_test_loss = []

        # 2. 计算各聚类内客户端的平均准确率和损失
        cluster_train_metrics = defaultdict(dict)
        cluster_test_metrics = defaultdict(dict)

        # 3. 收集每个客户端的指标
        for cluster_id, clients in client_clusters.items():
            # 初始化聚类指标列表
            cluster_local_train_acc = []
            cluster_local_train_loss = []
            cluster_global_train_acc = []
            cluster_global_train_loss = []
            
            cluster_local_test_acc = []
            cluster_local_test_loss = []
            cluster_global_test_acc = []
            cluster_global_test_loss = []
            
            # 兼容原有代码
            cluster_clients_train_acc = []
            cluster_clients_train_loss = []
            cluster_clients_test_acc = []
            cluster_clients_test_loss = []
            
            for client_id in clients:
                # 获取训练指标
                if client_id in client_stats:
                    client_metrics = client_stats.get(client_id, {})
                    
                    # 本地分类器训练指标
                    if 'local_train_accuracy' in client_metrics:
                        local_train_acc = client_metrics['local_train_accuracy']
                        all_client_local_train_acc.append(local_train_acc)
                        cluster_local_train_acc.append(local_train_acc)
                    
                    if 'local_train_loss' in client_metrics:
                        local_train_loss = client_metrics['local_train_loss']
                        all_client_local_train_loss.append(local_train_loss)
                        cluster_local_train_loss.append(local_train_loss)
                    
                    # 本地分类器拆分学习指标
                    if 'local_sl_accuracy' in client_metrics:
                        local_sl_acc = client_metrics['local_sl_accuracy']
                        # 这里不添加到全局列表，因为是拆分学习过程中的指标
                    
                    # 全局分类器训练指标（拆分学习）
                    if 'global_sl_accuracy' in client_metrics:
                        global_train_acc = client_metrics['global_sl_accuracy']
                        all_client_global_train_acc.append(global_train_acc)
                        cluster_global_train_acc.append(global_train_acc)
                    
                    if 'global_sl_loss' in client_metrics:
                        global_train_loss = client_metrics['global_sl_loss']
                        all_client_global_train_loss.append(global_train_loss)
                        cluster_global_train_loss.append(global_train_loss)
                    
                    # 兼容原有代码
                    if 'accuracy' in client_metrics:
                        train_acc = client_metrics['accuracy']
                        all_client_train_acc.append(train_acc)
                        cluster_clients_train_acc.append(train_acc)
                    
                    if 'loss' in client_metrics:
                        train_loss = client_metrics['loss']
                        all_client_train_loss.append(train_loss)
                        cluster_clients_train_loss.append(train_loss)
                
                # 获取测试指标
                for result in cluster_results.values():
                    if 'evaluation_metrics' in result and client_id in result['evaluation_metrics']:
                        test_metrics = result['evaluation_metrics'][client_id]
                        
                        # 本地分类器测试指标
                        if 'local_accuracy' in test_metrics:
                            local_test_acc = test_metrics['local_accuracy']
                            all_client_local_test_acc.append(local_test_acc)
                            cluster_local_test_acc.append(local_test_acc)
                        
                        if 'local_loss' in test_metrics:
                            local_test_loss = test_metrics['local_loss']
                            all_client_local_test_loss.append(local_test_loss)
                            cluster_local_test_loss.append(local_test_loss)
                        
                        # 全局分类器测试指标
                        if 'global_accuracy' in test_metrics:
                            global_test_acc = test_metrics['global_accuracy']
                            all_client_global_test_acc.append(global_test_acc)
                            cluster_global_test_acc.append(global_test_acc)
                        
                        if 'global_loss' in test_metrics:
                            global_test_loss = test_metrics['global_loss']
                            all_client_global_test_loss.append(global_test_loss)
                            cluster_global_test_loss.append(global_test_loss)
                        
                        # 兼容原有代码
                        if 'accuracy' in test_metrics:
                            test_acc = test_metrics['accuracy']
                            all_client_test_acc.append(test_acc)
                            cluster_clients_test_acc.append(test_acc)
                        
                        if 'loss' in test_metrics:
                            test_loss = test_metrics['loss']
                            all_client_test_loss.append(test_loss)
                            cluster_clients_test_loss.append(test_loss)
            
            # 计算聚类平均指标
            # 本地分类器
            if cluster_local_train_acc:
                cluster_train_metrics[cluster_id]['local_train_accuracy'] = np.mean(cluster_local_train_acc)
            if cluster_local_train_loss:
                cluster_train_metrics[cluster_id]['local_train_loss'] = np.mean(cluster_local_train_loss)
            if cluster_local_test_acc:
                cluster_test_metrics[cluster_id]['local_test_accuracy'] = np.mean(cluster_local_test_acc)
            if cluster_local_test_loss:
                cluster_test_metrics[cluster_id]['local_test_loss'] = np.mean(cluster_local_test_loss)
            
            # 全局分类器
            if cluster_global_train_acc:
                cluster_train_metrics[cluster_id]['global_train_accuracy'] = np.mean(cluster_global_train_acc)
            if cluster_global_train_loss:
                cluster_train_metrics[cluster_id]['global_train_loss'] = np.mean(cluster_global_train_loss)
            if cluster_global_test_acc:
                cluster_test_metrics[cluster_id]['global_test_accuracy'] = np.mean(cluster_global_test_acc)
            if cluster_global_test_loss:
                cluster_test_metrics[cluster_id]['global_test_loss'] = np.mean(cluster_global_test_loss)
            
            # 兼容原有代码
            if cluster_clients_train_acc:
                cluster_train_metrics[cluster_id]['avg_train_accuracy'] = np.mean(cluster_clients_train_acc)
            if cluster_clients_train_loss:
                cluster_train_metrics[cluster_id]['avg_train_loss'] = np.mean(cluster_clients_train_loss)
            if cluster_clients_test_acc:
                cluster_test_metrics[cluster_id]['avg_test_accuracy'] = np.mean(cluster_clients_test_acc)
            if cluster_clients_test_loss:
                cluster_test_metrics[cluster_id]['avg_test_loss'] = np.mean(cluster_clients_test_loss)

        # 计算全局平均指标
        # 本地分类器
        avg_local_train_acc = np.mean(all_client_local_train_acc) if all_client_local_train_acc else 0
        avg_local_train_loss = np.mean(all_client_local_train_loss) if all_client_local_train_loss else 0
        avg_local_test_acc = np.mean(all_client_local_test_acc) if all_client_local_test_acc else 0
        avg_local_test_loss = np.mean(all_client_local_test_loss) if all_client_local_test_loss else 0

        # 全局分类器
        avg_global_train_acc = np.mean(all_client_global_train_acc) if all_client_global_train_acc else 0
        avg_global_train_loss = np.mean(all_client_global_train_loss) if all_client_global_train_loss else 0
        avg_global_test_acc = np.mean(all_client_global_test_acc) if all_client_global_test_acc else 0
        avg_global_test_loss = np.mean(all_client_global_test_loss) if all_client_global_test_loss else 0

        # 兼容原有代码
        avg_train_acc = np.mean(all_client_train_acc) if all_client_train_acc else 0
        avg_train_loss = np.mean(all_client_train_loss) if all_client_train_loss else 0
        avg_test_acc = np.mean(all_client_test_acc) if all_client_test_acc else 0
        avg_test_loss = np.mean(all_client_test_loss) if all_client_test_loss else 0

        # 记录轮次时间
        round_time = time.time() - round_start_time

        # 记录结果
        logger.info(f"全局模型测试结果 - 准确率: {accuracy:.2f}%, 损失: {test_loss:.4f}")
        logger.info(f"客户端本地分类器平均训练指标 - 准确率: {avg_local_train_acc:.2f}%, 损失: {avg_local_train_loss:.4f}")
        logger.info(f"客户端本地分类器平均测试指标 - 准确率: {avg_local_test_acc:.2f}%, 损失: {avg_local_test_loss:.4f}")
        logger.info(f"全局分类器平均训练指标 - 准确率: {avg_global_train_acc:.2f}%, 损失: {avg_global_train_loss:.4f}")
        logger.info(f"全局分类器平均测试指标 - 准确率: {avg_global_test_acc:.2f}%, 损失: {avg_global_test_loss:.4f}")
        logger.info(f"轮次耗时: {round_time:.2f}秒")

        # 记录到wandb
        round_metrics = {
            "round": round_idx + 1,
            
            # 全局模型测试结果
            "global/test_accuracy": accuracy,
            "global/test_loss": test_loss,
            "global/best_accuracy": best_accuracy,
            "global/is_best_model": 1 if is_best else 0,
            
            # 客户端本地分类器平均指标
            "client/local_train_accuracy": avg_local_train_acc,
            "client/local_train_loss": avg_local_train_loss,
            "client/local_test_accuracy": avg_local_test_acc,
            "client/local_test_loss": avg_local_test_loss,
            
            # 全局分类器平均指标
            "client/global_train_accuracy": avg_global_train_acc,
            "client/global_train_loss": avg_global_train_loss,
            "client/global_test_accuracy": avg_global_test_acc,
            "client/global_test_loss": avg_global_test_loss,
            
            # 兼容原有代码
            "server/avg_train_accuracy": avg_train_acc,
            "server/avg_train_loss": avg_train_loss,
            "server/avg_test_accuracy": avg_test_acc,
            "server/avg_test_loss": avg_test_loss,
            
            "server/training_time": training_time,
            "time/round_seconds": round_time
        }

        # 记录每个类别的准确率
        for i, acc in enumerate(per_class_acc):
            round_metrics[f"global/class_{i}_accuracy"] = acc

        # 记录每个聚类的指标
        for cluster_id in client_clusters.keys():
            # 本地分类器
            if cluster_id in cluster_train_metrics and 'local_train_accuracy' in cluster_train_metrics[cluster_id]:
                round_metrics[f"cluster_{cluster_id}/local_train_accuracy"] = cluster_train_metrics[cluster_id]['local_train_accuracy']
            if cluster_id in cluster_test_metrics and 'local_test_accuracy' in cluster_test_metrics[cluster_id]:
                round_metrics[f"cluster_{cluster_id}/local_test_accuracy"] = cluster_test_metrics[cluster_id]['local_test_accuracy']
            
            # 全局分类器
            if cluster_id in cluster_train_metrics and 'global_train_accuracy' in cluster_train_metrics[cluster_id]:
                round_metrics[f"cluster_{cluster_id}/global_train_accuracy"] = cluster_train_metrics[cluster_id]['global_train_accuracy']
            if cluster_id in cluster_test_metrics and 'global_test_accuracy' in cluster_test_metrics[cluster_id]:
                round_metrics[f"cluster_{cluster_id}/global_test_accuracy"] = cluster_test_metrics[cluster_id]['global_test_accuracy']
            
            # 兼容原有代码
            if cluster_id in cluster_train_metrics and 'avg_train_accuracy' in cluster_train_metrics[cluster_id]:
                round_metrics[f"cluster_{cluster_id}/avg_train_accuracy"] = cluster_train_metrics[cluster_id]['avg_train_accuracy']
            if cluster_id in cluster_test_metrics and 'avg_test_accuracy' in cluster_test_metrics[cluster_id]:
                round_metrics[f"cluster_{cluster_id}/avg_test_accuracy"] = cluster_test_metrics[cluster_id]['avg_test_accuracy']

        wandb.log(round_metrics)
        
        # 定期记录系统资源使用情况
        if round_idx % 5 == 0 or round_idx == args.rounds - 1:
            log_system_resources()
        
        # 使用设备管理器和增强聚合器更新客户端模型
        print("更新客户端和服务器模型...")
        updated_client_models = aggregator.update_client_models(
            client_models_dict, 
            global_model=global_model, 
            cluster_models=cluster_models, 
            cluster_map=client_clusters
        )
        
        # 更新客户端模型
        client_models_dict = updated_client_models
        
        # 更新服务器模型，使用全局模型拆分
        for client_id in client_models_dict.keys():
            # 获取客户端tier
            client = client_manager.get_client(client_id)
            
            if client is None:
                continue
                
            tier = client.tier
            
            # 服务器模型更新 - 使用拆分全局模型的方式（确保服务器模型一致性）
            _, server_model = split_global_model(
                global_model,
                tier=tier,
                class_num=class_num,
                model_type=args.model,
                groups_per_channel=args.groups_per_channel  # 添加这个参数
            )
            
            # 更新服务器模型
            server_models_dict[client_id] = server_model
            
            print(f"已用全局模型更新客户端 {client_id} (Tier {tier}) 的服务器模型")
    
    # 训练完成，记录最终统计信息
    final_stats = {
        "final/best_accuracy": best_accuracy,
        "final/rounds": args.rounds,
        "final/client_number": args.client_number,
        "final/clusters": len(client_clusters),
        "final/model_type": "GroupNorm-ResNet"  # 标记使用了GroupNorm版本
    }
    wandb.log(final_stats)
    
    logger.info(f"GroupNorm版联邦学习训练完成! 最佳准确率: {best_accuracy:.2f}%")
    wandb.finish()


# 模型创建函数 - 传递GroupNorm参数
def create_models(args, class_num, groups_per_channel=32):
    """创建客户端和服务器模型"""
    print("创建GroupNorm版模型架构...")
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 通过拆分全局模型创建所有tier的客户端和服务器模型
    client_models, server_models, unified_classifier, init_glob_model, num_tiers = create_models_by_splitting(
        class_num=class_num,
        model_type=args.model,  # 'resnet110' 或 'resnet56'
        device=device,
        groups_per_channel=groups_per_channel  # 传递GroupNorm参数
    )
    
    # 打印模型架构信息
    print("\n===== GroupNorm模型架构信息 =====")
    print(f"模型类型: {args.model} (GroupNorm版)")
    print(f"Tier数量: {num_tiers}")
    print(f"每通道分组数: {groups_per_channel}")
    print(f"分类器输入维度: 256, 分类数: {class_num}")
    
    return client_models, server_models, unified_classifier, init_glob_model, num_tiers


if __name__ == "__main__":
    main()
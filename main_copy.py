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

# 导入自定义模块
from model.resnet import create_tierhfl_client_model, create_tierhfl_server_model
from utils.tierhfl_aggregator import StabilizedAggregator
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_trainer import ClusterAwareParallelTrainer, AdaptiveTrainingController, DataDistributionClusterer

# 导入数据加载和处理模块
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

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
    parser = argparse.ArgumentParser(description='TierHFL: 分层异构联邦学习框架')
    
    # 实验标识
    parser.add_argument('--running_name', default="TierHFL", type=str, help='实验名称')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.001, type=float, help='初始学习率')
    parser.add_argument('--lr_factor', default=0.85, type=float, help='学习率衰减因子')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet56', help='使用的神经网络 (resnet56 或 resnet110)')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='cifar10', help='训练数据集')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='本地工作节点上数据集的划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=1, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=10, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=100, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    parser.add_argument('--n_clusters', default=3, type=int, help='客户端聚类数量')
    parser.add_argument('--max_workers', default=None, type=int, help='最大并行工作线程数')
    
    # TierHFL特有参数
    parser.add_argument('--init_alpha', default=0.5, type=float, help='初始本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.1, type=float, help='初始特征对齐损失权重')
    parser.add_argument('--beta', default=0.8, type=float, help='聚合动量因子')
    
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
    
    # 增强wandb配置
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
            "tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed,
            "storage_capacity": storage_capacity
        }
    
    return resources

# 简化的数据分布聚类器
class SimpleDataDistributionClusterer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clustering_history = []  # 记录聚类历史
    
    def cluster_clients(self, client_models, client_ids, eval_dataset=None, device='cuda'):
        # 简单地根据client_id平均分配到各个聚类
        clusters = {}
        for i in range(self.num_clusters):
            clusters[i] = []
            
        # 平均分配
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

    def get_clustering_statistics(self):
        """获取聚类统计信息"""
        stats = []
        for i, cluster_record in enumerate(self.clustering_history):
            clusters = cluster_record['clusters']
            
            # 计算聚类大小分布
            cluster_sizes = [len(clients) for clients in clusters.values()]
            avg_size = sum(cluster_sizes) / max(1, len(cluster_sizes))
            min_size = min(cluster_sizes) if cluster_sizes else 0
            max_size = max(cluster_sizes) if cluster_sizes else 0
            
            stats.append({
                'iteration': i,
                'num_clusters': len(clusters),
                'num_clients': cluster_record['num_clients'],
                'avg_cluster_size': avg_size,
                'min_cluster_size': min_size,
                'max_cluster_size': max_size,
                'cluster_sizes': cluster_sizes
            })
        
        return stats

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

# 主函数
def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logging(args)
    logger.info("初始化TierHFL: 分层异构联邦学习框架...")
    
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
        
        # 设置初始alpha和lambda
        client.update_alpha(args.init_alpha)
        client.update_lambda_feature(args.init_lambda)
    
    # 创建客户端模型
    logger.info(f"创建 {args.model} 客户端模型...")
    client_models = {}
    
    # 使用简单的ResNet模型代替我们的自定义模型
    for client_id, resource in client_resources.items():
        tier = resource["tier"]
        # 简单起见，为每个客户端创建一个简单的ResNet模型
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, class_num)
        )
        
        # 为每个tier添加get_shared_params和get_personalized_params方法
        def get_shared_params(self):
            return {name: param for name, param in self.named_parameters() 
                   if not any(x in name for x in ['4', 'fc', 'linear', 'classifier'])}
            
        def get_personalized_params(self):
            return {name: param for name, param in self.named_parameters() 
                   if any(x in name for x in ['4', 'fc', 'linear', 'classifier'])}
            
        # 添加forward方法来返回logits和特征
        def forward(self, x):
            for i in range(3):  # 直到BatchNorm2d
                x = self[i](x)
            features = x
            for i in range(3, len(self)):
                x = self[i](x)
            return x, features
        
        # 动态添加方法
        model.get_shared_params = get_shared_params.__get__(model)
        model.get_personalized_params = get_personalized_params.__get__(model)
        model.forward = forward.__get__(model)
        
        client_models[client_id] = model
    
    # 创建服务器模型
    logger.info("创建服务器模型...")
    
    # 简单的服务器模型
    server_model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, class_num)
    )
    
    # 添加forward方法
    def server_forward(self, x, tier=None):
        # 更清晰的命名，反映维度变化
        pooled_features = self[0](x)  # (batch_size, channels, 1, 1)
        flattened_features = self[1](pooled_features)  # (batch_size, channels)
        logits = self[2](flattened_features)  # (batch_size, class_num)
        
        # 明确返回展平后的2D特征
        return logits, flattened_features
    
    server_model.forward = server_forward.__get__(server_model)
    server_model.num_classes = class_num
    
    # 创建稳定化聚合器
    logger.info("创建稳定化聚合器...")
    aggregator = StabilizedAggregator(beta=args.beta, device=device)
    
    # 创建客户端聚类器
    logger.info("创建数据分布聚类器...")
    clusterer = SimpleDataDistributionClusterer(num_clusters=args.n_clusters)
    
    # 创建自适应训练控制器
    logger.info("创建自适应训练控制器...")
    controller = AdaptiveTrainingController(
        initial_alpha=args.init_alpha,
        initial_lambda=args.init_lambda
    )
    
    # 对客户端进行初始聚类
    logger.info("执行初始客户端聚类...")
    client_ids = list(range(args.client_number))
    cluster_map = clusterer.cluster_clients(
        client_models=client_models,
        client_ids=client_ids
    )

    # 打印初始聚类信息
    print_cluster_info(cluster_map, client_resources, logger)
    
    # 创建并行训练器
    logger.info("创建并行训练器...")
    trainer = ClusterAwareParallelTrainer(
        client_manager=client_manager,
        server_model=server_model,
        device=device,
        max_workers=args.max_workers
    )
    
    # 注册客户端模型
    trainer.register_client_models(client_models)
    
    # 设置训练环境
    trainer.setup_training(cluster_map=cluster_map)
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0

    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    
    # 创建学习率追踪字典
    lr_tracker = {client_id: args.lr for client_id in range(args.client_number)}
    
    # 记录每个客户端的性能历史
    client_performance_history = {client_id: {"train_acc": [], "test_acc": [], "local_acc": [], "global_acc": []} 
                                for client_id in range(args.client_number)}
    
    # 记录时间开销历史
    time_history = {
        "round_time": [],
        "training_time": [],
        "communication_time": [],
        "aggregation_time": [],
        "client_times": {client_id: [] for client_id in range(args.client_number)}
    }
    
    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 执行并行训练
        train_results, eval_results, server_models, time_stats, training_time = trainer.execute_parallel_training(round_idx)
        
        # 防止空结果
        if not train_results or not eval_results:
            logger.error("训练或评估结果为空，跳过本轮")
            continue
        
        # 更新自适应控制器的历史记录
        controller.update_history(eval_results)

        # 记录每个客户端的学习率
        for client_id in range(args.client_number):
            client = client_manager.get_client(client_id)
            if client:
                lr_tracker[client_id] = client.lr
        
        # 调整训练参数
        if round_idx > 0 and round_idx % 5 == 0:  # 每5轮调整一次
            params = controller.adjust_parameters()
            logger.info(f"调整训练参数: alpha={params['alpha']:.3f}, lambda_feature={params['lambda_feature']:.3f}")
            
            # 更新所有客户端的参数
            client_manager.update_all_clients_alpha(params['alpha'])
            client_manager.update_all_clients_feature_lambda(params['lambda_feature'])
        
        # 计算全局聚合权重 - 基于客户端评估性能
        client_weights = {}
        for client_id, result in eval_results.items():
            if 'global_accuracy' in result:
                client_weights[client_id] = result['global_accuracy'] / 100.0  # 归一化到0-1
        
        # 聚合服务器模型
        logger.info("聚合服务器模型...")
        aggregation_start_time = time.time()
        client_states = {client_id: result['model_state'] for client_id, result in train_results.items()}
        
        # 提取客户端tier信息
        client_tiers = {client_id: client_resources[client_id]['tier'] for client_id in client_states.keys()}
        
        # 聚合模型
        aggregated_model = aggregator.aggregate(
            client_states=client_states,
            client_weights=client_weights,
            client_clusters=cluster_map
        )
        aggregation_time = time.time() - aggregation_start_time
        
        # 更新客户端模型的共享部分
        comm_start_time = time.time()
        for client_id, model in client_models.items():
            # 保存个性化参数
            personalized_params = {}
            for name, param in model.named_parameters():
                if any(x in name for x in ['4', 'fc', 'linear', 'classifier']):
                    personalized_params[name] = param.clone()
            
            # 加载聚合模型 - 防止错误
            try:
                model.load_state_dict(aggregated_model, strict=False)
            except Exception as e:
                logger.error(f"加载聚合模型失败: {str(e)}")
                # 跳过本次更新
                continue
            
            # 恢复个性化参数
            for name, param in model.named_parameters():
                if name in personalized_params:
                    param.data = personalized_params[name].data
        
        # 更新服务器模型
        try:
            server_model.load_state_dict(aggregated_model, strict=False)
        except Exception as e:
            logger.error(f"更新服务器模型失败: {str(e)}")
        
        communication_time = time.time() - comm_start_time


        # 计算全局指标
        total_samples = 0
        weighted_acc = 0.0
        weighted_loss = 0.0
        class_accs = [0.0] * class_num
        class_counts = [0] * class_num
        
        for client_id, result in eval_results.items():
            samples = train_data_local_num_dict[client_id]
            total_samples += samples
            
            # 加权准确率和损失
            weighted_acc += result['global_accuracy'] * samples
            weighted_loss += result['test_loss'] * samples
            
            # 统计每个类别的准确率
            if 'global_per_class_acc' in result and len(result['global_per_class_acc']) == class_num:
                for i, acc in enumerate(result['global_per_class_acc']):
                    class_accs[i] += acc * samples
                    class_counts[i] += samples
            
            # 更新客户端性能历史记录
            client_performance_history[client_id]['test_acc'].append(result.get('test_loss', 0))
            client_performance_history[client_id]['local_acc'].append(result.get('local_accuracy', 0))
            client_performance_history[client_id]['global_acc'].append(result.get('global_accuracy', 0))
        
        # 计算平均值
        if total_samples > 0:
            global_acc = weighted_acc / total_samples
            global_loss = weighted_loss / total_samples
            
            # 计算每个类别的平均准确率
            for i in range(class_num):
                if class_counts[i] > 0:
                    class_accs[i] /= class_counts[i]
            
            # 计算类别平衡度 - 最高准确率与最低准确率的比值
            non_zero_accs = [acc for acc in class_accs if acc > 0]
            if non_zero_accs:
                class_balance = max(non_zero_accs) / max(0.1, min(non_zero_accs))
            else:
                class_balance = float('inf')
        else:
            global_acc = 0.0
            global_loss = 0.0
            class_balance = float('inf')
        
        # 更新最佳准确率
        is_best = global_acc > best_accuracy
        if is_best:
            best_accuracy = global_acc
            # 保存最佳模型
            try:
                torch.save({
                    'server_model': server_model.state_dict(),
                    'round': round_idx,
                    'accuracy': best_accuracy
                }, f"{args.running_name}_best_model.pth")
                logger.info(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"保存模型失败: {str(e)}")
        
        # 计算客户端统计信息
        if eval_results:
            avg_local_acc = sum(result.get('local_accuracy', 0) for result in eval_results.values()) / max(1, len(eval_results))
            avg_global_acc = sum(result.get('global_accuracy', 0) for result in eval_results.values()) / max(1, len(eval_results))
            
            # 计算每个客户端精度分布统计
            client_acc_std = np.std([result.get('global_accuracy', 0) for result in eval_results.values()])
            client_acc_min = min([result.get('global_accuracy', 0) for result in eval_results.values()])
            client_acc_max = max([result.get('global_accuracy', 0) for result in eval_results.values()])
        else:
            avg_local_acc = 0.0
            avg_global_acc = 0.0
            client_acc_std = 0.0
            client_acc_min = 0.0
            client_acc_max = 0.0
            
        # 计算平均特征对齐损失
        if train_results:
            avg_feature_loss = sum(result.get('avg_feature_loss', 0) for result in train_results.values()) / max(1, len(train_results))
        else:
            avg_feature_loss = 0.0
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 更新时间历史记录
        time_history["round_time"].append(round_time)
        time_history["training_time"].append(training_time)
        time_history["communication_time"].append(communication_time)
        time_history["aggregation_time"].append(aggregation_time)
        
        for client_id, stats in time_stats.items():
            if client_id in time_history["client_times"]:
                time_history["client_times"][client_id].append(stats.get("total_time", 0))
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"全局准确率: {global_acc:.2f}%, 最佳: {best_accuracy:.2f}%")
        logger.info(f"平均本地准确率: {avg_local_acc:.2f}%, 平均全局准确率: {avg_global_acc:.2f}%")
        logger.info(f"客户端精度标准差: {client_acc_std:.2f}, 最低: {client_acc_min:.2f}%, 最高: {client_acc_max:.2f}%")
        logger.info(f"类别平衡度: {class_balance:.2f}")
        logger.info(f"特征对齐损失: {avg_feature_loss:.4f}")
        logger.info(f"alpha: {controller.alpha:.3f}, lambda_feature: {controller.lambda_feature:.3f}")
        logger.info(f"轮次总时间: {round_time:.2f}秒, 训练: {training_time:.2f}秒, 通信: {communication_time:.2f}秒, 聚合: {aggregation_time:.2f}秒")
        
        # 记录到wandb
        try:
            metrics = {
                "round": round_idx + 1,
                
                # 全局模型性能
                "global/test_accuracy": global_acc,
                "global/test_loss": global_loss,
                "global/best_accuracy": best_accuracy,
                "global/is_best_model": 1 if is_best else 0,
                "global/class_balance": class_balance,
                
                # 客户端性能
                "local/accuracy": avg_local_acc,
                "global/accuracy": avg_global_acc,
                "global/accuracy_std": client_acc_std,
                "global/accuracy_min": client_acc_min,
                "global/accuracy_max": client_acc_max,
                "feature/alignment": avg_feature_loss,
                
                # 训练参数
                "params/alpha": controller.alpha,
                "params/lambda_feature": controller.lambda_feature,
                
                # 时间统计
                "time/round_seconds": round_time,
                "time/training_seconds": training_time,
                "time/communication_seconds": communication_time,
                "time/aggregation_seconds": aggregation_time,
            }
            
            # 记录每个类别的准确率
            for i, acc in enumerate(class_accs):
                metrics[f"global/class_{i}_accuracy"] = acc
            
            # 记录每个客户端的学习率和精度
            for client_id in range(args.client_number):
                if client_id in eval_results:
                    metrics[f"client/{client_id}/global_accuracy"] = eval_results[client_id].get('global_accuracy', 0)
                    metrics[f"client/{client_id}/local_accuracy"] = eval_results[client_id].get('local_accuracy', 0)
                    metrics[f"client/{client_id}/test_loss"] = eval_results[client_id].get('test_loss', 0)
                
                # 记录学习率
                metrics[f"client/{client_id}/learning_rate"] = lr_tracker.get(client_id, args.lr)
                
                # 记录时间开销
                if client_id in time_stats:
                    metrics[f"client/{client_id}/training_time"] = time_stats[client_id].get("training_time", 0)
                    metrics[f"client/{client_id}/total_time"] = time_stats[client_id].get("total_time", 0)
                    metrics[f"client/{client_id}/copy_time"] = time_stats[client_id].get("copy_time", 0)
                    metrics[f"client/{client_id}/evaluation_time"] = time_stats[client_id].get("evaluation_time", 0)
            
            # 聚类情况
            cluster_stats = {
                f"cluster/{cluster_id}/size": len(clients) 
                for cluster_id, clients in cluster_map.items()
            }
            metrics.update(cluster_stats)

            # 计算聚类异质性指标
            total_clients = sum(len(clients) for clients in cluster_map.values())
            cluster_size_std = np.std([len(clients) for clients in cluster_map.values()])
            metrics["cluster/size_std"] = cluster_size_std
            metrics["cluster/count"] = len(cluster_map)

            # 计算聚类内客户端Tier异质性
            for cluster_id, clients in cluster_map.items():
                if clients:
                    client_tiers = [client_resources[cid]['tier'] for cid in clients]
                    tier_std = np.std(client_tiers)
                    metrics[f"cluster/{cluster_id}/tier_std"] = tier_std
                    metrics[f"cluster/{cluster_id}/avg_tier"] = sum(client_tiers) / len(client_tiers)

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
                # 打印重新聚类信息
                print_cluster_info(cluster_map, client_resources, logger)
            except Exception as e:
                logger.error(f"重新聚类失败: {str(e)}")
    
    # 训练完成 - 记录最终性能总结
    logger.info(f"TierHFL训练完成! 最佳准确率: {best_accuracy:.2f}%")

    # 记录聚类历史统计
    try:
        clustering_stats = clusterer.get_clustering_statistics()
        for i, stats in enumerate(clustering_stats):
            logger.info(f"聚类迭代 {i}: {stats['num_clusters']}个聚类, " 
                    f"平均大小={stats['avg_cluster_size']:.2f}, "
                    f"大小分布={stats['cluster_sizes']}")
            
        # 添加到wandb摘要
        if len(clustering_stats) > 0:
            final_stats = clustering_stats[-1]
            wandb.run.summary.update({
                "final_num_clusters": final_stats['num_clusters'],
                "final_avg_cluster_size": final_stats['avg_cluster_size'],
                "final_cluster_size_std": np.std(final_stats['cluster_sizes']) if final_stats['cluster_sizes'] else 0
            })
    except Exception as e:
        logger.error(f"记录聚类统计信息失败: {str(e)}")
    
    # 输出时间统计摘要
    avg_round_time = sum(time_history["round_time"]) / max(1, len(time_history["round_time"]))
    avg_training_time = sum(time_history["training_time"]) / max(1, len(time_history["training_time"]))
    avg_comm_time = sum(time_history["communication_time"]) / max(1, len(time_history["communication_time"]))
    avg_agg_time = sum(time_history["aggregation_time"]) / max(1, len(time_history["aggregation_time"]))
    
    logger.info(f"平均轮次时间: {avg_round_time:.2f}秒")
    logger.info(f"平均训练时间: {avg_training_time:.2f}秒 ({100*avg_training_time/avg_round_time:.1f}%)")
    logger.info(f"平均通信时间: {avg_comm_time:.2f}秒 ({100*avg_comm_time/avg_round_time:.1f}%)")
    logger.info(f"平均聚合时间: {avg_agg_time:.2f}秒 ({100*avg_agg_time/avg_round_time:.1f}%)")
    
    try:
        # 记录最终性能数据
        wandb.run.summary.update({
            "best_accuracy": best_accuracy,
            "final_global_accuracy": global_acc,
            "final_local_accuracy": avg_local_acc,
            "final_class_balance": class_balance,
            "avg_round_time": avg_round_time,
            "avg_training_time": avg_training_time,
            "avg_communication_time": avg_comm_time,
            "avg_aggregation_time": avg_agg_time,
            "total_rounds": args.rounds
        })
        
        # 关闭wandb
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()

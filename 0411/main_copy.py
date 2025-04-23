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
import torchvision
import torchvision.transforms as transforms
import math

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# 导入自定义模块
from model.resnet import EnhancedServerModel, TierAwareClientModel,ImprovedGlobalClassifier
from utils.tierhfl_aggregator import StabilizedAggregator
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_trainer import ClusterAwareParallelTrainer, AdaptiveTrainingController, ModelFeatureClusterer
from utils.tierhfl_loss import TierHFLLoss


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
    parser.add_argument('--max_workers', default=None, type=int, help='最大并行工作线程数')
    
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
    """评估全局模型在全局测试集上的性能"""
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
            except Exception as e:
                print(f"评估中出现错误: {str(e)}")
                continue
    
    accuracy = 100.0 * correct / max(1, total)
    return accuracy

# 主函数
def main():
    """主函数，集成所有优化改动"""
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
    
    # 加载全局测试集
    logger.info("加载全局IID测试集用于评估泛化性能...")
    global_test_loader = load_global_test_set(args)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配异构资源...")
    client_resources = allocate_device_resources(args.client_number)
    
    # 创建客户端管理器
    logger.info("创建客户端管理器...")
    client_manager = TierHFLClientManager()
    
    # 注册客户端并优化参数设置
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
        
        # 根据tier调整学习率
        if tier <= 3:  # 高性能设备
            client.lr = args.lr * 1.2
        elif tier >= 6:  # 低性能设备
            client.lr = args.lr * 0.8
    
    # 创建客户端模型
    logger.info(f"创建双路径客户端模型...")
    client_models = {}
    
    for client_id, resource in client_resources.items():
        tier = resource["tier"]
        client_models[client_id] = TierAwareClientModel(num_classes=class_num, tier=tier)
    
    # 创建服务器特征提取模型
    logger.info("创建服务器特征提取模型...")
    server_model = EnhancedServerModel().to(device)
    
    # 创建全局分类器
    logger.info("创建全局分类器...")
    global_classifier = ImprovedGlobalClassifier(feature_dim=128, num_classes=class_num).to(device)
    
    # 创建稳定化聚合器
    logger.info("创建稳定化聚合器...")
    aggregator = StabilizedAggregator(beta=args.beta, device=device)
    
    # 创建客户端聚类器
    logger.info("创建数据分布聚类器...")
    clusterer = ModelFeatureClusterer(num_clusters=args.n_clusters)
    
    # 创建损失函数
    logger.info("创建混合损失函数...")
    loss_fn = TierHFLLoss(alpha=args.init_alpha, lambda_feature=args.init_lambda)
    
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
        global_classifier=global_classifier,
        loss_fn=loss_fn,
        device=device,
        max_workers=args.max_workers
    )
    trainer.rounds = args.rounds  # 添加总轮数属性
    
    # 注册客户端模型
    trainer.register_client_models(client_models)
    
    # 设置训练环境
    trainer.setup_training(cluster_map=cluster_map)
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    
    # 创建学习率追踪字典
    lr_tracker = {client_id: args.lr for client_id in range(args.client_number)}
    
    # 记录每个客户端的性能历史
    client_performance_history = {client_id: {
        "train_acc": [], 
        "test_acc": [], 
        "local_acc": [], 
        "global_acc": [], 
        "cross_client_acc": []
    } for client_id in range(args.client_number)}
    
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
        train_results, eval_results, server_models, global_classifier_states, shared_states, time_stats, training_time = trainer.execute_parallel_training(round_idx=round_idx)

        # 防止空结果
        if not train_results or not eval_results:
            logger.error("训练或评估结果为空，跳过本轮")
            continue
        
        # 记录每个客户端的学习率
        for client_id in range(args.client_number):
            client = client_manager.get_client(client_id)
            if client:
                lr_tracker[client_id] = client.lr
        
        # 计算本轮客户端性能指标 - 用于聚合权重计算
        client_performance = {}
        for client_id, result in eval_results.items():
            client_performance[client_id] = {
                'local_accuracy': result.get('local_accuracy', 0),
                'global_accuracy': result.get('global_accuracy', 0),
                'cross_client_accuracy': result.get('cross_client_accuracy', 0)
            }
        
        # 计算全局聚合权重
        aggregator.round_progress = round_idx / args.rounds  # 更新训练进度
        
        # 聚合服务器模型
        logger.info("聚合服务器模型和客户端共享层...")
        logger.info(f"开始聚合客户端共享层，收集到 {len(shared_states)} 个客户端状态")
        aggregation_start_time = time.time()
        
        # 聚合客户端共享层
        aggregated_shared_state = aggregator.aggregate_clients(
            client_states=shared_states,
            client_performance=client_performance,
            client_clusters=cluster_map
        )
        
        # 聚合服务器模型
        if server_models:
            aggregated_server_model = aggregator.aggregate_server(
                server_models, 
                eval_results=eval_results,
                cluster_map=cluster_map
            )
            
            # 更新服务器模型
            try:
                server_model.load_state_dict(aggregated_server_model)
            except Exception as e:
                logger.error(f"更新服务器模型失败: {str(e)}")
        
        # 聚合全局分类器
        if global_classifier_states:
            # 计算基于性能的聚合权重
            classifier_weights = {}
            total_weight = 0.0
            for cluster_id in global_classifier_states.keys():
                cluster_clients = cluster_map.get(cluster_id, [])
                valid_clients = [c for c in cluster_clients if c in eval_results]
                
                if valid_clients:
                    # 使用全局准确率加权
                    avg_acc = sum(eval_results[c].get('global_accuracy', 0) 
                               for c in valid_clients) / len(valid_clients)
                    # Sigmoid加权
                    weight = 1.0 / (1.0 + math.exp(-0.1 * (avg_acc - 50)))
                    classifier_weights[cluster_id] = weight
                    total_weight += weight
                else:
                    classifier_weights[cluster_id] = 0.5
                    total_weight += 0.5
            
            # 归一化权重
            if total_weight > 0:
                for cluster_id in classifier_weights:
                    classifier_weights[cluster_id] /= total_weight
            
            # 聚合分类器
            aggregated_classifier = trainer._aggregate_classifiers(
                global_classifier_states, 
                classifier_weights
            )
            
            # 更新全局分类器
            try:
                global_classifier.load_state_dict(aggregated_classifier)
            except Exception as e:
                logger.error(f"更新全局分类器失败: {str(e)}")

        aggregation_time = time.time() - aggregation_start_time
        
        # 更新客户端模型的共享层
        comm_start_time = time.time()
        for client_id, model in client_models.items():
            # 保存个性化参数
            personalized_params = {}
            for name, param in model.named_parameters():
                if 'personalized_path' in name or 'local_classifier' in name:
                    personalized_params[name] = param.clone()

            # 确保模型在CPU上以节省GPU内存
            model = model.cpu()
            
            # 更新共享层参数
            for name, param in model.named_parameters():
                if 'shared_base' in name and name in aggregated_shared_state:
                    param.data = aggregated_shared_state[name].clone()
            
            # 将更新后的模型存回字典
            client_models[client_id] = model
        
        communication_time = time.time() - comm_start_time

        # 评估全局模型在独立测试集上的性能
        # 选择一个tier 1的客户端进行评估
        tier1_clients = [cid for cid, resource in client_resources.items() if resource['tier'] == 1]
        if tier1_clients:
            sample_client_id = tier1_clients[0]
        else:
            sample_client_id = list(client_models.keys())[0]
            
        # 获取选中客户端的模型
        sample_client_model = client_models[sample_client_id]

        # 确保所有模型在同一设备上
        sample_client_model = sample_client_model.to(device)
        server_model = server_model.to(device)
        global_classifier = global_classifier.to(device)

        global_model_accuracy = evaluate_global_model(
            sample_client_model, server_model, global_classifier, 
            global_test_loader, device)

        logger.info(f"全局模型在独立测试集上的准确率: {global_model_accuracy:.2f}%")
        
        # 计算本轮平均准确率
        avg_local_acc = np.mean([result.get('local_accuracy', 0) for result in eval_results.values()])
        avg_global_acc = np.mean([result.get('global_accuracy', 0) for result in eval_results.values()])
        avg_cross_client_acc = np.mean([result.get('cross_client_accuracy', 0) 
                                      for result in eval_results.values() 
                                      if 'cross_client_accuracy' in result])
        
        # 更新最佳准确率
        is_best = global_model_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_model_accuracy
            # 保存最佳模型
            try:
                torch.save({
                    'client_model': sample_client_model.state_dict(),
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
        
        # 更新时间历史记录
        time_history["round_time"].append(round_time)
        time_history["training_time"].append(training_time)
        time_history["communication_time"].append(communication_time)
        time_history["aggregation_time"].append(aggregation_time)
        
        # 更新客户端性能历史
        for client_id, result in eval_results.items():
            client_performance_history[client_id]['local_acc'].append(result.get('local_accuracy', 0))
            client_performance_history[client_id]['global_acc'].append(result.get('global_accuracy', 0))
            client_performance_history[client_id]['cross_client_acc'].append(
                result.get('cross_client_accuracy', 0) if 'cross_client_accuracy' in result else 0)
            
            if client_id in time_stats:
                time_history["client_times"][client_id].append(time_stats[client_id].get("training_time", 0))
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"本地平均准确率: {avg_local_acc:.2f}%, 全局平均准确率: {avg_global_acc:.2f}%")
        if avg_cross_client_acc > 0:
            logger.info(f"跨客户端平均准确率: {avg_cross_client_acc:.2f}%")
        logger.info(f"全局模型在独立测试集上的准确率: {global_model_accuracy:.2f}%")
        logger.info(f"最佳准确率: {best_accuracy:.2f}%")
        logger.info(f"轮次总时间: {round_time:.2f}秒, 训练: {training_time:.2f}秒, 通信: {communication_time:.2f}秒, 聚合: {aggregation_time:.2f}秒")
        
        # 记录到wandb
        try:
            metrics = {
                "round": round_idx + 1,
                
                # 全局模型性能
                "global/test_accuracy": global_model_accuracy,
                "global/best_accuracy": best_accuracy,
                
                # 客户端性能
                "local/avg_accuracy": avg_local_acc,
                "global/avg_accuracy": avg_global_acc,
                "cross_client/avg_accuracy": avg_cross_client_acc if avg_cross_client_acc > 0 else None,
                
                # 训练参数
                "params/beta": aggregator.beta,
                
                # 时间统计
                "time/round_seconds": round_time,
                "time/training_seconds": training_time,
                "time/communication_seconds": communication_time,
                "time/aggregation_seconds": aggregation_time,
            }
            
            # 记录每个客户端的性能
            for client_id in range(args.client_number):
                if client_id in eval_results:
                    metrics[f"client/{client_id}/local_accuracy"] = eval_results[client_id].get('local_accuracy', 0)
                    metrics[f"client/{client_id}/global_accuracy"] = eval_results[client_id].get('global_accuracy', 0)
                    if 'cross_client_accuracy' in eval_results[client_id]:
                        metrics[f"client/{client_id}/cross_client_accuracy"] = eval_results[client_id]['cross_client_accuracy']
            
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

        # 每10轮更新学习率
        if round_idx > 0 and round_idx % 10 == 0:  # 每10轮衰减一次学习率
            logger.info("更新客户端学习率...")
            for client_id in range(args.client_number):
                client = client_manager.get_client(client_id)
                if client:
                    # 使用学习率衰减因子
                    client.lr *= args.lr_factor
                    lr_tracker[client_id] = client.lr
                    logger.info(f"客户端 {client_id} 学习率更新为: {client.lr:.6f}")
    
    # 训练完成 - 记录最终性能总结
    logger.info(f"TierHFL训练完成! 最佳准确率: {best_accuracy:.2f}%")

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
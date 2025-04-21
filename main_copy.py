import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import random
import logging
import wandb
import time
from collections import defaultdict

# 导入自定义模块
from model.tierhfl_models import TierHFLClientModel, TierHFLServerModel, TierHFLGlobalClassifier
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_server import TierHFLCentralServer, TierHFLServerGroup
from utils.tierhfl_loss import TierHFLLoss, GradientGuideModule, ContrastiveLearningLoss
from utils.tierhfl_trainer import TierHFLTrainer
from utils.tierhfl_evaluator import TierHFLEvaluator
from utils.tierhfl_grouping import TierHFLGroupingStrategy
from utils.data_utils import load_data, create_iid_test_dataset

# 导入数据加载函数
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10


def add_args(parser):
    """添加命令行参数"""
    # 实验标识参数
    parser.add_argument('--running_name', default="TierHFL", type=str, help='实验名称')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.001, type=float, help='初始学习率')
    parser.add_argument('--lr_factor', default=0.7, type=float, help='学习率衰减因子')
    parser.add_argument('--lr_patience', default=5, type=int, help='学习率调度器耐心值')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet56', 
                       help='神经网络模型 (resnet56 或 resnet110)')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='cifar10', help='训练数据集')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', 
                       help='数据划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=1, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=10, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    parser.add_argument('--n_groups', default=3, type=int, help='服务器组数量')
    parser.add_argument('--max_workers', default=None, type=int, help='最大并行工作线程数')
    # 添加新参数
    parser.add_argument('--buffer_size', default=1000, type=int, 
                       help='全局分类器经验回放缓冲区大小')
    parser.add_argument('--balance_alpha', default=0.7, type=float, 
                       help='平衡权重中性能与多样性的因子(更高的值更重视性能)')

    parser.add_argument('--disable_buffer', type=bool, default=True, 
                   help='是否禁用经验回放缓冲区')
    
    # 添加内存优化相关参数
    parser.add_argument('--batch_size_server', type=int, default=32, 
                       help='服务器训练批次大小，控制内存使用')
    parser.add_argument('--accum_steps', type=int, default=4, 
                       help='梯度累积步骤数')
    parser.add_argument('--feature_extract_batch', type=int, default=16,
                       help='特征提取时的批次大小')
    parser.add_argument('--max_cache_size', type=int, default=10,
                       help='投影矩阵缓存的最大大小')
    
    # TierHFL特有参数
    parser.add_argument('--init_alpha', default=0.5, type=float, 
                       help='初始本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.1, type=float, 
                       help='初始特征对齐损失权重')
    parser.add_argument('--init_beta', default=0.7, type=float, 
                       help='初始梯度引导系数')
    
    # 其他参数
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--eval_interval', default=1, type=int, 
                       help='评估间隔（轮数）')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(args):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.running_name}.log")
        ]
    )
    logger = logging.getLogger("TierHFL")
    
    return logger

def setup_wandb(args):
    """设置wandb"""
    try:
        wandb.init(
            project="TierHFL",
            name=args.running_name,
            config=args,
            tags=[f"model_{args.model}", f"dataset_{args.dataset}"]
        )
        
        # 设置自定义面板
        wandb.define_metric("round")
        wandb.define_metric("global/*", step_metric="round")
        wandb.define_metric("local/*", step_metric="round")
        wandb.define_metric("client/*", step_metric="round")
        wandb.define_metric("time/*", step_metric="round")
        wandb.define_metric("params/*", step_metric="round")
        
    except Exception as e:
        print(f"wandb初始化失败: {e}")
        # 使用离线模式
        try:
            wandb.init(mode="offline", project="TierHFL", name=args.running_name)
        except:
            print("完全禁用wandb")

def allocate_device_resources(client_number):
    """分配客户端资源"""
    resources = {}
    
    # 随机分配tier (范围1-4)
    tier_weights = [0.25, 0.35, 0.25, 0.15]  # 各tier的分布概率
    tiers = random.choices(range(1, 5), weights=tier_weights, k=client_number)
    
    for client_id in range(client_number):
        tier = tiers[client_id]
        
        # 根据tier分配计算能力
        if tier == 1:  # 高性能设备
            compute_power = random.uniform(0.8, 1.0)
            network_speed = random.choice([50, 100, 200])  # Mbps
        elif tier == 2:  # 中高性能设备
            compute_power = random.uniform(0.6, 0.8)
            network_speed = random.choice([30, 50, 100])
        elif tier == 3:  # 中等性能设备
            compute_power = random.uniform(0.4, 0.6)
            network_speed = random.choice([20, 30, 50])
        else:  # tier == 4，低性能设备
            compute_power = random.uniform(0.1, 0.4)
            network_speed = random.choice([5, 10, 20])
        
        # 存储资源信息
        resources[client_id] = {
            "tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed
        }
    
    return resources

def create_models(client_resources, args, device='cuda'):
    """创建客户端和服务器模型"""
    # 客户端模型
    client_models = {}
    for client_id, resource in client_resources.items():
        client_models[client_id] = TierHFLClientModel(
            base_model=args.model,
            num_classes=10 if args.dataset == 'cifar10' else 100,
            tier=resource['tier']
        ).to(device)
    
    # 服务器模型和全局分类器
    server_models = {}
    global_classifiers = {}
    for group_id in range(args.n_groups):
        server_models[group_id] = TierHFLServerModel(
            base_model=args.model,
            in_channels=32,  # 客户端共享层的输出通道数，layer2_shared的输出
            feature_dim=128
        ).to(device)
        
        global_classifiers[group_id] = TierHFLGlobalClassifier(
            feature_dim=128,
            num_classes=10 if args.dataset == 'cifar10' else 100,
            buffer_size=args.buffer_size
        ).to(device)
    
    return client_models, server_models, global_classifiers

def main():
    """主函数修改，集成新功能"""
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args)
    logger.info(f"开始TierHFL实验: {args.running_name}")
    
    # 设置wandb
    setup_wandb(args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_data(args)
    
    if args.dataset != "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, _ = dataset
    
    # 创建均衡测试集
    logger.info("创建均衡IID测试集")
    iid_test_loader = create_iid_test_dataset(args)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配资源")
    client_resources = allocate_device_resources(args.client_number)
    
    # 创建客户端和服务器模型
    logger.info("创建客户端和服务器模型")
    client_models, server_models, global_classifiers = create_models(
        client_resources, args, device)
    
    # 创建客户端管理器
    logger.info("创建客户端管理器")
    client_manager = TierHFLClientManager()
    
    # 注册客户端 - 传递特征提取批次大小
    for client_id in range(args.client_number):
        resource = client_resources[client_id]
        
        # 创建客户端
        client = client_manager.add_client(
            client_id=client_id,
            tier=resource["tier"],
            train_data=train_data_local_dict[client_id],
            test_data=test_data_local_dict[client_id],
            model=client_models[client_id],
            device=device,
            lr=args.lr,
            local_epochs=args.client_epoch,
            feature_extract_batch=args.feature_extract_batch  # 传递新参数
        )
    
    # 创建中央服务器
    logger.info("创建中央服务器")
    central_server = TierHFLCentralServer(device)
    
    # 创建服务器组
    for group_id in range(args.n_groups):
        central_server.create_server_group(
            group_id, server_models[group_id], global_classifiers[group_id])
    
    # 创建分组策略
    logger.info(f"创建分组策略，组数: {args.n_groups}")
    grouping_strategy = TierHFLGroupingStrategy(num_groups=args.n_groups)
    
    # 创建损失函数 - 传递缓存大小限制
    logger.info("创建损失函数")
    loss_fn = TierHFLLoss(
        alpha=args.init_alpha, 
        lambda_feature=args.init_lambda,
        max_cache_size=args.max_cache_size
    )
    
    # 创建训练器 - 传递批处理和梯度累积参数
    logger.info("创建训练器")
    trainer = TierHFLTrainer(
        client_manager=client_manager,
        central_server=central_server,
        grouping_strategy=grouping_strategy,
        loss_fn=loss_fn,
        gradient_guide=None,  # 不再使用梯度引导
        max_workers=args.max_workers
    )
    trainer.rounds = args.rounds  # 添加总轮数属性
    trainer.batch_size_server = args.batch_size_server  # 添加服务器批次大小
    trainer.accum_steps = args.accum_steps  # 添加梯度累积步骤数
    trainer.disable_buffer = args.disable_buffer  # 添加禁用缓冲区标志
    # 创建评估器
    logger.info("创建评估器")
    evaluator = TierHFLEvaluator(
        client_manager=client_manager,
        central_server=central_server,
        test_loader=iid_test_loader,
        device=device
    )

    # 在训练前评估之前执行一次分组
    logger.info("执行初始客户端分组")
    initial_client_groups = grouping_strategy.group_clients(
        client_manager, client_models, 0)  # 0表示第0轮

    # 将分组结果应用到中央服务器
    for group_id, client_ids in initial_client_groups.items():
        for client_id in client_ids:
            central_server.assign_client_to_group(client_id, group_id)

    # 然后进行训练前评估
    logger.info("训练前初始评估")
    eval_result = evaluator.conduct_comprehensive_evaluation(
        client_models=client_models, round_idx=0)
    
    # 记录初始评估结果到wandb
    wandb.log({
        "round": 0,
        "global/accuracy": eval_result['global_result']['global_accuracy'],
        "global/tier_accuracy": eval_result['global_result'].get('tier_accuracy', {}),
        "global/cross_client_accuracy": eval_result['cross_client_result']['overall_avg_performance'] 
                                       if eval_result['cross_client_result'] else 0
    })
    
    # 训练循环
    logger.info(f"开始训练 {args.rounds} 轮")
    start_time = time.time()
    
    # 记录最佳性能
    best_accuracy = 0.0
    
    for round_idx in range(1, args.rounds + 1):
        logger.info(f"===== 轮次 {round_idx}/{args.rounds} =====")
        round_start_time = time.time()
        
        # 执行并行训练 - 使用优化的并行训练方法
        train_result = trainer.execute_parallel_training(
            client_models=client_models, round_idx=round_idx)
        
        # 提取训练结果
        train_results = train_result['train_results']
        eval_results = train_result['eval_results']
        client_states = train_result['client_states']
        server_states = train_result['server_states']
        classifier_states = train_result['classifier_states']
        time_stats = train_result['time_stats']
        training_time = train_result['training_time']
        avg_local_acc = train_result['avg_local_acc']
        avg_global_acc = train_result['avg_global_acc']
        
        # 每args.eval_interval轮进行一次全面评估
        if round_idx % args.eval_interval == 0 or round_idx == args.rounds:
            logger.info(f"轮次 {round_idx} 进行全面评估")
            eval_result = evaluator.conduct_comprehensive_evaluation(
                client_models=client_models, round_idx=round_idx)
            
            # 提取评估结果
            global_accuracy = eval_result['global_result']['global_accuracy']
            tier_accuracy = eval_result['global_result'].get('tier_accuracy', {})
            cross_client_accuracy = eval_result['cross_client_result']['overall_avg_performance'] \
                                   if eval_result['cross_client_result'] else 0
            
            # 更新最佳性能
            if global_accuracy > best_accuracy:
                best_accuracy = global_accuracy
                logger.info(f"新的最佳性能: {best_accuracy:.2f}%")
                
                # 保存最佳模型
                try:
                    # 保存客户端模型
                    torch.save({
                        'round': round_idx,
                        'accuracy': best_accuracy,
                        'client_states': {cid: model.state_dict() for cid, model in client_models.items()},
                        'server_states': {gid: model.state_dict() for gid, model in server_models.items()},
                        'classifier_states': {gid: model.state_dict() for gid, model in global_classifiers.items()},
                    }, f"{args.running_name}_best_model.pth")
                    logger.info(f"保存最佳模型, 轮次 {round_idx}, 准确率: {best_accuracy:.2f}%")
                except Exception as e:
                    logger.error(f"保存模型失败: {str(e)}")
        else:
            # 非全面评估轮次，使用训练结果中的评估数据
            global_accuracy = avg_global_acc
            tier_accuracy = defaultdict(float)
            for client_id, result in eval_results.items():
                tier = client_resources[client_id]['tier']
                tier_accuracy[tier] = tier_accuracy[tier] + result['global_accuracy']
            
            # 计算每个tier的平均准确率
            for tier in tier_accuracy:
                tier_count = sum(1 for cid, res in client_resources.items() 
                               if res['tier'] == tier and cid in eval_results)
                if tier_count > 0:
                    tier_accuracy[tier] /= tier_count
            
            cross_client_accuracy = 0  # 非全面评估轮次不计算跨客户端性能
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 记录到wandb
        wandb_metrics = {
            "round": round_idx,
            "global/accuracy": global_accuracy,
            "global/best_accuracy": best_accuracy,
            "local/accuracy": avg_local_acc,
            "global/avg_accuracy": avg_global_acc,
            "time/round_seconds": round_time,
            "time/training_seconds": training_time,
        }
        
        # 添加每个tier的准确率
        for tier, acc in tier_accuracy.items():
            wandb_metrics[f"tier_{tier}/accuracy"] = acc
        
        # 如果有跨客户端评估结果，添加
        if cross_client_accuracy > 0:
            wandb_metrics["global/cross_client_accuracy"] = cross_client_accuracy
        
        # 记录每个客户端的性能
        for client_id, result in eval_results.items():
            tier = client_resources[client_id]['tier']
            wandb_metrics[f"client/{client_id}/tier"] = tier
            wandb_metrics[f"client/{client_id}/local_accuracy"] = result['local_accuracy']
            wandb_metrics[f"client/{client_id}/global_accuracy"] = result['global_accuracy']
            
            # 记录时间统计
            if client_id in time_stats:
                wandb_metrics[f"client/{client_id}/training_time"] = time_stats[client_id]['training_time']
        
        wandb.log(wandb_metrics)
    
    # 训练完成
    total_time = time.time() - start_time
    logger.info(f"训练完成，总耗时: {total_time/60:.2f}分钟")
    logger.info(f"最佳准确率: {best_accuracy:.2f}%")
    
    # 记录最终统计
    wandb.run.summary.update({
        "best_accuracy": best_accuracy,
        "total_training_time": total_time/60,  # 分钟
        "total_rounds": args.rounds
    })
    
    # 关闭wandb
    wandb.finish()

def load_data(args):
    """加载数据集"""
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

if __name__ == "__main__":
    main()
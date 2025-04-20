import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np


def load_data(args):
    """加载数据集"""
    if args.dataset == "cifar10":
        from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10 as data_loader
    elif args.dataset == "cifar100":
        from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100 as data_loader
    elif args.dataset == "cinic10":
        from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10 as data_loader
        args.data_dir = './data/cinic10/'
    else:
        from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10 as data_loader

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

def create_iid_test_dataset(args, num_classes=10):
    """创建均衡分布的IID测试数据集"""
    # 根据数据集选择合适的转换
    if args.dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                [0.2673, 0.2564, 0.2762])
        ])
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "cinic10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                [0.24205776, 0.23828046, 0.25874835])
        ])
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'cinic10', 'test'),
            transform=transform_test)
    else:
        # 默认使用CIFAR10
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test)
    
    # 创建均衡的数据加载器
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return test_loader

def analyze_client_data_distribution(train_data_local_dict, class_num=10):
    """分析客户端数据分布"""
    distributions = {}
    
    for client_id, dataloader in train_data_local_dict.items():
        # 初始化类别计数
        class_counts = {i: 0 for i in range(class_num)}
        total_samples = 0
        
        # 统计各类别数量
        for _, labels in dataloader:
            for label in labels.numpy():
                class_counts[label] += 1
                total_samples += 1
        
        # 计算百分比
        if total_samples > 0:
            class_percentage = {k: v/total_samples*100 for k, v in class_counts.items()}
        else:
            class_percentage = {k: 0 for k in class_counts.keys()}
        
        distributions[client_id] = {
            'counts': class_counts,
            'percentage': class_percentage,
            'total': total_samples
        }
    
    return distributions
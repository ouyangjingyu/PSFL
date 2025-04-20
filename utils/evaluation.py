import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import math
import copy

def create_iid_test_dataset(args, class_num=10):
    """创建独立的IID测试数据集"""
    if args.dataset.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset.lower() == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                [0.2673, 0.2564, 0.2762])
        ])
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform)
    else:  # CINIC10或其他
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                [0.24205776, 0.23828046, 0.25874835])
        ])
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'cinic10', 'test'),
            transform=transform)
    
    # 创建平衡的数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return test_loader

def evaluate_global_generalization(client_models, server_model, global_classifier, 
                                   iid_test_loader, client_manager, device):
    """评估全局模型在独立IID测试集上的泛化性能"""
    # 确保模型在正确的设备上
    server_model = server_model.to(device)
    global_classifier = global_classifier.to(device)
    
    print("\n===== 评估全局模型泛化性 =====")
    results = {}
    
    # 使用每个客户端模型评估全局泛化性
    for client_id, client_model in client_models.items():
        client = client_manager.get_client(client_id)
        if client is None:
            continue
            
        tier = client.tier
        client_model = client_model.to(device)
        client_model.eval()
        
        # 评估指标
        correct = 0
        total = 0
        class_correct = [0] * 10  # 默认10类
        class_total = [0] * 10
        
        with torch.no_grad():
            for data, target in iid_test_loader:
                data, target = data.to(device), target.to(device)
                
                # 客户端特征提取
                _, features = client_model(data)
                
                # 服务器处理
                server_features = server_model(features, tier=tier)
                
                # 全局分类
                logits = global_classifier(server_features)
                
                # 计算准确率
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 计算每个类别的准确率
                for i in range(len(target)):
                    label = target[i].item()
                    if label < len(class_correct):
                        class_total[label] += 1
                        if predicted[i] == label:
                            class_correct[label] += 1
        
        # 计算总体准确率
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        # 计算每个类别的准确率
        class_acc = [100.0 * c / max(1, t) for c, t in zip(class_correct, class_total)]
        
        results[client_id] = {
            "tier": tier,
            "accuracy": accuracy,
            "class_acc": class_acc
        }
        
        print(f"客户端 {client_id} (Tier {tier}) - IID测试集准确率: {accuracy:.2f}%")
        print(f"  类别准确率: {[f'{acc:.1f}%' for acc in class_acc]}")
    
    # 计算平均准确率
    avg_accuracy = sum(r["accuracy"] for r in results.values()) / max(1, len(results))
    print(f"\n平均IID测试集准确率: {avg_accuracy:.2f}%")
    
    return results

def evaluate_cross_client_generalization(client_models, server_model, global_classifier, 
                                         client_manager, device):
    """评估模型在跨客户端数据上的泛化性能"""
    print("\n===== 评估跨客户端泛化性 =====")
    
    # 确保模型在正确的设备上
    server_model = server_model.to(device)
    global_classifier = global_classifier.to(device)
    
    # 初始化结果矩阵
    client_ids = list(client_models.keys())
    num_clients = len(client_ids)
    cross_acc_matrix = np.zeros((num_clients, num_clients))
    
    # 对每对客户端进行交叉评估
    for i, client_id_i in enumerate(client_ids):
        client_i = client_manager.get_client(client_id_i)
        if client_i is None:
            continue
            
        client_model_i = client_models[client_id_i].to(device)
        client_model_i.eval()
        
        # 使用每个客户端的测试集评估
        for j, client_id_j in enumerate(client_ids):
            if i == j:  # 跳过自身评估
                cross_acc_matrix[i, j] = float('nan')
                continue
                
            client_j = client_manager.get_client(client_id_j)
            if client_j is None:
                continue
                
            tier_i = client_i.tier
            
            # 使用客户端j的测试数据
            test_data = client_j.test_data
            
            # 评估指标
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_data:
                    data, target = data.to(device), target.to(device)
                    
                    # 客户端i的特征提取
                    _, features = client_model_i(data)
                    
                    # 服务器处理
                    server_features = server_model(features, tier=tier_i)
                    
                    # 全局分类
                    logits = global_classifier(server_features)
                    
                    # 计算准确率
                    _, predicted = logits.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            # 计算准确率
            if total > 0:
                accuracy = 100.0 * correct / total
                cross_acc_matrix[i, j] = accuracy
                print(f"客户端{client_id_i}模型在客户端{client_id_j}数据上的准确率: {accuracy:.2f}%")
    
    # 计算每个客户端模型的平均跨客户端准确率
    client_avg_acc = []
    for i, client_id in enumerate(client_ids):
        # 排除自身评估结果(nan)
        cross_accs = [acc for j, acc in enumerate(cross_acc_matrix[i]) if j != i and not np.isnan(acc)]
        if cross_accs:
            avg_acc = sum(cross_accs) / len(cross_accs)
            client_avg_acc.append((client_id, avg_acc))
            print(f"客户端{client_id}模型的平均跨客户端准确率: {avg_acc:.2f}%")
    
    # 计算总体平均跨客户端准确率
    if client_avg_acc:
        overall_avg = sum(acc for _, acc in client_avg_acc) / len(client_avg_acc)
        print(f"\n总体平均跨客户端准确率: {overall_avg:.2f}%")
    
    return cross_acc_matrix, client_avg_acc

def analyze_client_data_distribution(train_data_local_dict, dataset_name="cifar10"):
    """统计并打印每个客户端训练集的数据类别分布"""
    print("\n===== 客户端数据分布分析 =====")
    
    # 根据数据集设置类别名称
    if dataset_name.lower() == "cifar10":
        class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    elif dataset_name.lower() == "cifar100":
        class_names = [f"类别{i}" for i in range(100)]  # 简化处理
    else:
        class_names = [f"类别{i}" for i in range(10)]
        
    client_distribution = {}
    
    # 遍历每个客户端的数据
    for client_id, dataloader in train_data_local_dict.items():
        # 初始化类别计数
        class_counts = {i: 0 for i in range(len(class_names))}
        total_samples = 0
        
        # 统计每个类别的样本数
        for _, (_, labels) in enumerate(dataloader):
            for label in labels.numpy():
                class_counts[label] += 1
                total_samples += 1
        
        # 计算类别百分比
        class_percentage = {label: count/total_samples*100 for label, count in class_counts.items() if count > 0}
        
        # 存储结果
        client_distribution[client_id] = {
            "total_samples": total_samples,
            "class_counts": class_counts,
            "class_percentage": class_percentage
        }
        
        # 打印详细信息
        print(f"\n客户端 {client_id} - 总样本数: {total_samples}")
        print("主要类别分布:")
        
        # 按照百分比从高到低排序
        sorted_classes = sorted(class_percentage.items(), key=lambda x: x[1], reverse=True)
        for label, percentage in sorted_classes:
            print(f"  {class_names[label]}: {percentage:.2f}% ({class_counts[label]}样本)")
        
        # 计算主导类别 (>10%)
        dominant_classes = [class_names[label] for label, perc in sorted_classes if perc > 10]
        print(f"主导类别: {', '.join(dominant_classes)}")
    
    # 分析聚类情况
    print("\n===== 聚类适用性分析 =====")
    # 计算客户端间的数据分布相似度
    for i in range(len(client_distribution)):
        for j in range(i+1, len(client_distribution)):
            # 计算Jensen-Shannon距离作为分布相似度
            similarity = calculate_distribution_similarity(
                client_distribution[i]["class_percentage"],
                client_distribution[j]["class_percentage"]
            )
            print(f"客户端{i}与客户端{j}的分布相似度: {similarity:.4f}")
    
    return client_distribution

def calculate_distribution_similarity(dist1, dist2):
    """计算两个分布之间的相似度 (基于Jensen-Shannon散度)"""
    import numpy as np
    import math
    
    # 确保两个分布包含所有类别
    all_classes = set(dist1.keys()).union(set(dist2.keys()))
    
    # 转换为数组并归一化
    p = np.zeros(len(all_classes))
    q = np.zeros(len(all_classes))
    
    for i, cls in enumerate(all_classes):
        p[i] = dist1.get(cls, 0) / 100
        q[i] = dist2.get(cls, 0) / 100
    
    # 归一化
    if np.sum(p) > 0:
        p = p / np.sum(p)
    if np.sum(q) > 0:
        q = q / np.sum(q)
    
    # 计算Jensen-Shannon距离
    m = 0.5 * (p + q)
    js_divergence = 0
    
    for i in range(len(all_classes)):
        if p[i] > 0:
            js_divergence += 0.5 * p[i] * math.log(p[i] / m[i])
        if q[i] > 0:
            js_divergence += 0.5 * q[i] * math.log(q[i] / m[i])
    
    # 转换为相似度 (1 - 距离)
    return 1 - min(1, math.sqrt(js_divergence))
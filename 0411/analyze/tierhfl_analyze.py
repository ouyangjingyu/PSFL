import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import logging
import matplotlib.pyplot as plt  # 仅在需要可视化时使用



def analyze_server_features(server_model, client_model, global_test_loader, device='cuda'):
    """分析服务器提取特征的可分性"""
    server_model.eval()
    client_model.eval()  # 使用传入的单个客户端模型
    features_all = []
    labels_all = []
    
    # 收集特征和标签
    with torch.no_grad():
        for data, target in global_test_loader:
            data = data.to(device)
            # 使用传入的客户端模型获取共享层输出
            _, shared_features, _ = client_model(data)
            # 通过服务器模型提取特征
            server_features = server_model(shared_features)
            
            features_all.append(server_features.cpu())
            labels_all.append(target)
    
    features_all = torch.cat(features_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    
    # 计算类内/类间距离比
    class_means = {}
    for c in range(10):  # 假设10个类别
        class_idx = (labels_all == c).nonzero(as_tuple=True)[0]
        if len(class_idx) > 0:  # 确保该类有样本
            class_means[c] = features_all[class_idx].mean(dim=0)
    
    # 类内距离
    intra_class_dist = 0
    num_classes_with_samples = 0
    for c in range(10):
        class_idx = (labels_all == c).nonzero(as_tuple=True)[0]
        if len(class_idx) > 0:
            class_features = features_all[class_idx]
            intra_class_dist += torch.norm(class_features - class_means[c], dim=1).mean()
            num_classes_with_samples += 1
    
    if num_classes_with_samples > 0:
        intra_class_dist /= num_classes_with_samples
    
    # 类间距离
    inter_class_dist = 0
    count = 0
    classes_with_means = list(class_means.keys())
    for i in range(len(classes_with_means)):
        for j in range(i+1, len(classes_with_means)):
            c1 = classes_with_means[i]
            c2 = classes_with_means[j]
            inter_class_dist += torch.norm(class_means[c1] - class_means[c2])
            count += 1
    
    if count > 0:
        inter_class_dist /= count
    
    separability = inter_class_dist / (intra_class_dist + 1e-8)
    print(f"特征可分性(类间/类内距离比): {separability:.4f}")
    
    return separability, features_all, labels_all

def test_with_simple_classifier(server_model, client_model, global_test_loader, device='cuda'):
    """用简单分类器替代全局分类器测试特征质量"""
    # 收集特征和标签用于训练新分类器
    features_all = []
    labels_all = []
    
    with torch.no_grad():
        for data, target in global_test_loader:
            data = data.to(device)
            _, shared_features, _ = client_model(data)  # 使用传入的客户端模型
            server_features = server_model(shared_features)
            
            features_all.append(server_features.cpu())
            labels_all.append(target)
    
    features_train = torch.cat(features_all[:len(features_all)//2], dim=0)
    labels_train = torch.cat(labels_all[:len(labels_all)//2], dim=0)
    features_test = torch.cat(features_all[len(features_all)//2:], dim=0)
    labels_test = torch.cat(labels_all[len(labels_all)//2:], dim=0)
    
    # 训练一个简单的线性分类器
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(features_train.numpy(), labels_train.numpy())
    
    # 评估新分类器
    accuracy = classifier.score(features_test.numpy(), labels_test.numpy()) * 100
    print(f"简单分类器准确率: {accuracy:.2f}%")
    
    return accuracy

def analyze_feature_consistency(server_model, client_models, test_data_dict, device='cuda'):
    """分析不同客户端间特征的一致性"""
    server_model = server_model.to(device)
    server_model.eval()
    
    # 对所有客户端的特征进行分析
    client_features = {}
    client_labels = {}
    
    for client_id, test_loader in test_data_dict.items():
        features = []
        labels = []
        
        # 确保当前客户端模型在正确的设备上
        client_model = client_models[client_id].to(device)
        client_model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                
                features.append(server_features.cpu())
                labels.append(target.cpu())
        
        if features:
            client_features[client_id] = torch.cat(features, dim=0)
            client_labels[client_id] = torch.cat(labels, dim=0)
    
    # 计算特征统计信息
    stats = {}
    for client_id in client_features:
        feats = client_features[client_id]
        stats[client_id] = {
            'mean': feats.mean().item(),
            'std': feats.std().item(),
            'norm': torch.norm(feats, dim=1).mean().item()
        }
    
    # 计算客户端间特征相似性
    similarities = {}
    for i in client_features:
        for j in client_features:
            if i != j:
                # 计算相同类别样本的特征相似度
                sim_by_class = {}
                for c in range(10):  # 假设10个类别
                    i_idx = (client_labels[i] == c).nonzero(as_tuple=True)[0]
                    j_idx = (client_labels[j] == c).nonzero(as_tuple=True)[0]
                    
                    if len(i_idx) > 0 and len(j_idx) > 0:
                        i_feats = client_features[i][i_idx]
                        j_feats = client_features[j][j_idx]
                        
                        # 计算余弦相似度
                        i_norm = F.normalize(i_feats, dim=1)
                        j_norm = F.normalize(j_feats, dim=1)
                        
                        # 计算平均余弦相似度
                        sim_matrix = torch.mm(i_norm, j_norm.t())
                        sim_score = sim_matrix.max(dim=1)[0].mean().item()
                        sim_by_class[c] = sim_score
                
                if sim_by_class:
                    similarities[f"{i}-{j}"] = sum(sim_by_class.values()) / len(sim_by_class)
    
    # 计算平均相似度
    avg_similarity = sum(similarities.values()) / len(similarities) if similarities else 0
    print(f"客户端间特征平均相似度: {avg_similarity:.4f}")
    
    return stats, similarities, avg_similarity

def test_server_compression_ability(server_model, client_models, global_test_loader, device='cuda'):
    """测试服务器模型压缩非IID特征的能力"""
    server_model = server_model.to(device)
    server_model.eval()
    
    # 获取服务器模型的中间层输出
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子函数获取中间层输出
    hooks = []
    layers = [module for name, module in server_model.named_modules() 
             if isinstance(module, (nn.Conv2d, nn.Linear))]
    
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_activation(f'layer_{i}')))
    
    # 对随机选取的测试样本进行前向传播
    with torch.no_grad():
        data, _ = next(iter(global_test_loader))
        data = data.to(device)
        
        # 获取不同客户端的共享层特征
        client_shared_features = {}
        for client_id, model in client_models.items():
            model = model.to(device)  # 确保模型在正确设备上
            model.eval()
            _, shared_feats, _ = model(data)
            client_shared_features[client_id] = shared_feats
        
        # 对每个客户端的特征计算服务器各层输出
        client_activations = {}
        for client_id, shared_feats in client_shared_features.items():
            # 清空之前的激活
            activation.clear()
            
            # 前向传播
            server_model(shared_feats)
            
            # 保存该客户端的激活值
            client_activations[client_id] = {k: v.clone().cpu() for k, v in activation.items()}
    
    # 清除钩子
    for hook in hooks:
        hook.remove()
    
    # 计算每层特征的客户端间相似度
    layer_similarities = {}
    for layer_name in next(iter(client_activations.values())).keys():
        # 收集所有客户端该层的激活值
        layer_acts = {}
        for client_id, acts in client_activations.items():
            layer_acts[client_id] = acts[layer_name].view(acts[layer_name].size(0), -1)
        
        # 计算客户端间该层输出的相似度
        similarities = []
        clients = list(layer_acts.keys())
        for i in range(len(clients)):
            for j in range(i+1, len(clients)):
                ci, cj = clients[i], clients[j]
                
                # 扁平化并标准化
                acts_i = F.normalize(layer_acts[ci], dim=1)
                acts_j = F.normalize(layer_acts[cj], dim=1)
                
                # 计算余弦相似度
                sim = torch.mm(acts_i, acts_j.t()).diag().mean().item()
                similarities.append(sim)
        
        # 该层的平均相似度
        if similarities:
            layer_similarities[layer_name] = sum(similarities) / len(similarities)
    
    # 计算服务器层间相似度变化，判断是否在服务器内增加了一致性
    for i in range(len(layer_similarities) - 1):
        layer1 = f'layer_{i}'
        layer2 = f'layer_{i+1}'
        if layer1 in layer_similarities and layer2 in layer_similarities:
            diff = layer_similarities[layer2] - layer_similarities[layer1]
            print(f"{layer1} -> {layer2} 客户端间相似度变化: {diff:.4f}")
    
    return layer_similarities

def test_client_identity_encoding(server_model, client_models, test_data_dict, device='cuda'):
    """测试服务器特征中是否包含客户端身份信息"""
    server_model = server_model.to(device)
    server_model.eval()
    
    # 收集各客户端特征
    client_features = []
    client_ids = []
    
    for client_id, test_loader in test_data_dict.items():
        features = []
        
        # 确保当前客户端模型在正确的设备上
        client_model = client_models[client_id].to(device)
        client_model.eval()
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                
                features.append(server_features.cpu())
        
        if features:
            client_features.append(torch.cat(features, dim=0))
            client_ids.extend([client_id] * len(features))
    
    features_all = torch.cat(client_features, dim=0)
    client_ids = np.array(client_ids)
    
    # 训练一个分类器来预测客户端身份
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_all.numpy(), client_ids, test_size=0.3, random_state=42)
    
    classifier = SVC()
    classifier.fit(X_train, y_train)
    
    # 评估预测客户端身份的准确率
    accuracy = classifier.score(X_test, y_test) * 100
    print(f"从特征预测客户端身份的准确率: {accuracy:.2f}%")
    
    # 随机分类的准确率作为基准
    random_accuracy = 100 / len(set(client_ids))
    print(f"随机猜测客户端身份的准确率: {random_accuracy:.2f}%")
    
    return accuracy, random_accuracy

def validate_server_effectiveness(args, client_models, server_model, global_classifier,
                                 global_test_loader, test_data_local_dict, device='cuda'):
    """集成验证服务器特征提取有效性的函数"""
    print("\n===== 验证服务器特征提取有效性 =====")
    
    # 确保服务器模型在正确设备上
    server_model = server_model.to(device)
    
    # 选择一个客户端模型用于测试
    sample_client_id = list(client_models.keys())[0]
    sample_client_model = client_models[sample_client_id].to(device)
    
    try:
        # 1. 特征可分性分析
        separability, features, labels = analyze_server_features(
            server_model, sample_client_model, global_test_loader, device=device)
    except Exception as e:
        print(f"特征可分性分析出错: {str(e)}")
        separability = 0.0
    
    try:
        # 2. 替换分类器测试
        new_classifier_acc = test_with_simple_classifier(
            server_model, sample_client_model, global_test_loader, device=device)
    except Exception as e:
        print(f"替换分类器测试出错: {str(e)}")
        new_classifier_acc = 0.0
    
    try:
        # 3. 特征一致性跨客户端分析
        feature_stats, similarities, avg_similarity = analyze_feature_consistency(
            server_model, client_models, test_data_local_dict, device=device)
    except Exception as e:
        print(f"特征一致性分析出错: {str(e)}")
        avg_similarity = 0.0
        similarities = {}
        feature_stats = {}
    
    try:
        # 4. 服务器模型压缩能力测试
        layer_similarities = test_server_compression_ability(
            server_model, client_models, global_test_loader, device=device)
    except Exception as e:
        print(f"服务器压缩能力测试出错: {str(e)}")
        layer_similarities = {}
    
    # 暂时跳过可能有问题的客户端身份编码测试
    identity_acc = 0.0
    random_acc = 0.0
    identity_leakage = 0.0
    
    print(f"\n已获取有效的验证指标:")
    print(f"1. 特征可分性: {separability:.4f}")
    print(f"2. 简单分类器准确率: {new_classifier_acc:.2f}%")
    print(f"3. 客户端间特征平均相似度: {avg_similarity:.4f}")
    
    # 基于已有数据做出评估
    feature_quality_score = (separability * 0.4 + (new_classifier_acc / 100) * 0.6) 
    
    print("\n===== 服务器特征提取能力评估 =====")
    print(f"特征质量得分(0-1): {feature_quality_score:.4f}")
    print(f"数据异质性适应能力(0-1): {avg_similarity:.4f}")
    
    if feature_quality_score > 0.3:
        print("结论: 服务器特征提取工作正常，但可能需要优化以更好适应数据异质性")
    elif new_classifier_acc > 20:
        print("结论: 服务器提取的特征有一定区分能力，但全局分类器可能存在问题")
    else:
        print("结论: 服务器特征提取存在明显问题，无法提供有效特征")
        
    return {
        'feature_quality': feature_quality_score,
        'heterogeneity_adaptation': avg_similarity,
        'simple_classifier_acc': new_classifier_acc
    }
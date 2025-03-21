import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import torch.nn.functional as F
from collections import defaultdict


def extract_model_predictions(model, dataset, device, num_classes=10, num_samples=500):
    """
    提取模型在指定数据集上的预测分布
    
    Args:
        model: 要评估的模型
        dataset: 评估数据集
        device: 计算设备
        num_classes: 类别数量
        num_samples: 采样数量
        
    Returns:
        预测分布向量
    """
    # 创建数据加载器
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
    
    # 初始化预测分布
    pred_distribution = torch.zeros(num_classes)
    total_samples = 0
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            
            # 前向传播获取预测
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # 计算预测类别
            _, predicted = torch.max(outputs, 1)
            
            # 更新预测分布
            for i in range(num_classes):
                pred_distribution[i] += (predicted == i).sum().item()
            
            total_samples += predicted.size(0)
    
    # 归一化预测分布
    if total_samples > 0:
        pred_distribution = pred_distribution / total_samples
    
    return pred_distribution.cpu().numpy()


def extract_model_features(model, dataset, device, num_samples=500):
    """
    提取模型在指定数据集上的深层特征
    
    Args:
        model: 要评估的模型
        dataset: 评估数据集
        device: 计算设备
        num_samples: 采样数量
        
    Returns:
        模型特征（降维后）
    """
    # 创建数据加载器
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
    
    # 收集特征
    features_list = []
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            
            # 前向传播获取特征
            if hasattr(model, 'extract_features'):
                features = model.extract_features(data)
            else:
                # 如果模型实现了本地训练，应该返回特征
                outputs = model(data)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    features = outputs[1]  # 假设第二个输出是特征
                else:
                    continue  # 无法提取特征，跳过
            
            # 收集特征
            if features is not None:
                # 如果是卷积特征，先进行全局平均池化
                if len(features.shape) > 2:
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)
                
                features_list.append(features.cpu())
    
    # 合并所有特征
    if not features_list:
        return np.array([])
    
    all_features = torch.cat(features_list, dim=0).numpy()
    
    # 如果特征维度太高，使用PCA降维
    if all_features.shape[1] > 50:
        pca = PCA(n_components=50)
        all_features = pca.fit_transform(all_features)
    
    # 计算特征的平均值作为模型特征表示
    return np.mean(all_features, axis=0)


def calculate_model_similarity_matrix(client_features, similarity_metric='cosine'):
    """
    计算客户端模型之间的相似度矩阵
    
    Args:
        client_features: 客户端特征列表
        similarity_metric: 相似度度量方式
        
    Returns:
        相似度矩阵
    """
    n_clients = len(client_features)
    similarity_matrix = np.zeros((n_clients, n_clients))
    
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 自身相似度为1
                continue
            
            if similarity_metric == 'cosine':
                # 计算余弦相似度
                dot_product = np.dot(client_features[i], client_features[j])
                norm_i = np.linalg.norm(client_features[i])
                norm_j = np.linalg.norm(client_features[j])
                
                if norm_i > 0 and norm_j > 0:
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                else:
                    similarity_matrix[i, j] = 0
            
            elif similarity_metric == 'euclidean':
                # 计算欧氏距离的倒数作为相似度
                distance = np.linalg.norm(client_features[i] - client_features[j])
                similarity_matrix[i, j] = 1.0 / (1.0 + distance)
    
    return similarity_matrix


def cluster_clients(client_features, n_clusters=3, random_state=42):
    """
    使用KMeans聚类对客户端进行分组
    
    Args:
        client_features: 客户端特征列表
        n_clusters: 聚类数量
        random_state: 随机种子
        
    Returns:
        聚类标签, 聚类质量评分, 聚类中心
    """
    # 如果客户端数量小于聚类数量，调整聚类数量
    n_clients = len(client_features)
    if n_clients < n_clusters:
        n_clusters = max(2, n_clients)
    
    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(client_features)
    
    # 计算聚类质量（如果客户端数量足够）
    silhouette_avg = -1
    if n_clients > n_clusters:
        try:
            silhouette_avg = silhouette_score(client_features, cluster_labels)
        except:
            silhouette_avg = -1
    
    return cluster_labels, silhouette_avg, kmeans.cluster_centers_


def adaptive_clustering(client_models, eval_dataset, device, n_clusters=3, min_cluster_size=2):
    """
    自适应聚类算法，结合预测分布和模型特征进行聚类
    
    Args:
        client_models: 客户端模型列表
        eval_dataset: 评估数据集
        device: 计算设备
        n_clusters: 默认聚类数量
        min_cluster_size: 最小聚类大小
        
    Returns:
        客户端聚类标签, 聚类信息字典
    """
    n_clients = len(client_models)
    print(f"开始对 {n_clients} 个客户端模型进行自适应聚类")
    
    # 如果客户端数量太少，直接返回单一聚类
    if n_clients < min_cluster_size * n_clusters:
        n_clusters = max(1, n_clients // min_cluster_size)
        print(f"客户端数量较少，调整聚类数量为 {n_clusters}")
    
    # 1. 收集预测分布特征
    prediction_features = []
    for i, model in enumerate(client_models):
        try:
            pred_dist = extract_model_predictions(model, eval_dataset, device)
            prediction_features.append(pred_dist)
        except Exception as e:
            print(f"提取客户端 {i} 预测分布时出错: {str(e)}")
            # 使用均匀分布代替
            prediction_features.append(np.ones(10) / 10)
    
    # 2. 收集模型特征
    model_features = []
    for i, model in enumerate(client_models):
        try:
            features = extract_model_features(model, eval_dataset, device)
            model_features.append(features)
        except Exception as e:
            print(f"提取客户端 {i} 模型特征时出错: {str(e)}")
            # 使用零向量代替
            if len(model_features) > 0:
                model_features.append(np.zeros_like(model_features[0]))
            else:
                model_features.append(np.zeros(50))
    
    # 3. 组合特征
    combined_features = []
    for i in range(n_clients):
        # 确保特征向量有效
        if len(prediction_features[i]) > 0 and len(model_features[i]) > 0:
            # 处理预测特征，支持字典或数组格式
            if isinstance(prediction_features[i], dict) and 'prediction_counts' in prediction_features[i]:
                # 是字典格式，提取计数数组
                counts = prediction_features[i]['prediction_counts']
                if isinstance(counts, list):
                    counts = np.array(counts)
                pred_feat = counts / np.sum(counts) if np.sum(counts) > 0 else counts
            else:
                # 是数组格式，直接归一化
                pred_feat = prediction_features[i] / np.sum(prediction_features[i]) if np.sum(prediction_features[i]) > 0 else prediction_features[i]
            
            # 标准化模型特征
            if np.sum(np.abs(model_features[i])) > 0:
                model_feat = model_features[i] / np.linalg.norm(model_features[i])
            else:
                model_feat = model_features[i]
            
            # 组合特征 (预测分布权重更高)
            combined = np.concatenate([
                pred_feat * 0.7,  # 预测分布特征权重
                model_feat * 0.3   # 模型特征权重
            ])
            
            combined_features.append(combined)
        else:
            print(f"客户端 {i} 的特征无效，使用随机特征")
            # 创建随机特征
            combined = np.random.rand(60)  # 假设组合特征长度为60
            combined_features.append(combined)
    
    # 4. 执行聚类
    labels, score, centers = cluster_clients(combined_features, n_clusters)
    
    # 5. 评估并调整聚类
    cluster_sizes = {}
    for label in range(n_clusters):
        size = np.sum(labels == label)
        cluster_sizes[label] = size
        print(f"聚类 {label}: {size} 个客户端")
    
    # 如果有某些聚类过小，进行调整
    if min(cluster_sizes.values()) < min_cluster_size and n_clusters > 1:
        print("存在过小的聚类，尝试减少聚类数量")
        n_clusters = max(1, n_clusters - 1)
        labels, score, centers = cluster_clients(combined_features, n_clusters)
    
    # 6. 构建聚类信息
    cluster_info = {}
    for label in range(n_clusters):
        # 获取该聚类的客户端索引
        client_indices = np.where(labels == label)[0]
        
        # 收集该聚类的特征
        cluster_features = [combined_features[i] for i in client_indices]
        
        # 计算聚类内相似度
        if len(client_indices) > 1:
            similarity_matrix = calculate_model_similarity_matrix(cluster_features)
            avg_similarity = np.mean(similarity_matrix) - 1/len(client_indices)  # 排除对角线
        else:
            avg_similarity = 1.0
        
        # 存储聚类信息
        cluster_info[label] = {
            'client_indices': client_indices.tolist(),
            'size': len(client_indices),
            'center': centers[label],
            'avg_similarity': float(avg_similarity)
        }
    
    # 7. 打印聚类质量信息
    print(f"聚类质量评分 (silhouette): {score:.4f}")
    for label, info in cluster_info.items():
        print(f"聚类 {label}: {info['size']} 客户端, 平均相似度: {info['avg_similarity']:.4f}")
    
    return labels, cluster_info


def create_client_clusters_map(cluster_labels, client_indices):
    """
    创建客户端聚类映射字典
    
    Args:
        cluster_labels: 聚类标签
        client_indices: 客户端索引列表
        
    Returns:
        客户端聚类映射字典
    """
    client_clusters = {}
    
    # 初始化每个聚类的客户端列表
    for label in set(cluster_labels):
        client_clusters[label] = []
    
    # 将客户端分配到相应的聚类
    for i, label in enumerate(cluster_labels):
        if i < len(client_indices):
            client_idx = client_indices[i]
            client_clusters[label].append(client_idx)
    
    return client_clusters


def adaptive_cluster_assignment(client_models, client_indices, eval_dataset, device, 
                               n_clusters=3, stability_threshold=0.6):
    """
    自适应聚类分配，考虑聚类稳定性
    
    Args:
        client_models: 客户端模型列表
        client_indices: 客户端索引列表
        eval_dataset: 评估数据集
        device: 计算设备
        n_clusters: 聚类数量
        stability_threshold: 稳定性阈值
        
    Returns:
        客户端聚类映射
    """
    # 执行当前轮次的聚类
    current_labels, cluster_info = adaptive_clustering(
        client_models, eval_dataset, device, n_clusters
    )
    
    # 创建客户端聚类映射
    client_clusters = create_client_clusters_map(current_labels, client_indices)
    
    # 确保所有聚类至少有一个客户端
    empty_clusters = []
    for cluster_id in range(n_clusters):
        if cluster_id not in client_clusters or len(client_clusters[cluster_id]) == 0:
            empty_clusters.append(cluster_id)
    
    # 如果有空聚类，重新分配客户端
    if empty_clusters:
        print(f"发现 {len(empty_clusters)} 个空聚类，进行客户端重新分配")
        
        # 找出客户端最多的聚类
        largest_cluster_id = max(client_clusters.keys(), key=lambda k: len(client_clusters[k]))
        
        # 从最大聚类中分配一些客户端到空聚类
        for empty_id in empty_clusters:
            if largest_cluster_id in client_clusters and len(client_clusters[largest_cluster_id]) > 1:
                # 移动一个客户端到空聚类
                client_to_move = client_clusters[largest_cluster_id].pop()
                if empty_id not in client_clusters:
                    client_clusters[empty_id] = []
                client_clusters[empty_id].append(client_to_move)
                print(f"将客户端 {client_to_move} 从聚类 {largest_cluster_id} 移动到聚类 {empty_id}")
    
    return client_clusters
def extract_behavior_features(client_models, eval_dataset, device='cuda'):
    """
    提取客户端模型在统一评估数据集上的行为特征
    
    Args:
        client_models: 客户端模型列表
        eval_dataset: 用于测试模型行为的评估数据集
        device: 计算设备
        
    Returns:
        模型行为特征矩阵
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    behavior_features = []
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=32, shuffle=False
    )
    
    print(f"正在从{len(client_models)}个客户端模型中提取行为特征...")
    
    for i, model in enumerate(client_models):
        print(f"处理客户端 {i+1}/{len(client_models)}")
        model.to(device)
        model.eval()
        
        # 存储各层激活值统计信息
        activation_stats = {}
        
        # 存储预测分布信息
        all_predictions = []
        all_probs = []
        class_distribution = None
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                # 限制处理的批次数量以提高效率
                if batch_idx >= 10:
                    break
                    
                data, targets = data.to(device), targets.to(device)
                
                # 获取模型输出和各层激活
                try:
                    # 对于支持local_loss的模型
                    if hasattr(model, 'local_loss') and model.local_loss:
                        outputs, features = model(data)
                    else:
                        outputs = model(data)
                        features = None
                        
                    # 确保输出是tensor
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # 获取预测和概率
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    # 累积预测和概率
                    all_predictions.extend(preds.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                    
                    # 如果是首个批次，初始化类别分布计数
                    if class_distribution is None:
                        num_classes = outputs.size(1)
                        class_distribution = np.zeros(num_classes)
                    
                    # 更新类别分布
                    for cls in range(num_classes):
                        class_distribution[cls] += (preds == cls).sum().item()
                        
                    # 收集激活统计信息
                    if features is not None:
                        if isinstance(features, torch.Tensor):
                            # 单个特征张量的情况
                            feat_flat = features.view(features.size(0), -1)
                            
                            if 'feature_mean' not in activation_stats:
                                activation_stats['feature_mean'] = feat_flat.mean(dim=0).cpu().numpy()
                                activation_stats['feature_std'] = feat_flat.std(dim=0).cpu().numpy()
                            else:
                                # 增量更新统计信息
                                activation_stats['feature_mean'] = (activation_stats['feature_mean'] + 
                                                                    feat_flat.mean(dim=0).cpu().numpy()) / 2
                                activation_stats['feature_std'] = (activation_stats['feature_std'] + 
                                                                   feat_flat.std(dim=0).cpu().numpy()) / 2
                except Exception as e:
                    print(f"处理客户端{i}时出错: {str(e)}")
                    continue
        
        # 处理累积的行为特征
        if all_probs:
            # 合并所有批次的概率
            all_probs = np.vstack(all_probs)
            
            # 计算各种行为统计特征
            
            # 1. 预测分布特征
            pred_distribution = class_distribution / np.sum(class_distribution) if np.sum(class_distribution) > 0 else np.zeros_like(class_distribution)
            
            # 2. 预测置信度特征
            confidences = np.max(all_probs, axis=1)
            mean_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            
            # 3. 熵特征 (反映预测的不确定性)
            entropies = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            
            # 4. 边界距离特征 (top1和top2概率的差距)
            sorted_probs = np.sort(all_probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]  # top1 - top2
            mean_margin = np.mean(margins)
            std_margin = np.std(margins)
            
            # 组合所有行为特征
            client_features = np.concatenate([
                pred_distribution,  # 预测分布
                [mean_confidence, std_confidence],  # 置信度统计
                [mean_entropy, std_entropy],  # 熵统计
                [mean_margin, std_margin],  # 边界距离统计
            ])
            
            # 添加激活统计信息(如果有)
            if activation_stats:
                for key, value in activation_stats.items():
                    # 使用降维减少特征维度
                    if len(value) > 50:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=min(50, len(value) - 1))
                        try:
                            value_reduced = pca.fit_transform(value.reshape(1, -1))
                            value = value_reduced.flatten()
                        except:
                            # 如果PCA失败，只取前50个值
                            value = value[:50]
                    
                    client_features = np.concatenate([client_features, value])
        else:
            # 如果没有有效的预测结果，使用零向量
            # 估计一个合理的特征维度
            feature_dim = 50  # 默认特征维度
            if behavior_features:
                feature_dim = behavior_features[0].shape[0]
            client_features = np.zeros(feature_dim)
        
        # 确保所有特征向量维度一致
        if behavior_features and behavior_features[0].shape[0] != client_features.shape[0]:
            # 如果维度不一致，调整当前特征或之前的特征
            current_dim = client_features.shape[0]
            previous_dim = behavior_features[0].shape[0]
            
            if current_dim > previous_dim:
                # 截断当前特征
                client_features = client_features[:previous_dim]
            else:
                # 扩展当前特征
                client_features = np.pad(client_features, (0, previous_dim - current_dim))
                
            print(f"警告：调整客户端{i}的特征维度以匹配之前的特征")
                
        behavior_features.append(client_features)
    
    # 转换为numpy数组
    behavior_features = np.array(behavior_features)
    
    print(f"行为特征提取完成，特征维度: {behavior_features.shape}")
    
    return behavior_features


def create_proxy_dataset(client_datasets, sample_per_client=10):
    """
    从客户端数据集创建代理评估数据集
    
    Args:
        client_datasets: 客户端数据集列表
        sample_per_client: 每个客户端抽样的数据点数量
        
    Returns:
        代理评估数据集
    """
    import torch
    import random
    from torch.utils.data import TensorDataset, ConcatDataset
    
    proxy_datasets = []
    
    for dataset in client_datasets:
        # 随机抽样
        indices = random.sample(range(len(dataset)), min(sample_per_client, len(dataset)))
        
        samples = []
        labels = []
        
        # 收集数据和标签
        for idx in indices:
            data, label = dataset[idx]
            samples.append(data)
            labels.append(label)
        
        # 创建张量数据集
        if samples:
            samples_tensor = torch.stack(samples)
            labels_tensor = torch.tensor(labels)
            proxy_datasets.append(TensorDataset(samples_tensor, labels_tensor))
    
    # 合并所有代理数据集
    if proxy_datasets:
        return ConcatDataset(proxy_datasets)
    else:
        return None


def cluster_based_on_behavior(behavior_features, n_clusters=None, max_clusters=10):
    """
    基于行为特征对客户端进行聚类
    
    Args:
        behavior_features: 行为特征矩阵
        n_clusters: 指定的聚类数量(None表示自动确定)
        max_clusters: 最大聚类数量限制
        
    Returns:
        聚类标签, 聚类中心
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    import numpy as np
    
    # 数据规范化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(behavior_features)
    
    # 自动确定最佳聚类数量
    if n_clusters is None:
        min_clusters = 2
        max_clusters = min(max_clusters, features_scaled.shape[0] - 1)
        
        if max_clusters < min_clusters:
            # 如果样本太少，直接返回一个聚类
            return np.zeros(features_scaled.shape[0]), np.mean(features_scaled, axis=0).reshape(1, -1)
        
        best_score = -1
        best_n_clusters = min_clusters
        
        for n in range(min_clusters, max_clusters + 1):
            # 训练GMM模型
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=42,
                n_init=5
            )
            
            gmm.fit(features_scaled)
            labels = gmm.predict(features_scaled)
            
            # 计算聚类质量指标
            if len(set(labels)) > 1:
                try:
                    sil_score = silhouette_score(features_scaled, labels)
                    ch_score = calinski_harabasz_score(features_scaled, labels)
                    
                    # 组合分数 (同时考虑轮廓系数和Calinski-Harabasz指数)
                    combined_score = 0.6 * sil_score + 0.4 * (ch_score / 10000)  # 缩放CH分数
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_n_clusters = n
                except:
                    continue
        
        n_clusters = best_n_clusters
        print(f"自动确定的最佳聚类数量: {n_clusters}")
    
    # 使用高斯混合模型进行聚类
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=42,
        n_init=5
    )
    
    gmm.fit(features_scaled)
    labels = gmm.predict(features_scaled)
    
    # 提取聚类中心
    centers = gmm.means_
    
    # 评估聚类质量
    if len(set(labels)) > 1:
        try:
            sil_score = silhouette_score(features_scaled, labels)
            ch_score = calinski_harabasz_score(features_scaled, labels)
            
            print(f"聚类评估:")
            print(f"聚类数量: {len(set(labels))}")
            print(f"轮廓系数: {sil_score:.4f} (越高越好)")
            print(f"Calinski-Harabasz指数: {ch_score:.4f} (越高越好)")
        except Exception as e:
            print(f"计算聚类质量时出错: {str(e)}")
    
    # 打印每个聚类的样本数
    unique_labels = set(labels)
    print("\n聚类分布:")
    for label in unique_labels:
        count = np.sum(labels == label)
        percentage = count / len(labels) * 100
        print(f"聚类 {label}: {count} 客户端 ({percentage:.2f}%)")
    
    return labels, centers


def data_distribution_aware_clustering(client_models, client_datasets=None, eval_dataset=None, 
                                      n_clusters=None, device='cuda'):
    """
    数据分布感知的客户端聚类主函数
    
    Args:
        client_models: 客户端模型列表
        client_datasets: 客户端数据集列表(用于创建代理数据集)
        eval_dataset: 外部评估数据集(如果没有则由client_datasets创建)
        n_clusters: 指定的聚类数量(None表示自动确定)
        device: 计算设备
        
    Returns:
        聚类标签, 聚类信息
    """
    import numpy as np
    
    # 1. 准备评估数据集
    if eval_dataset is None and client_datasets is not None:
        print("正在从客户端数据集创建代理评估数据集...")
        eval_dataset = create_proxy_dataset(client_datasets)
    
    if eval_dataset is None:
        raise ValueError("必须提供eval_dataset或client_datasets以创建评估数据集")
    
    # 2. 提取行为特征
    print("正在提取客户端模型的行为特征...")
    behavior_features = extract_behavior_features(client_models, eval_dataset, device)
    
    # 3. 基于行为特征进行聚类
    print("正在执行数据分布感知的聚类...")
    labels, centers = cluster_based_on_behavior(behavior_features, n_clusters)
    
    # 4. 创建聚类分配信息
    cluster_assignments = {}
    for i, label in enumerate(labels):
        if label not in cluster_assignments:
            cluster_assignments[label] = []
        cluster_assignments[label].append(i)
    
    # 5. 针对每个聚类分析tier分布情况
    if hasattr(client_models[0], 'tier'):
        print("\n各聚类的tier分布:")
        for cluster_id, client_indices in cluster_assignments.items():
            tier_counts = {}
            for idx in client_indices:
                tier = client_models[idx].tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            print(f"聚类 {cluster_id} tier分布: {tier_counts}")
    
    return labels, cluster_assignments


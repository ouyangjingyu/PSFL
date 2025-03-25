import torch
import numpy as np
from collections import OrderedDict

def balance_classifier_weights_enhanced(model, num_classes=10, max_ratio=2.0):
    """
    增强版分类器权重平衡函数，解决类别不平衡问题
    
    Args:
        model: 模型参数字典
        num_classes: 类别数量
        max_ratio: 最大允许的权重范数比率
        
    Returns:
        平衡后的模型参数
    """
    # 查找分类器层
    classifier_keys = []
    for key in model.keys():
        if ('classifier.fc3.weight' in key or 'classifier.fc.weight' in key or 
            'classifier.weight' in key or 'fc.weight' in key):
            if isinstance(model[key], torch.Tensor) and model[key].size(0) == num_classes:
                classifier_keys.append(key)
    
    # 如果找到分类器层，执行权重平衡
    for key in classifier_keys:
        weight = model[key]
        
        # 计算每个类别的权重范数
        class_norms = torch.norm(weight, dim=1)
        mean_norm = torch.mean(class_norms)
        
        # 检测异常值
        max_norm = torch.max(class_norms)
        min_norm = torch.min(class_norms)
        
        # 计算范数比率
        norm_ratio = max_norm / min_norm if min_norm > 0 else float('inf')
        
        print(f"分类器权重范数 - 最大: {max_norm:.4f}, 最小: {min_norm:.4f}, 平均: {mean_norm:.4f}, 比率: {norm_ratio:.4f}")
        
        # 如果比率超过阈值，进行平衡
        if norm_ratio > max_ratio:
            print(f"检测到分类器权重不平衡 (比率 {norm_ratio:.2f} > {max_ratio})，执行平衡...")
            
            # 方法1: 直接归一化所有类别权重到均值附近
            weight_balanced = weight.clone()
            for i in range(num_classes):
                if class_norms[i] > 0:  # 避免除以零
                    # 将范数调整到均值的0.8-1.2倍之间
                    target_norm = mean_norm * (0.8 + 0.4 * (i / (num_classes - 1)))
                    scale_factor = target_norm / class_norms[i]
                    weight_balanced[i] = weight[i] * scale_factor
            
            # 更新模型参数
            model[key] = weight_balanced
            
            # 验证平衡后的范数
            new_norms = torch.norm(weight_balanced, dim=1)
            new_ratio = torch.max(new_norms) / torch.min(new_norms) if torch.min(new_norms) > 0 else float('inf')
            print(f"平衡后的分类器权重范数比率: {new_ratio:.4f}")
    
    # 检查并平衡最后一个隐藏层
    fc2_keys = []
    for key in model.keys():
        if 'classifier.fc2.weight' in key or 'fc2.weight' in key:
            fc2_keys.append(key)
    
    for key in fc2_keys:
        weight = model[key]
        # 计算输出维度的范数
        output_norms = torch.norm(weight, dim=1)
        mean_norm = torch.mean(output_norms)
        max_norm = torch.max(output_norms)
        min_norm = torch.min(output_norms)
        
        # 如果最大最小相差太大，也进行平衡
        if max_norm > min_norm * max_ratio and min_norm > 0:
            print(f"检测到隐藏层权重不平衡，执行平衡...")
            weight_balanced = weight.clone()
            for i in range(weight.size(0)):
                if output_norms[i] > 0:
                    scale_factor = mean_norm / output_norms[i]
                    weight_balanced[i] = weight[i] * scale_factor
            
            model[key] = weight_balanced
    
    return model

def normalize_batch_norm_stats(model):
    """
    规范化批标准化层的统计信息，确保它们在合理范围内
    
    Args:
        model: 模型参数字典
        
    Returns:
        规范化后的模型参数
    """
    bn_count = 0
    
    for key in list(model.keys()):
        # 处理运行均值
        if 'running_mean' in key:
            running_mean = model[key]
            # 限制值范围
            clamped_mean = torch.clamp(running_mean, min=-2.0, max=2.0)
            if not torch.allclose(running_mean, clamped_mean):
                bn_count += 1
                model[key] = clamped_mean
        
        # 处理运行方差
        elif 'running_var' in key:
            running_var = model[key]
            # 确保方差为正且不太大
            clamped_var = torch.clamp(running_var, min=0.01, max=5.0)
            if not torch.allclose(running_var, clamped_var):
                bn_count += 1
                model[key] = clamped_var
    
    if bn_count > 0:
        print(f"规范化了 {bn_count} 个批标准化层参数")
    
    return model

def normalize_feature_scales(client_model, server_model):
    """
    规范化客户端和服务器模型之间的特征尺度
    
    Args:
        client_model: 客户端模型参数
        server_model: 服务器模型参数
        
    Returns:
        规范化后的客户端和服务器模型参数
    """
    # 寻找客户端模型最后一层卷积/线性层的权重尺度
    client_last_layer_key = None
    client_last_layer_scale = 1.0
    
    for key in client_model.keys():
        if ('layer' in key and 'weight' in key and 'conv' in key) or 'projection.weight' in key:
            client_last_layer_key = key
    
    if client_last_layer_key:
        client_last_layer = client_model[client_last_layer_key]
        client_last_layer_scale = torch.norm(client_last_layer).item()
    
    # 寻找服务器模型第一层卷积/线性层的权重尺度
    server_first_layer_key = None
    server_first_layer_scale = 1.0
    
    for key in server_model.keys():
        if ('layer' in key and 'weight' in key and 'conv' in key):
            server_first_layer_key = key
            break
    
    if server_first_layer_key:
        server_first_layer = server_model[server_first_layer_key]
        server_first_layer_scale = torch.norm(server_first_layer).item()
    
    # 如果尺度差异过大，调整服务器模型的第一层
    if server_first_layer_key and client_last_layer_scale > 0 and server_first_layer_scale > 0:
        scale_ratio = server_first_layer_scale / client_last_layer_scale
        
        if scale_ratio > 5.0 or scale_ratio < 0.2:
            target_ratio = 1.0
            adjust_factor = target_ratio / scale_ratio
            
            print(f"特征尺度不匹配 (比率 {scale_ratio:.2f})，调整服务器模型第一层...")
            server_model[server_first_layer_key] = server_model[server_first_layer_key] * adjust_factor
    
    return client_model, server_model

def enhanced_hierarchical_aggregation_improved(client_models, client_weights, cluster_map, client_tiers=None,
                                     global_model_template=None, num_classes=10):
    """修改后的层次聚合函数，针对特征尺度和类别平衡问题进行了增强"""
    
    from memory_utils import free_memory
    
    # 初始化日志信息
    log_info = {
        'cluster_stats': {},
        'repair_stats': {},
        'aggregation_stats': {}
    }
    
    # 执行双层聚合前过滤掉projection相关参数
    filtered_client_models = {}
    for client_id, model_state in client_models.items():
        filtered_state = {k: v for k, v in model_state.items() 
                         if 'projection' not in k}
        filtered_client_models[client_id] = filtered_state
    
    # 1. 聚类统计信息
    for cluster_id, client_ids in cluster_map.items():
        log_info['cluster_stats'][f'cluster_{cluster_id}'] = {
            'size': len(client_ids),
            'clients': client_ids
        }
        
        # 记录tier分布
        if client_tiers:
            tier_dist = {}
            for cid in client_ids:
                if cid in client_tiers:
                    tier = client_tiers[cid]
                    tier_dist[tier] = tier_dist.get(tier, 0) + 1
            log_info['cluster_stats'][f'cluster_{cluster_id}']['tier_distribution'] = tier_dist

    # 2. 执行聚类内聚合 - 使用改进的聚合策略
    cluster_models = {}
    
    for cluster_id, client_ids in cluster_map.items():
        # 收集该聚类中的客户端模型
        cluster_client_models = {}
        cluster_client_weights = {}
        
        for client_id in client_ids:
            if client_id in filtered_client_models:
                cluster_client_models[client_id] = filtered_client_models[client_id]
                cluster_client_weights[client_id] = client_weights.get(client_id, 1.0)
        
        if not cluster_client_models:
            continue
        
        # 归一化权重
        total_weight = sum(cluster_client_weights.values())
        if total_weight > 0:
            for client_id in cluster_client_weights:
                cluster_client_weights[client_id] /= total_weight
        
        # 找出这个聚类中的客户端tier
        if client_tiers:
            cluster_tiers = [client_tiers.get(client_id, 7) for client_id in cluster_client_models.keys()]
            min_tier = min(cluster_tiers)
            print(f"聚类 {cluster_id} 的最小tier: {min_tier}")
        
        # 创建聚合模型模板
        aggregated_model = OrderedDict()
        
        # 首先，收集所有键并确保它们存在于聚合模型中
        all_keys = set()
        for client_model in cluster_client_models.values():
            all_keys.update(client_model.keys())
        
        # 预填充聚合模型
        reference_client_id = list(cluster_client_models.keys())[0]
        reference_model = cluster_client_models[reference_client_id]
        
        for key in all_keys:
            if key in reference_model:
                aggregated_model[key] = torch.zeros_like(reference_model[key])
            else:
                # 对于其他客户端有但参考客户端没有的键，查找任何有这个键的客户端
                for client_id, model in cluster_client_models.items():
                    if key in model:
                        aggregated_model[key] = torch.zeros_like(model[key])
                        break
        
        # 分别聚合特征提取层和分类器层
        feature_keys = [k for k in all_keys if 'classifier' not in k and 'fc' not in k]
        classifier_keys = [k for k in all_keys if 'classifier' in k or 'fc' in k]
        
        # 1. 先聚合特征提取层
        for key in feature_keys:
            # 记录贡献这个参数的客户端数量和总权重
            contributors = 0
            total_key_weight = 0.0
            
            # 加权聚合
            for client_id, model in cluster_client_models.items():
                if key in model:
                    weight = cluster_client_weights[client_id]
                    
                    # 针对BN层的特殊处理
                    if 'running_mean' in key or 'running_var' in key:
                        # BN层统计量使用最近更新的客户端的值
                        if contributors == 0:
                            aggregated_model[key] = model[key].clone()
                    else:
                        # 普通层使用加权平均
                        aggregated_model[key] += model[key].to(dtype=torch.float32) * weight
                    
                    contributors += 1
                    total_key_weight += weight
            
            # 对于没有任何贡献的参数，保持零
            if contributors == 0:
                continue
                
            # 对于BN层以外的参数，如果总权重不为1，重新归一化
            if 'running_mean' not in key and 'running_var' not in key and total_key_weight > 0:
                if not torch.isclose(torch.tensor(total_key_weight), torch.tensor(1.0)):
                    aggregated_model[key] = aggregated_model[key] / total_key_weight
        
        # 2. 再聚合分类器层，应用特殊处理以平衡不同类别
        for key in classifier_keys:
            contributors = 0
            total_key_weight = 0.0
            
            for client_id, model in cluster_client_models.items():
                if key in model:
                    weight = cluster_client_weights[client_id]
                    aggregated_model[key] += model[key].to(dtype=torch.float32) * weight
                    contributors += 1
                    total_key_weight += weight
            
            if contributors == 0:
                continue
                
            if total_key_weight > 0:
                if not torch.isclose(torch.tensor(total_key_weight), torch.tensor(1.0)):
                    aggregated_model[key] = aggregated_model[key] / total_key_weight
        
        # 3. 应用批归一化层统计量规范化
        aggregated_model = normalize_batch_norm_stats(aggregated_model)
        
        # 4. 应用类别权重平衡，解决类别偏向问题
        aggregated_model = balance_classifier_weights_enhanced(aggregated_model, num_classes)
        
        # 存储聚类聚合结果
        cluster_models[cluster_id] = aggregated_model
        
        # 打印聚合统计信息
        print(f"聚类 {cluster_id} 聚合结果: 包含 {len(aggregated_model)} 个参数")
        
        # 释放内存
        free_memory()
    
    # 全局聚合
    global_model = OrderedDict()
    
    # 如果提供了全局模型模板，使用它初始化参数
    if global_model_template is not None:
        if isinstance(global_model_template, dict):
            template_dict = global_model_template
        else:
            template_dict = global_model_template.state_dict()
        
        # 复制模板
        for key, param in template_dict.items():
            global_model[key] = param.clone()
    
    # 合并所有聚类模型创建全局模型
    # 首先创建键集合
    all_keys = set()
    for cluster_id, model in cluster_models.items():
        all_keys.update(model.keys())
    
    # 区分特征层和分类器层
    feature_keys = [k for k in all_keys if 'classifier' not in k and 'fc' not in k]
    classifier_keys = [k for k in all_keys if 'classifier' in k or 'fc' in k]
    
    # 聚合特征层
    for key in feature_keys:
        contributors = []
        weights = []
        
        for cluster_id, model in cluster_models.items():
            if key in model:
                contributors.append(model[key])
                # 可以根据聚类大小来调整权重
                cluster_size = len(cluster_map.get(cluster_id, []))
                weights.append(cluster_size)
        
        if not contributors:
            continue
            
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
        # 聚合参数
        if 'running_mean' in key or 'running_var' in key:
            # 对于BN层统计量，使用中位数或最近更新的值
            global_model[key] = contributors[0].clone()
        else:
            # 对于其他参数，使用加权平均
            global_model[key] = torch.zeros_like(contributors[0])
            for param, weight in zip(contributors, weights):
                global_model[key] += param.to(dtype=torch.float32) * weight
    
    # 聚合分类器层，额外应用平衡
    for key in classifier_keys:
        contributors = []
        weights = []
        
        for cluster_id, model in cluster_models.items():
            if key in model:
                contributors.append(model[key])
                cluster_size = len(cluster_map.get(cluster_id, []))
                weights.append(cluster_size)
        
        if not contributors:
            continue
            
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
        # 聚合参数
        global_model[key] = torch.zeros_like(contributors[0])
        for param, weight in zip(contributors, weights):
            global_model[key] += param.to(dtype=torch.float32) * weight
    
    # 最终的批归一化层规范化
    global_model = normalize_batch_norm_stats(global_model)
    
    # 最终的类别权重平衡，确保全局模型不会偏向某些类别
    global_model = balance_classifier_weights_enhanced(global_model, num_classes, max_ratio=1.5)
    
    # 如果提供了全局模型模板，填充缺失的参数
    if global_model_template is not None:
        keys_from_template = 0
        for key, param in template_dict.items():
            if key not in global_model:
                global_model[key] = param.clone()
                keys_from_template += 1
        
        log_info['aggregation_stats']['template_keys_used'] = keys_from_template
    
    # 打印全局聚合统计信息
    feature_count = len([k for k in global_model.keys() if 'classifier' not in k and 'fc' not in k])
    print(f"全局聚合模型: 聚合了 {feature_count} 个特征提取层参数，"
          f"{len(global_model) - feature_count} 个分类器参数")
    
    # 释放内存
    free_memory()
    
    return global_model, cluster_models, log_info
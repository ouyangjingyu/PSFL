import torch
import copy
import numpy as np
from collections import OrderedDict
import torch.nn as nn


def cluster_aware_aggregation(client_models, client_weights, cluster_map):
    """
    聚类内部聚合方法，为每个聚类生成代表性模型
    
    Args:
        client_models: 客户端模型参数字典，键为客户端ID，值为模型参数
        client_weights: 客户端权重字典，键为客户端ID，值为权重
        cluster_map: 聚类映射，键为聚类ID，值为客户端ID列表
        
    Returns:
        聚类聚合结果字典，键为聚类ID，值为聚合后的模型参数
    """
    # 初始化聚类聚合结果
    cluster_models = {}
    
    # 处理每个聚类
    for cluster_id, client_ids in cluster_map.items():
        if not client_ids:
            continue
            
        # 收集聚类内客户端模型和权重
        cluster_client_models = [client_models[cid] for cid in client_ids if cid in client_models]
        cluster_client_weights = [client_weights.get(cid, 1.0) for cid in client_ids if cid in client_models]
        
        if not cluster_client_models:
            continue
            
        # 归一化权重
        weight_sum = sum(cluster_client_weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in cluster_client_weights]
        else:
            normalized_weights = [1.0 / len(cluster_client_weights)] * len(cluster_client_weights)
        
        # 执行聚合 - 修改对同构模型的要求
        # 在这里，我们假设每个聚类内的客户端可能有相同的tier和模型结构
        # 先使用第一个客户端的模型结构作为基础
        base_model = cluster_client_models[0]
        aggregated_model = OrderedDict()
        
        # 遍历基础模型的键
        for key in base_model.keys():
            # 收集所有拥有这个键的模型的参数
            params_list = []
            params_weights = []
            
            for idx, model in enumerate(cluster_client_models):
                if key in model and model[key].shape == base_model[key].shape:
                    params_list.append(model[key])
                    params_weights.append(normalized_weights[idx])
            
            # 如果没有匹配的参数，使用基础模型的参数
            if not params_list:
                aggregated_model[key] = base_model[key].clone()
                continue
            
            # 重新归一化权重
            weight_sum = sum(params_weights)
            if weight_sum > 0:
                params_weights = [w / weight_sum for w in params_weights]
            else:
                params_weights = [1.0 / len(params_weights)] * len(params_weights)
            
            # 聚合参数
            weighted_param = params_list[0].clone() * 0.0
            for param, weight in zip(params_list, params_weights):
                weighted_param += param.to(dtype=torch.float32) * weight
            
            aggregated_model[key] = weighted_param
        
        # 存储聚类聚合结果
        cluster_models[cluster_id] = aggregated_model
    
    return cluster_models


def global_unified_aggregation(cluster_models, cluster_weights):
    """
    全局统一聚合方法，整合所有聚类模型
    
    Args:
        cluster_models: 聚类模型字典，键为聚类ID，值为模型参数
        cluster_weights: 聚类权重字典，键为聚类ID，值为权重
        
    Returns:
        全局聚合模型参数
    """
    # 收集聚类模型和权重
    models_list = [model for _, model in cluster_models.items()]
    weights_list = [cluster_weights.get(cid, 1.0) for cid in cluster_models.keys()]
    
    # 归一化权重
    weight_sum = sum(weights_list)
    if weight_sum > 0:
        normalized_weights = [w / weight_sum for w in weights_list]
    else:
        normalized_weights = [1.0 / len(weights_list)] * len(weights_list)
    
    # 执行全局聚合，保留分类器
    global_model = aggregate_models(
        models_list, 
        normalized_weights,
        special_bn_treatment=True,
        preserve_classifier=True
    )
    
    return global_model


def aggregate_models(model_list, weights, special_bn_treatment=True, preserve_classifier=False):
    """
    聚合多个模型参数，可选择性地对BatchNorm层和分类器层进行特殊处理
    
    Args:
        model_list: 模型参数列表
        weights: 归一化权重列表
        special_bn_treatment: 是否对BatchNorm层进行特殊处理
        preserve_classifier: 是否保留最好模型的分类器参数
        
    Returns:
        聚合后的模型参数
    """
    if not model_list:
        return None
    
    # 初始化聚合模型 - 使用第一个模型作为基础
    base_model = model_list[0]
    aggregated_model = OrderedDict()
    
    # 遍历基础模型的所有键
    for key in base_model.keys():
        # 跳过特殊参数（如num_batches_tracked）
        if 'num_batches_tracked' in key:
            aggregated_model[key] = base_model[key].clone()
            continue
        
        # 收集所有相同形状的参数
        param_values = []
        param_weights = []
        
        for i, model in enumerate(model_list):
            if key in model and model[key].shape == base_model[key].shape:
                param_values.append(model[key])
                param_weights.append(weights[i])
        
        # 如果没有匹配的参数，使用基础模型的参数
        if not param_values:
            aggregated_model[key] = base_model[key].clone()
            continue
        
        # 归一化权重
        weight_sum = sum(param_weights)
        if weight_sum > 0:
            param_weights = [w / weight_sum for w in param_weights]
        else:
            param_weights = [1.0 / len(param_weights)] * len(param_weights)
        
        # 特殊处理BatchNorm层
        if special_bn_treatment and ('running_mean' in key or 'running_var' in key):
            # 对于BN层，使用中位数或限制范围
            if 'running_mean' in key:
                # 对均值使用中位数
                stacked_values = torch.stack([v.to(dtype=torch.float32) for v in param_values])
                median_value, _ = torch.median(stacked_values, dim=0)
                aggregated_model[key] = torch.clamp(median_value, min=-2.0, max=2.0)
            elif 'running_var' in key:
                # 对方差使用加权平均后限制范围
                weighted_avg = torch.zeros_like(param_values[0], dtype=torch.float32)
                for v, w in zip(param_values, param_weights):
                    weighted_avg += v.to(dtype=torch.float32) * w
                aggregated_model[key] = torch.clamp(weighted_avg, min=0.01, max=5.0)
        else:
            # 普通参数使用加权平均
            weighted_avg = torch.zeros_like(param_values[0], dtype=torch.float32)
            for v, w in zip(param_values, param_weights):
                weighted_avg += v.to(dtype=torch.float32) * w
            aggregated_model[key] = weighted_avg
    
    # 如果保留分类器，使用第一个模型的分类器参数
    if preserve_classifier:
        for key in base_model.keys():
            if 'classifier' in key or 'fc' in key:
                aggregated_model[key] = base_model[key].clone()
    
    return aggregated_model


def hybrid_aggregation_strategy(client_models, client_weights, cluster_map, cluster_weights=None):
    """
    混合聚合策略，实现双层聚合
    
    Args:
        client_models: 客户端模型参数字典
        client_weights: 客户端权重字典
        cluster_map: 聚类映射
        cluster_weights: 聚类权重字典，如果为None则使用均等权重
        
    Returns:
        全局聚合模型参数，聚类聚合模型字典
    """
    # 1. 执行聚类内部聚合
    cluster_models = cluster_aware_aggregation(
        client_models, client_weights, cluster_map
    )
    
    # 2. 如果没有提供聚类权重，创建均等权重
    if cluster_weights is None:
        # 基于聚类大小设置权重
        cluster_weights = {}
        total_clients = sum(len(clients) for clients in cluster_map.values())
        
        if total_clients > 0:
            for cluster_id, clients in cluster_map.items():
                cluster_weights[cluster_id] = len(clients) / total_clients
        else:
            # 均等权重
            n_clusters = len(cluster_map)
            for cluster_id in cluster_map.keys():
                cluster_weights[cluster_id] = 1.0 / max(1, n_clusters)
    
    # 3. 执行全局统一聚合
    global_model = global_unified_aggregation(
        cluster_models, cluster_weights
    )
    
    return global_model, cluster_models


def balance_classifier_weights(model, num_classes=10):
    """
    平衡分类器权重，确保不同类别的决策边界均衡
    
    Args:
        model: 模型参数
        num_classes: 类别数量
        
    Returns:
        平衡后的模型参数
    """
    # 查找分类器层
    classifier_keys = []
    for key in model.keys():
        if any(x in key for x in ['classifier', 'fc']) and 'weight' in key:
            if model[key].shape[0] == num_classes:
                classifier_keys.append(key)
    
    # 如果找到分类器层，执行权重平衡
    for key in classifier_keys:
        weight = model[key]
        
        # 计算每个类别的权重范数
        class_norms = torch.norm(weight, dim=1)
        mean_norm = torch.mean(class_norms)
        
        # 如果存在显著差异，进行平衡
        max_norm = torch.max(class_norms)
        min_norm = torch.min(class_norms)
        
        if max_norm > min_norm * 2:
            # 平衡权重，使所有类别的范数接近均值
            weight_balanced = weight.clone()
            for i in range(num_classes):
                if class_norms[i] > 0:  # 避免除以零
                    scale_factor = mean_norm / class_norms[i]
                    weight_balanced[i] = weight[i] * scale_factor
            
            # 更新模型参数
            model[key] = weight_balanced
    
    return model


def validate_model_parameters(model):
    """
    验证模型参数，检查并修复NaN或Inf值
    
    Args:
        model: 模型参数
        
    Returns:
        验证后的模型参数，修复计数
    """
    fixed_count = 0
    
    for key, param in model.items():
        # 检查NaN值
        nan_mask = torch.isnan(param)
        if nan_mask.any():
            # 将NaN替换为0
            param[nan_mask] = 0.0
            fixed_count += nan_mask.sum().item()
        
        # 检查Inf值
        inf_mask = torch.isinf(param)
        if inf_mask.any():
            # 将Inf替换为相应的大值（正无穷或负无穷）
            large_val = 1e6
            pos_inf_mask = (param == float('inf'))
            neg_inf_mask = (param == float('-inf'))
            param[pos_inf_mask] = large_val
            param[neg_inf_mask] = -large_val
            fixed_count += inf_mask.sum().item()
    
    return model, fixed_count


def enhanced_hierarchical_aggregation(client_models, client_weights, cluster_map, 
                                     global_model_template=None, num_classes=10):
    """
    增强的分层聚合策略，包含参数验证和分类器平衡
    
    Args:
        client_models: 客户端模型参数字典
        client_weights: 客户端权重字典
        cluster_map: 聚类映射
        global_model_template: 全局模型模板，用于确保参数完整性
        num_classes: 类别数量
        
    Returns:
        全局聚合模型，聚类模型字典，日志信息
    """
    log_info = {
        'cluster_stats': {},
        'repair_stats': {},
        'aggregation_stats': {}
    }
    
    # 1. 聚类统计信息
    for cluster_id, client_ids in cluster_map.items():
        log_info['cluster_stats'][f'cluster_{cluster_id}'] = {
            'size': len(client_ids),
            'clients': client_ids
        }
    
    # 2. 执行双层聚合
    global_model, cluster_models = hybrid_aggregation_strategy(
        client_models, client_weights, cluster_map
    )
    
    # 3. 验证和修复全局模型参数
    global_model, fixed_count = validate_model_parameters(global_model)
    log_info['repair_stats']['global_fixed_count'] = fixed_count
    
    # 4. 平衡分类器权重
    global_model = balance_classifier_weights(global_model, num_classes)
    
    # 5. 如果提供了全局模型模板，确保参数完整性
    if global_model_template is not None:
        # 复制模板中存在但聚合结果中不存在的参数
        for key, param in global_model_template.items():
            if key not in global_model:
                global_model[key] = param.clone()
        
        log_info['aggregation_stats']['template_keys_used'] = len(global_model_template.keys()) - len(global_model.keys())
    
    # 6. 验证和修复聚类模型参数
    for cluster_id, model in cluster_models.items():
        cluster_models[cluster_id], fixed_count = validate_model_parameters(model)
        log_info['repair_stats'][f'cluster_{cluster_id}_fixed_count'] = fixed_count
    
    return global_model, cluster_models, log_info


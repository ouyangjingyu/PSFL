import torch
import copy
import numpy as np
from collections import OrderedDict
import torch.nn as nn


def cluster_aware_aggregation(client_models, client_weights, cluster_map, client_tiers=None):
    """
    改进的聚类内部聚合方法，基于tier最小的客户端模型结构
    
    Args:
        client_models: 客户端模型参数字典，键为客户端ID，值为模型参数
        client_weights: 客户端权重字典，键为客户端ID，值为权重
        cluster_map: 聚类映射，键为聚类ID，值为客户端ID列表
        client_tiers: 客户端tier信息字典，键为客户端ID，值为tier级别
        
    Returns:
        聚类聚合结果字典，键为聚类ID，值为聚合后的模型参数
    """
    # 初始化聚类聚合结果
    cluster_models = {}
    
    # 处理每个聚类
    for cluster_id, client_ids in cluster_map.items():
        if not client_ids:
            continue
            
        # 收集聚类内客户端模型、权重和tier信息
        cluster_client_data = []
        for cid in client_ids:
            if cid in client_models:
                tier = client_tiers.get(cid, 9999) if client_tiers else 9999  # 默认高tier
                cluster_client_data.append({
                    'id': cid,
                    'model': client_models[cid],
                    'weight': client_weights.get(cid, 1.0),
                    'tier': tier,
                    'layer_count': len(set(k.split('.')[0] for k in client_models[cid].keys()))
                })
        
        if not cluster_client_data:
            continue
        
        # 按tier从小到大排序（或者按层数从多到少排序，如果没有tier信息）
        if client_tiers:
            cluster_client_data.sort(key=lambda x: x['tier'])
        else:
            cluster_client_data.sort(key=lambda x: x['layer_count'], reverse=True)
        
        # 选择tier最小（层数最多）的客户端模型作为基础
        base_client_data = cluster_client_data[0]
        base_model = base_client_data['model']
        
        # 创建包含所有可能层的合并键集合
        all_keys = set()
        for client_data in cluster_client_data:
            all_keys.update(client_data['model'].keys())
        
        # 归一化所有客户端权重
        total_weight = sum(client_data['weight'] for client_data in cluster_client_data)
        if total_weight > 0:
            for client_data in cluster_client_data:
                client_data['norm_weight'] = client_data['weight'] / total_weight
        else:
            equal_weight = 1.0 / len(cluster_client_data)
            for client_data in cluster_client_data:
                client_data['norm_weight'] = equal_weight
        
        # 执行聚合，遍历所有可能的键
        aggregated_model = OrderedDict()
        
        for key in sorted(all_keys):  # 排序以保持一致性
            # 收集拥有此键且形状匹配的所有模型参数
            compatible_params = []
            compatible_weights = []
            
            # 检查基础模型中是否存在该键，确定参考形状
            reference_shape = None
            if key in base_model:
                reference_shape = base_model[key].shape
            
            # 收集所有兼容参数
            for client_data in cluster_client_data:
                client_model = client_data['model']
                if key in client_model:
                    # 如果尚未确定参考形状，使用当前模型的形状
                    if reference_shape is None:
                        reference_shape = client_model[key].shape
                    
                    # 仅收集形状匹配的参数
                    if client_model[key].shape == reference_shape:
                        compatible_params.append(client_model[key])
                        compatible_weights.append(client_data['norm_weight'])
            
            # 如果没有兼容参数（不应该发生，但为安全起见）
            if not compatible_params:
                continue
                
            # 如果只有一个客户端有该参数，直接使用
            if len(compatible_params) == 1:
                aggregated_model[key] = compatible_params[0].clone()
                continue
            
            # 多个客户端有该参数，执行加权平均
            # 重新归一化权重
            weight_sum = sum(compatible_weights)
            if weight_sum > 0:
                compatible_weights = [w / weight_sum for w in compatible_weights]
            else:
                compatible_weights = [1.0 / len(compatible_weights)] * len(compatible_weights)
            
            # 聚合参数
            weighted_param = torch.zeros_like(compatible_params[0], dtype=torch.float32)
            for param, weight in zip(compatible_params, compatible_weights):
                weighted_param += param.to(dtype=torch.float32) * weight
            
            aggregated_model[key] = weighted_param
        
        # 存储聚类聚合结果
        cluster_models[cluster_id] = aggregated_model
        
        # 打印聚合统计信息（可选）
        base_keys = set(base_model.keys())
        agg_keys = set(aggregated_model.keys())
        print(f"聚类 {cluster_id} 聚合结果: 基础模型键数={len(base_keys)}, 聚合模型键数={len(agg_keys)}")
        print(f"  - 额外包含的键数: {len(agg_keys - base_keys)}")
        
    return cluster_models


def global_unified_aggregation(cluster_models, cluster_weights):
    """
    改进的全局统一聚合方法，整合所有聚类模型的所有特征提取层
    
    Args:
        cluster_models: 聚类模型字典，键为聚类ID，值为模型参数
        cluster_weights: 聚类权重字典，键为聚类ID，值为权重
        
    Returns:
        全局聚合模型参数
    """
    # 收集聚类模型和权重
    models_list = [model for _, model in cluster_models.items()]
    weights_list = [cluster_weights.get(cid, 1.0) for cid in cluster_models.keys()]

    # 检查是否有模型可聚合
    if not models_list:
        print("警告: 没有有效的模型可聚合!")
        return None
    
    # 归一化权重
    weight_sum = sum(weights_list)
    if weight_sum > 0:
        normalized_weights = [w / weight_sum for w in weights_list]
    else:
        normalized_weights = [1.0 / max(1, len(weights_list))] * len(weights_list)
    
    # 创建包含所有可能层的合并键集合
    all_keys = set()
    for model in models_list:
        all_keys.update(model.keys())
    
    # 初始化聚合模型
    aggregated_model = OrderedDict()
    
    # 选择目标设备
    target_device = torch.device('cpu')
    
    # 遍历所有可能的键
    for key in sorted(all_keys):  # 排序以保持一致性
        # 跳过分类器相关层
        if 'classifier' in key or 'fc' in key:
            continue
            
        # 跳过特殊参数
        if 'num_batches_tracked' in key:
            # 使用第一个包含此键的模型
            for model in models_list:
                if key in model:
                    aggregated_model[key] = model[key].clone()
                    break
            continue
        
        # 收集拥有此键的所有模型参数
        compatible_params = []
        compatible_weights = []
        reference_shape = None
        
        # 先确定参考形状
        for model in models_list:
            if key in model:
                reference_shape = model[key].shape
                break
        
        if reference_shape is None:
            continue  # 如果没找到参考形状，跳过该键
        
        # 收集所有兼容参数
        for i, model in enumerate(models_list):
            if key in model and model[key].shape == reference_shape:
                compatible_params.append(model[key])
                compatible_weights.append(normalized_weights[i])
        
        # 如果没有兼容参数，跳过
        if not compatible_params:
            continue
            
        # 如果只有一个模型有该参数，直接使用
        if len(compatible_params) == 1:
            aggregated_model[key] = compatible_params[0].clone()
            continue
        
        # 重新归一化权重
        weight_sum = sum(compatible_weights)
        if weight_sum > 0:
            compatible_weights = [w / weight_sum for w in compatible_weights]
        else:
            compatible_weights = [1.0 / len(compatible_weights)] * len(compatible_weights)
        
        # 特殊处理BatchNorm层
        if 'running_mean' in key or 'running_var' in key:
            if 'running_mean' in key:
                # 对于均值，使用中位数或限制范围
                stacked_values = torch.stack([v.to(dtype=torch.float32, device=target_device) for v in compatible_params])
                median_value, _ = torch.median(stacked_values, dim=0)
                aggregated_model[key] = torch.clamp(median_value, min=-2.0, max=2.0)
            elif 'running_var' in key:
                # 对于方差，使用加权平均后限制范围
                weighted_avg = torch.zeros_like(compatible_params[0], dtype=torch.float32, device=target_device)
                for v, w in zip(compatible_params, compatible_weights):
                    weighted_avg += v.to(dtype=torch.float32, device=target_device) * w
                aggregated_model[key] = torch.clamp(weighted_avg, min=0.01, max=5.0)
        else:
            # 普通参数使用加权平均
            weighted_avg = torch.zeros_like(compatible_params[0], dtype=torch.float32, device=target_device)
            for v, w in zip(compatible_params, compatible_weights):
                weighted_avg += v.to(dtype=torch.float32, device=target_device) * w
            aggregated_model[key] = weighted_avg
    
    # 打印聚合统计
    print(f"全局聚合模型: 聚合了 {len(aggregated_model)}/{len(all_keys)} 个特征提取层参数")
    
    return aggregated_model

# def aggregate_models_feature_only(model_list, weights, special_bn_treatment=True):
#     """
#     仅聚合特征提取层的版本
#     """
#     if not model_list:
#         return None
    
#     # 初始化聚合模型
#     base_model = model_list[0]
#     aggregated_model = OrderedDict()
    
#     # 选择目标设备
#     target_device = torch.device('cpu')
    
#     # 遍历基础模型的所有键
#     for key in base_model.keys():
#         # 跳过分类器相关层
#         if 'classifier' in key or 'fc' in key:
#             continue
            
#         # 跳过特殊参数
#         if 'num_batches_tracked' in key:
#             aggregated_model[key] = base_model[key].clone()
#             continue
        
#         # 收集所有相同形状的参数
#         param_values = []
#         param_weights = []
        
#         for i, model in enumerate(model_list):
#             if key in model and model[key].shape == base_model[key].shape:
#                 param_values.append(model[key])
#                 param_weights.append(weights[i])
        
#         # 如果没有匹配的参数，使用基础模型的参数
#         if not param_values:
#             aggregated_model[key] = base_model[key].clone()
#             continue
        
#         # 归一化权重
#         weight_sum = sum(param_weights)
#         if weight_sum > 0:
#             param_weights = [w / weight_sum for w in param_weights]
#         else:
#             param_weights = [1.0 / len(param_weights)] * len(param_weights)
        
#         # 特殊处理BatchNorm层
#         if special_bn_treatment and ('running_mean' in key or 'running_var' in key):
#             # 对于BN层，使用中位数或限制范围
#             if 'running_mean' in key:
#                 # 确保所有值都在同一设备上进行操作
#                 stacked_values = torch.stack([v.to(dtype=torch.float32, device=target_device) for v in param_values])
#                 median_value, _ = torch.median(stacked_values, dim=0)
#                 aggregated_model[key] = torch.clamp(median_value, min=-2.0, max=2.0)
#             elif 'running_var' in key:
#                 # 对方差使用加权平均后限制范围，确保在同一设备上
#                 weighted_avg = torch.zeros_like(param_values[0], dtype=torch.float32, device=target_device)
#                 for v, w in zip(param_values, param_weights):
#                     weighted_avg += v.to(dtype=torch.float32, device=target_device) * w
#                 aggregated_model[key] = torch.clamp(weighted_avg, min=0.01, max=5.0)
#         else:
#             # 普通参数使用加权平均，确保所有值都在同一设备上
#             weighted_avg = torch.zeros_like(param_values[0], dtype=torch.float32, device=target_device)
#             for v, w in zip(param_values, param_weights):
#                 weighted_avg += v.to(dtype=torch.float32, device=target_device) * w
#             aggregated_model[key] = weighted_avg
    
#     # # 如果保留分类器，使用第一个模型的分类器参数
#     # if preserve_classifier:
#     #     for key in base_model.keys():
#     #         if 'classifier' in key or 'fc' in key:
#     #             aggregated_model[key] = base_model[key].clone()
    
#     return aggregated_model


# def aggregate_models(model_list, weights, special_bn_treatment=True, preserve_classifier=False):
#     """
#     聚合多个模型参数，可选择性地对BatchNorm层和分类器层进行特殊处理
#     """
#     if not model_list:
#         return None
    
#     # 初始化聚合模型 - 使用第一个模型作为基础
#     base_model = model_list[0]
#     aggregated_model = OrderedDict()
    
#     # 选择一个目标设备用于聚合(使用CPU以确保兼容性)
#     target_device = torch.device('cpu')
    
#     # 遍历基础模型的所有键
#     for key in base_model.keys():
#         # 跳过特殊参数（如num_batches_tracked）
#         if 'num_batches_tracked' in key:
#             aggregated_model[key] = base_model[key].clone()
#             continue
        
#         # 收集所有相同形状的参数
#         param_values = []
#         param_weights = []
        
#         for i, model in enumerate(model_list):
#             if key in model and model[key].shape == base_model[key].shape:
#                 param_values.append(model[key])
#                 param_weights.append(weights[i])
        
#         # 如果没有匹配的参数，使用基础模型的参数
#         if not param_values:
#             aggregated_model[key] = base_model[key].clone()
#             continue
        
#         # 归一化权重
#         weight_sum = sum(param_weights)
#         if weight_sum > 0:
#             param_weights = [w / weight_sum for w in param_weights]
#         else:
#             param_weights = [1.0 / len(param_weights)] * len(param_weights)
        
#         # 特殊处理BatchNorm层
#         if special_bn_treatment and ('running_mean' in key or 'running_var' in key):
#             # 对于BN层，使用中位数或限制范围
#             if 'running_mean' in key:
#                 # 确保所有值都在同一设备上进行操作
#                 stacked_values = torch.stack([v.to(dtype=torch.float32, device=target_device) for v in param_values])
#                 median_value, _ = torch.median(stacked_values, dim=0)
#                 aggregated_model[key] = torch.clamp(median_value, min=-2.0, max=2.0)
#             elif 'running_var' in key:
#                 # 对方差使用加权平均后限制范围，确保在同一设备上
#                 weighted_avg = torch.zeros_like(param_values[0], dtype=torch.float32, device=target_device)
#                 for v, w in zip(param_values, param_weights):
#                     weighted_avg += v.to(dtype=torch.float32, device=target_device) * w
#                 aggregated_model[key] = torch.clamp(weighted_avg, min=0.01, max=5.0)
#         else:
#             # 普通参数使用加权平均，确保所有值都在同一设备上
#             weighted_avg = torch.zeros_like(param_values[0], dtype=torch.float32, device=target_device)
#             for v, w in zip(param_values, param_weights):
#                 weighted_avg += v.to(dtype=torch.float32, device=target_device) * w
#             aggregated_model[key] = weighted_avg
    
#     # 如果保留分类器，使用第一个模型的分类器参数
#     if preserve_classifier:
#         for key in base_model.keys():
#             if 'classifier' in key or 'fc' in key:
#                 aggregated_model[key] = base_model[key].clone()
    
#     return aggregated_model


def hybrid_aggregation_strategy(client_models, client_weights, cluster_map, client_tiers=None):
    """
    混合聚合策略，实现双层聚合
    
    Args:
        client_models: 客户端模型参数字典
        client_weights: 客户端权重字典
        cluster_map: 聚类映射
        client_tiers: 客户端tier信息字典
        
    Returns:
        全局聚合模型参数，聚类聚合模型字典
    """
    # 1. 执行聚类内部聚合
    cluster_models = cluster_aware_aggregation(
        client_models, client_weights, cluster_map, client_tiers
    )
    
    # 检查是否有聚类模型
    if not cluster_models:
        print("警告: 聚类聚合未产生有效模型!")
        return None, {}

    # 2. 基于聚类大小，创建聚类权重
    cluster_weights = {}
    total_clients = sum(len(clients) for clients in cluster_map.values())
    
    if total_clients > 0:
        for cluster_id, clients in cluster_map.items():
            cluster_weights[cluster_id] = len(clients) / total_clients
    else:
        # 避免除零错误
        n_clusters = max(1, len(cluster_map))
        for cluster_id in cluster_map.keys():
            cluster_weights[cluster_id] = 1.0 / n_clusters
    
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


# def enhanced_hierarchical_aggregation(client_models, client_weights, cluster_map, 
#                                      global_model_template=None, num_classes=10):
#     """
#     增强的分层聚合策略，包含参数验证和分类器平衡
    
#     Args:
#         client_models: 客户端模型参数字典
#         client_weights: 客户端权重字典
#         cluster_map: 聚类映射
#         global_model_template: 全局模型模板，用于确保参数完整性
#         num_classes: 类别数量
        
#     Returns:
#         全局聚合模型，聚类模型字典，日志信息
#     """
#     log_info = {
#         'cluster_stats': {},
#         'repair_stats': {},
#         'aggregation_stats': {}
#     }
    
#     # 1. 聚类统计信息
#     for cluster_id, client_ids in cluster_map.items():
#         log_info['cluster_stats'][f'cluster_{cluster_id}'] = {
#             'size': len(client_ids),
#             'clients': client_ids
#         }
    
#     # 检查是否有客户端模型
#     if not client_models:
#         print("警告: 没有客户端模型可聚合!")
#         return global_model_template or {}, {}, log_info

#     # 2. 执行双层聚合
#     global_model, cluster_models = hybrid_aggregation_strategy(
#         client_models, client_weights, cluster_map
#     )
#     # 如果聚合失败，使用模板或返回空字典
#     if global_model is None:
#         print("警告: 聚合失败，使用全局模型模板或返回空字典")
#         global_model = global_model_template.copy() if global_model_template else {}
    
#     # 3. 验证和修复全局模型参数
#     global_model, fixed_count = validate_model_parameters(global_model)
#     log_info['repair_stats']['global_fixed_count'] = fixed_count
    
#     # 4. 平衡分类器权重
#     global_model = balance_classifier_weights(global_model, num_classes)
    
#     # 5. 如果提供了全局模型模板，确保参数完整性
#     if global_model_template is not None:
#         # 复制模板中存在但聚合结果中不存在的参数
#         for key, param in global_model_template.items():
#             if key not in global_model:
#                 global_model[key] = param.clone()
        
#         log_info['aggregation_stats']['template_keys_used'] = len(global_model_template.keys()) - len(global_model.keys())
    
#     # 6. 验证和修复聚类模型参数
#     for cluster_id, model in cluster_models.items():
#         cluster_models[cluster_id], fixed_count = validate_model_parameters(model)
#         log_info['repair_stats'][f'cluster_{cluster_id}_fixed_count'] = fixed_count
    
#     return global_model, cluster_models, log_info

def enhanced_hierarchical_aggregation_no_projection(client_models, client_weights, cluster_map, client_tiers=None,
                                     global_model_template=None, num_classes=10):
    """修改后的层次聚合函数，忽略projection层参数"""
    
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

    # 2. 执行双层聚合
    global_model, cluster_models = hybrid_aggregation_strategy(
        filtered_client_models, client_weights, cluster_map, client_tiers
    )
    
    # 如果聚合失败，使用模板或返回空字典
    if global_model is None:
        print("警告: 聚合失败，使用全局模型模板或返回空字典")
        global_model = copy.deepcopy(global_model_template) if global_model_template else {}
    
    # 3. 验证和修复全局模型参数
    global_model, fixed_count = validate_model_parameters(global_model)
    log_info['repair_stats']['global_fixed_count'] = fixed_count
    
    # 4. 平衡分类器权重
    global_model = balance_classifier_weights(global_model, num_classes)
    
    # 5. 如果提供了全局模型模板，确保参数完整性
    if global_model_template is not None:
        # 复制模板中存在但聚合结果中不存在的参数
        keys_from_template = 0
        for key, param in global_model_template.items():
            if key not in global_model:
                global_model[key] = param.clone()
                keys_from_template += 1
        
        log_info['aggregation_stats']['template_keys_used'] = keys_from_template
    
    # 6. 验证和修复聚类模型参数
    for cluster_id, model in cluster_models.items():
        cluster_models[cluster_id], fixed_count = validate_model_parameters(model)
        log_info['repair_stats'][f'cluster_{cluster_id}_fixed_count'] = fixed_count
    
    return global_model, cluster_models, log_info
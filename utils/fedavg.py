import torch
import copy
from collections import OrderedDict
import numpy as np

# def aggregate_clients_models(cluster_clients, w_locals_client_idx, num_clusters, whether_local_loss, client_sample, idxs_users, **kwargs):
#     local_v2 = kwargs.get('local_v2', False)
    
#     # 初始化存储结构
#     largest_client = 0
#     id_largest_client = {}
#     client_model = {}
    
#     # # 要忽略的层
#     # keys_to_remove = {
#     #     'classifier.fc1.weight', 'classifier.fc1.bias', 
#     #     'classifier.fc2.weight', 'classifier.fc2.bias', 
#     #     'classifier.fc3.weight', 'classifier.fc3.bias',
#     #     'module.classifier.fc1.weight', 'module.classifier.fc1.bias',
#     #     'module.classifier.fc2.weight', 'module.classifier.fc2.bias',
#     #     'module.classifier.fc3.weight', 'module.classifier.fc3.bias'
#     # }
    
#     # 第一步：找出每个cluster中最大的客户端模型
#     for cluster_id in range(num_clusters):
#         print(f"Processing cluster {cluster_id} with clients: {cluster_clients[cluster_id]}")
        
#         if len(cluster_clients[cluster_id]) > 0:
#             # 初始化该cluster的最大客户端
#             max_size = -1
#             max_client = None
            
#             for client_idx in cluster_clients[cluster_id]:
#                 if client_idx not in w_locals_client_idx:
#                     print(f"Warning: Client {client_idx} not found in w_locals_client_idx")
#                     continue
#                 # 客户端模型去除分类器部分，其余层的权重信息保留在client_model[client_idx]中    
#                 if whether_local_loss and not local_v2:
#                     client_model[client_idx] = {
#                         k: v for k, v in w_locals_client_idx[client_idx].items()
#                         if 'projection' not in k
#                     }
#                 else:
#                     client_model[client_idx] = w_locals_client_idx[client_idx].copy()
                
#                 current_size = len(client_model[client_idx])
#                 if current_size > max_size:
#                     max_size = current_size
#                     max_client = client_idx
            
#             if max_client is not None:
#                 id_largest_client[cluster_id] = max_client
#                 # largest_client = max(largest_client, max_size)
    
#     aggregated_client_state_dict = {}
    
#     # 对每个cluster进行处理
#     for cluster_id in range(num_clusters):
#         if cluster_id not in id_largest_client:
#             print(f"Warning: No valid clients found in cluster {cluster_id}")
#             continue
            
#         largest_client_idx = id_largest_client[cluster_id]
#         aggregated_client_state_dict[cluster_id] = {}
        
#         # 使用字典记录每个参数的累积和和权重和
#         param_sums = {}
#         client_weights_sum = {}
        
#         valid_clients = 0  # 记录有效客户端数量
        
#         # 对cluster中的每个客户端进行处理
#         for client_idx in cluster_clients[cluster_id]:
#             if client_idx not in client_model:
#                 print(f"Skipping client {client_idx} as it has no model")
#                 continue
                
#             valid_clients += 1
#             client_weight = client_sample[client_idx]
            
#             for k, v in client_model[client_idx].items():
#                 if any(classifier_key in k for classifier_key in ['projection']):
#                     continue

#                 if k not in param_sums:
#                     param_sums[k] = torch.zeros_like(v)
#                     client_weights_sum[k] = 0.0
                
#                 if param_sums[k].shape != v.shape:
#                     print(f"Warning: Parameter '{k}' shape mismatch. Expected {param_sums[k].shape}, got {v.shape}")
#                     continue
                
#                 if v.dtype == torch.int64:
#                     param_sums[k] = torch.maximum(param_sums[k], v)
#                 else:
#                     param_sums[k] += v * client_weight
#                     client_weights_sum[k] += client_weight
        
#         print(f"Cluster {cluster_id}: Processed {valid_clients} valid clients")
        
#         # 只有当有有效客户端时才进行聚合
#         if valid_clients > 0:
#             # 计算最终的聚合结果
#             for k in param_sums:
#                 if client_model[largest_client_idx][k].dtype == torch.int64:
#                     aggregated_client_state_dict[cluster_id][k] = param_sums[k]
#                 else:
#                     if client_weights_sum[k] > 0:
#                         aggregated_client_state_dict[cluster_id][k] = param_sums[k] / client_weights_sum[k]
#                     else:
#                         aggregated_client_state_dict[cluster_id][k] = torch.zeros_like(param_sums[k])
    
#     return aggregated_client_state_dict
 



# def aggregated_fedavg(w_locals_server_tier, w_locals_client_tier, num_tiers, num_users, whether_local_loss, client_sample, idxs_users, **kwargs):  
#     local_v2 = False

#     # 检查 w_locals_server_tier 是否为空
#     if len(w_locals_server_tier) == 0:
#         print("Error: w_locals_server_tier is empty. Cannot perform aggregation.")
#         return None

#     # 检查 w_locals_client_tier 是否为空
#     if len(w_locals_client_tier) == 0:
#         print("Error: w_locals_client_tier is empty. Cannot perform aggregation.")
#         return None
#     # 定义需要忽略的分类器相关层
#     classifier_keys = {
#         'classifier.fc1.weight', 'classifier.fc1.bias',
#         'classifier.fc2.weight', 'classifier.fc2.bias',
#         'classifier.fc3.weight', 'classifier.fc3.bias',
#         'module.classifier.fc1.weight', 'module.classifier.fc1.bias',
#         'module.classifier.fc2.weight', 'module.classifier.fc2.bias',
#         'module.classifier.fc3.weight', 'module.classifier.fc3.bias'
#     }
#     if kwargs:
#         local_v2 = kwargs['local_v2']
#     if local_v2:
#         for t in range(1, num_tiers + 1):
#             if t in w_locals_client_tier:
#                 for k in w_locals_server_tier[t].keys():
#                     if k in w_locals_client_tier[t].keys():
#                         del w_locals_client_tier[t][k]
    
    
#     largest_client, largest_server = 0, 0
#     largest_client_tier = 0
#     largest_server_tier = 0

#     # 检查客户端模型大小
#     for i in range(len(w_locals_client_tier)):
#         if len(w_locals_client_tier[i]) > largest_client:
#             largest_client = len(w_locals_client_tier[i])
#             largest_client_tier = i

#     # 检查服务器模型大小，使用 w_locals_server_tier 的长度
#     for i in range(len(w_locals_server_tier)):
#         if len(w_locals_server_tier[i]) > largest_server:
#             largest_server = len(w_locals_server_tier[i])
#             largest_server_tier = i        

#     # 确保索引在有效范围内
#     if largest_server_tier >= len(w_locals_server_tier):
#         print(f"Error: largest_server_tier ({largest_server_tier}) is out of range for w_locals_server_tier (length: {len(w_locals_server_tier)})")
#         return None

#     w_avg = copy.deepcopy(w_locals_server_tier[largest_server_tier]) # largest model weight (suppose last tier in server is the biggest)
                    

    
#     # for k in w_locals_client_tier[num_tiers]:
#     for k in w_locals_client_tier[largest_client_tier]:
#         if k not in w_avg.keys():
#             w_avg[k] = 0
#     for k in w_avg.keys():
#         w_avg[k] = 0
#         c = 0
#         for i in range(0, len(w_locals_client_tier)):
#             if k in w_locals_client_tier[i]:
#                 # if k == 'fc.bias':
#                 #     print(k)
#                 w_avg[k] += w_locals_client_tier[i][k] * client_sample[i]
#                 c += 1
#         for i in range(0, len(w_locals_server_tier)):
#             if k in w_locals_server_tier[i]:
#                 # if k == 'fc.bias':
#                 #     print(k)
#                 w_avg[k] += w_locals_server_tier[i][k] * client_sample[i]
#                 # print(k)
#                 c += 1
#         # if c != num_users:# and False:
#         #     print(k, c)            
#         # w_avg[k] = torch.div(w_avg[k], num_users)
#         #w_avg[k] = torch.div(w_avg[k], len(w_locals_server_tier))  # devide by number of involved clients
#         w_avg[k] = torch.div(w_avg[k], sum(client_sample))  # devide by number of involved clients
        
                
        
    
#     # for i in range(1,num_tiers+1):
        
        
    
    
#     return w_avg

# # multi_fedavg(w, num_tiers)


# # 2. 修改后的聚合函数
# def aggregated_fedavg(w_locals_server_tier, w_locals_client_tier, num_tiers, num_users, 
#                      whether_local_loss, client_sample, idxs_users, **kwargs):
#     """
#     修改后的聚合函数，处理列表类型的服务器端模型
#     """
#     print("\n=== 开始聚合过程 ===")
    
#     # 找到最大的模型作为基准
#     largest_model = None
#     largest_size = 0
    
#     # 检查服务器端模型
#     for i, model in enumerate(w_locals_server_tier):
#         if isinstance(model, dict):
#             current_size = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
#             if current_size > largest_size:
#                 largest_size = current_size
#                 largest_model = model
#                 print(f"Found larger model at index {i} with size {current_size}")
    
#     # 如果服务器端没有找到，检查客户端模型
#     if largest_model is None and isinstance(w_locals_client_tier, list):
#         for i, model in enumerate(w_locals_client_tier):
#             if isinstance(model, dict):
#                 current_size = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
#                 if current_size > largest_size:
#                     largest_size = current_size
#                     largest_model = model
#                     print(f"Found larger model in client models at index {i} with size {current_size}")
    
#     if largest_model is None:
#         raise ValueError("No valid model found in either server or client models")
    
#     # 初始化平均权重字典
#     w_avg = copy.deepcopy(largest_model)
    
#     # 初始化权重
#     for k in w_avg.keys():
#         w_avg[k] = torch.zeros_like(w_avg[k], dtype=torch.float32)
    
#     weight_count = {k: 0.0 for k in w_avg.keys()}
    
#     # 聚合服务器端模型
#     print("\nProcessing server models...")
#     for i, model in enumerate(w_locals_server_tier):
#         if not isinstance(model, dict):
#             continue
            
#         sample_weight = client_sample[i] if i < len(client_sample) else 1.0
        
#         for k in w_avg.keys():
#             if k in model:
#                 try:
#                     weight = model[k]
#                     if weight.dtype != torch.float32:
#                         weight = weight.to(torch.float32)
                    
#                     if weight.shape == w_avg[k].shape:
#                         w_avg[k] += weight * sample_weight
#                         weight_count[k] += sample_weight
#                     else:
#                         print(f"Shape mismatch in server model {i} for {k}: "
#                               f"expected {w_avg[k].shape}, got {weight.shape}")
#                 except Exception as e:
#                     print(f"Error processing server model {i} parameter {k}: {str(e)}")
    
#     # 聚合客户端模型（如果是列表类型）
#     if isinstance(w_locals_client_tier, list):
#         print("\nProcessing client models...")
#         for i, model in enumerate(w_locals_client_tier):
#             if not isinstance(model, dict):
#                 continue
                
#             sample_weight = client_sample[i] if i < len(client_sample) else 1.0
            
#             for k in w_avg.keys():
#                 if k in model:
#                     try:
#                         weight = model[k]
#                         if weight.dtype != torch.float32:
#                             weight = weight.to(torch.float32)
                        
#                         if weight.shape == w_avg[k].shape:
#                             w_avg[k] += weight * sample_weight
#                             weight_count[k] += sample_weight
#                         else:
#                             print(f"Shape mismatch in client model {i} for {k}: "
#                                   f"expected {w_avg[k].shape}, got {weight.shape}")
#                     except Exception as e:
#                         print(f"Error processing client model {i} parameter {k}: {str(e)}")
    
#     # 计算最终平均值
#     print("\nComputing final averages...")
#     for k in w_avg.keys():
#         if weight_count[k] > 0:
#             w_avg[k] = w_avg[k] / weight_count[k]
#         else:
#             print(f"Warning: No weights accumulated for parameter {k}")
    
#     print("\n=== 聚合完成 ===")
#     return w_avg

def aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, 
                       whether_local_loss, client_sample, idxs_users, target_device='cpu'):
    """
    增强版聚合函数，基于层数最多的服务器端和客户端模型，确保包含所有可能的层
    
    Args:
        w_locals_tier: 服务器端模型状态字典列表
        w_locals_client: 客户端模型状态字典列表
        num_tiers: tier数量
        num_users: 用户数量
        whether_local_loss: 是否使用本地损失
        client_sample: 客户端采样权重
        idxs_users: 用户索引
        target_device: 目标设备，默认为'cpu'
        
    Returns:
        聚合后的模型状态字典
    """
    
    print("\n=== 开始聚合过程 ===")
    
    # 1. 找到层数最多的服务器端模型
    largest_server_model = None
    largest_server_layers = 0
    largest_server_idx = -1
    
    for i, model in enumerate(w_locals_tier):
        if not isinstance(model, dict):
            continue
        
        # 计算层数（不重复计算）
        unique_layer_prefixes = set()
        for k in model.keys():
            # 提取层前缀 (例如 module.conv1, module.layer1.0, 等)
            parts = k.split('.')
            if len(parts) >= 2:
                if parts[0] == 'module':
                    prefix = f"{parts[0]}.{parts[1]}"
                else:
                    prefix = parts[0]
                unique_layer_prefixes.add(prefix)
        
        layer_count = len(unique_layer_prefixes)
        if layer_count > largest_server_layers:
            largest_server_layers = layer_count
            largest_server_model = model
            largest_server_idx = i
    
    if largest_server_model is None:
        print("错误: 未找到有效的服务器端模型!")
        return None
    
    print(f"找到层数最多的服务器端模型 (索引 {largest_server_idx}) 包含 {largest_server_layers} 个层前缀")
    
    # 2. 找到层数最多的客户端模型
    largest_client_model = None
    largest_client_layers = 0
    largest_client_idx = -1
    
    for i, model in enumerate(w_locals_client):
        if not isinstance(model, dict):
            continue
        
        # 计算层数（不重复计算）
        unique_layer_prefixes = set()
        for k in model.keys():
            # 提取层前缀
            parts = k.split('.')
            if len(parts) >= 2:
                if parts[0] == 'module':
                    prefix = f"{parts[0]}.{parts[1]}"
                else:
                    prefix = parts[0]
                unique_layer_prefixes.add(prefix)
        
        layer_count = len(unique_layer_prefixes)
        if layer_count > largest_client_layers:
            largest_client_layers = layer_count
            largest_client_model = model
            largest_client_idx = i
    
    if largest_client_model is None:
        print("警告: 未找到有效的客户端模型，将仅使用服务器端模型")
    else:
        print(f"找到层数最多的客户端模型 (索引 {largest_client_idx}) 包含 {largest_client_layers} 个层前缀")
    
    # 3. 初始化全局模型，使用服务器端模型作为基础
    w_glob = OrderedDict()
    
    # 首先从服务器端模型复制所有参数
    for k, v in largest_server_model.items():
        w_glob[k] = v.clone().to(dtype=torch.float32, device=target_device)
    
    # 如果找到客户端模型，检查是否有服务器端模型中不存在的层
    if largest_client_model is not None:
        for k, v in largest_client_model.items():
            if k not in w_glob:
                print(f"从客户端模型添加缺失的层: {k}")
                w_glob[k] = v.clone().to(dtype=torch.float32, device=target_device)
    
    # 检查初始化的全局模型的层数
    print(f"初始化的全局模型包含 {len(w_glob)} 个参数")
    
    # 4. 重置参数为零（为聚合做准备）
    weight_count = {k: 0.0 for k in w_glob.keys()}
    
    for k in w_glob.keys():
        w_glob[k] = torch.zeros_like(w_glob[k], dtype=torch.float32)
    
    # 5. 聚合服务器端模型
    print("\n聚合服务器端模型...")
    for i, model in enumerate(w_locals_tier):
        if not isinstance(model, dict):
            continue
            
        sample_weight = client_sample[i] if i < len(client_sample) else 1.0
        
        for k, v in model.items():
            if k in w_glob:
                try:
                    weight = v.to(dtype=torch.float32, device=target_device)
                    
                    if weight.shape == w_glob[k].shape:
                        w_glob[k] += weight * sample_weight
                        weight_count[k] += sample_weight
                    else:
                        print(f"形状不匹配 (服务器模型 {i}): {k}, "
                              f"预期 {w_glob[k].shape}, 实际 {weight.shape}")
                except Exception as e:
                    print(f"处理服务器模型 {i} 参数 {k} 时出错: {str(e)}")
    
    # 6. 聚合客户端模型
    print("\n聚合客户端模型...")
    for i, model in enumerate(w_locals_client):
        if not isinstance(model, dict):
            continue
            
        sample_weight = client_sample[i] if i < len(client_sample) else 1.0
        
        for k, v in model.items():
            if k in w_glob:
                try:
                    weight = v.to(dtype=torch.float32, device=target_device)
                    
                    if weight.shape == w_glob[k].shape:
                        w_glob[k] += weight * sample_weight
                        weight_count[k] += sample_weight
                    else:
                        print(f"形状不匹配 (客户端模型 {i}): {k}, "
                              f"预期 {w_glob[k].shape}, 实际 {weight.shape}")
                except Exception as e:
                    print(f"处理客户端模型 {i} 参数 {k} 时出错: {str(e)}")
    
    # 7. 计算平均值 
    print("\n计算最终的全局模型参数...")
    no_weight_params = []
    
    for k in w_glob.keys():
        if weight_count[k] > 0:
            w_glob[k] = w_glob[k] / weight_count[k]
        else:
            no_weight_params.append(k)
    
    # 8. 处理没有权重的参数
    if no_weight_params:
        print(f"\n警告: {len(no_weight_params)} 个参数没有累计权重")
        print("从最大服务器模型或客户端模型复制这些参数")
        
        for k in no_weight_params:
            # 首先尝试从服务器端模型获取
            if k in largest_server_model:
                w_glob[k] = largest_server_model[k].to(dtype=torch.float32, device=target_device)
            # 其次尝试从客户端模型获取
            elif largest_client_model is not None and k in largest_client_model:
                w_glob[k] = largest_client_model[k].to(dtype=torch.float32, device=target_device)
            # 如果仍然找不到，搜索所有模型
            else:
                found = False
                for model in w_locals_tier + w_locals_client:
                    if isinstance(model, dict) and k in model:
                        w_glob[k] = model[k].to(dtype=torch.float32, device=target_device)
                        found = True
                        break
                if not found:
                    print(f"警告: 参数 {k} 在所有模型中都找不到有效值")
    
    # 9. 验证关键层是否存在
    base_layers = ['module.conv1.weight', 'module.bn1.weight']
    classifier_layers = ['module.classifier.fc1.weight', 'module.classifier.fc3.bias']
    
    print("\n验证全局模型中的关键层:")
    
    print("基础层:")
    for layer in base_layers:
        if layer in w_glob:
            print(f"  ✓ {layer} - 形状: {w_glob[layer].shape}")
        else:
            print(f"  ✗ {layer} - 缺失!")
    
    print("\n分类器层:")
    for layer in classifier_layers:
        if layer in w_glob:
            print(f"  ✓ {layer} - 形状: {w_glob[layer].shape}")
        else:
            print(f"  ✗ {layer} - 缺失!")
    
    print(f"\n全局模型最终包含 {len(w_glob)} 个参数")
    print("=== 聚合完成 ===")
    
    return w_glob

def weighted_average_params(param_list):
    """计算参数的加权平均"""
    if not param_list:
        return None
    
    # 初始化为零张量
    result = torch.zeros_like(param_list[0][0])
    
    # 加权求和
    for param, weight in param_list:
        result.add_(param * weight)
    
    return result


def median_bn_params(param_list):
    """使用中位数计算BN均值参数"""
    if not param_list:
        return None
    
    # 收集所有参数
    params = [p[0].cpu().numpy() for p in param_list]
    
    # 计算中位数
    median_param = np.median(params, axis=0)
    
    # 转回张量
    result = torch.tensor(median_param, dtype=param_list[0][0].dtype, device=param_list[0][0].device)
    
    return result


def robust_variance_aggregation(param_list):
    """使用稳健方法聚合BN层方差参数"""
    if not param_list:
        return None
    
    # 收集所有参数
    params = [p[0].cpu().numpy() for p in param_list]
    weights = [p[1] for p in param_list]
    
    # 计算加权平均
    weighted_avg = np.zeros_like(params[0])
    for param, weight in zip(params, weights):
        weighted_avg += param * weight
    
    # 限制方差参数范围
    weighted_avg = np.clip(weighted_avg, 0.01, 5.0)
    
    # 转回张量
    result = torch.tensor(weighted_avg, dtype=param_list[0][0].dtype, device=param_list[0][0].device)
    
    return result


def max_bn_tracked(param_list):
    """使用最大值聚合BN层追踪批次数参数"""
    if not param_list:
        return None
    
    # 找出最大值
    max_val = max([p[0].item() for p in param_list])
    
    # 转为张量
    result = torch.tensor(max_val, dtype=param_list[0][0].dtype, device=param_list[0][0].device)
    
    return result


def intra_cluster_aggregation(client_models, client_weights, bn_fix=True):
    """
    聚类内部聚合方法，处理单个聚类内的客户端模型
    
    Args:
        client_models: 聚类内的客户端模型参数列表
        client_weights: 对应的客户端权重
        bn_fix: 是否特殊处理BN层
        
    Returns:
        聚合后的模型参数
    """
    if not client_models:
        return None
    
    # 归一化权重
    total_weight = sum(client_weights)
    if total_weight == 0:
        normalized_weights = [1.0/len(client_weights)] * len(client_weights)
    else:
        normalized_weights = [w/total_weight for w in client_weights]
    
    # 初始化聚合模型参数
    aggregated_model = {}
    
    # 根据参数类型分组
    bn_params = {}  # BatchNorm层参数
    other_params = {}  # 其他参数
    
    # 第一步：分离BatchNorm层和其他层
    for client_idx, model in enumerate(client_models):
        for key, param in model.items():
            is_bn_param = any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked'])
            
            if is_bn_param and bn_fix:
                if key not in bn_params:
                    bn_params[key] = []
                bn_params[key].append((param, normalized_weights[client_idx]))
            else:
                if key not in other_params:
                    other_params[key] = []
                other_params[key].append((param, normalized_weights[client_idx]))
    
    # 第二步：聚合非BN参数 (加权平均)
    for key, param_list in other_params.items():
        aggregated_model[key] = weighted_average_params(param_list)
    
    # 第三步：特殊处理BN参数
    if bn_fix:
        for key, param_list in bn_params.items():
            if 'running_mean' in key:
                # 使用中位数而非平均值
                aggregated_model[key] = median_bn_params(param_list)
            elif 'running_var' in key:
                # 使用稳健的方差聚合
                aggregated_model[key] = robust_variance_aggregation(param_list)
            elif 'num_batches_tracked' in key:
                # 使用最大值
                aggregated_model[key] = max_bn_tracked(param_list)
    
    return aggregated_model

def aggregate_clients_models(client_clusters, client_weights, num_clusters, whether_local_loss, client_sample, idxs_users):
    """
    基于聚类的客户端模型聚合方法
    
    Args:
        client_clusters: 客户端聚类结果
        client_weights: 客户端模型权重字典
        num_clusters: 聚类数量
        whether_local_loss: 是否使用本地损失
        client_sample: 客户端数据样本量
        idxs_users: 参与用户索引列表
        
    Returns:
        聚类聚合结果字典
    """
    print("\n使用改进的客户端模型聚合方法...")
    
    aggregated_models = {}
    
    # 遍历每个聚类
    for cluster_id in client_clusters:
        print(f"\n处理聚类 {cluster_id} 的客户端")
        
        # 获取当前聚类的客户端索引
        client_indices = client_clusters[cluster_id]
        
        # 收集该聚类的客户端模型和权重
        cluster_models = []
        cluster_weights = []
        
        for idx in client_indices:
            if idx in client_weights:  # 确保客户端模型存在
                cluster_models.append(client_weights[idx])
                
                # 获取客户端权重 (基于样本量)
                if idx in idxs_users.tolist():
                    client_idx = idxs_users.tolist().index(idx)
                    if client_idx < len(client_sample):
                        cluster_weights.append(client_sample[client_idx])
                    else:
                        # 默认权重
                        cluster_weights.append(1.0)
                else:
                    # 默认权重
                    cluster_weights.append(1.0)
        
        # 对该聚类执行内部聚合
        if cluster_models:
            print(f"聚类 {cluster_id}: 聚合 {len(cluster_models)} 个客户端模型")
            
            # 聚合该聚类的模型
            aggregated_models[cluster_id] = intra_cluster_aggregation(
                client_models=cluster_models,
                client_weights=cluster_weights,
                bn_fix=True  # 启用BN层特殊处理
            )
    
    return aggregated_models

def enhanced_aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample, idxs_users, target_device='cpu'):
    """
    改进的联邦平均聚合函数，包含BN层特殊处理
    
    Args:
        w_locals_tier: 按tier分组的本地模型权重
        w_locals_client: 客户端模型权重列表
        num_tiers: tier数量
        num_users: 用户数量
        whether_local_loss: 是否使用本地损失
        client_sample: 客户端样本量
        idxs_users: 参与用户索引
        target_device: 目标设备
        
    Returns:
        聚合后的全局模型权重
    """
    print("\n执行增强版联邦平均聚合...")
    
    # 初始化聚合权重
    w_glob = {}
    
    # 为不同tier的聚合添加调试信息
    for tier in range(1, num_tiers+1):
        if not w_locals_client:
            print(f"警告: Tier {tier} 没有客户端权重")
            continue
        
        print(f"\n处理 Tier {tier} 的客户端聚合...")
        
        # 获取当前tier的客户端索引和权重
        current_tier_indices = []
        current_tier_weights = []
        
        for i, idx in enumerate(idxs_users):
            if i < len(idxs_users) and client_tier[idx] == tier:
                if i < len(client_sample):  # 确保索引有效
                    current_tier_indices.append(i)
                    current_tier_weights.append(client_sample[i])
        
        if not current_tier_indices:
            print(f"Tier {tier} 没有活跃客户端，跳过")
            continue
        
        print(f"Tier {tier} 有 {len(current_tier_indices)} 个活跃客户端")
        
        # 收集当前tier的客户端权重
        w_current_tier = [w_locals_client[i] for i in current_tier_indices]
        
        # 归一化权重
        total_weight = sum(current_tier_weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in current_tier_weights]
        else:
            normalized_weights = [1.0/len(current_tier_weights)] * len(current_tier_weights)
        
        # 执行特殊的BN层聚合
        if w_current_tier:
            w_tier = {}
            
            # 获取第一个模型的所有键作为基础
            keys = w_current_tier[0].keys()
            
            for k in keys:
                # 检查是否是BN层参数
                is_bn = 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k
                
                if is_bn:
                    # 收集该参数的所有值
                    values = [model[k] for model in w_current_tier if k in model]
                    weights = normalized_weights[:len(values)]
                    
                    if 'running_mean' in k:
                        # 打印方差范围
                        min_val = min(tensor.min().item() for tensor in values)
                        max_val = max(tensor.max().item() for tensor in values)
                        print(f"  {k}: mean范围 [{min_val:.4f}, {max_val:.4f}]")
                        
                        if max_val > 2.0 or min_val < -2.0:
                            # 使用中位数聚合
                            stacked = torch.stack(values)
                            median_val, _ = torch.median(stacked, dim=0)
                            w_tier[k] = torch.clamp(median_val, min=-2.0, max=2.0)
                        else:
                            # 常规加权平均
                            w_tier[k] = sum(w * v for w, v in zip(weights, values)) / sum(weights)
                            
                    elif 'running_var' in k:
                        # 打印方差范围
                        min_val = min(tensor.min().item() for tensor in values)
                        max_val = max(tensor.max().item() for tensor in values)
                        print(f"  {k}: var范围 [{min_val:.4f}, {max_val:.4f}]")
                        
                        if max_val > 5.0 or min_val < 0.01:
                            print(f"  修复 {k} 的异常方差值")
                            # 特殊处理方差
                            w_tier[k] = sum(w * torch.clamp(v, min=0.01, max=5.0) for w, v in zip(weights, values)) / sum(weights)
                        else:
                            # 常规加权平均
                            w_tier[k] = sum(w * v for w, v in zip(weights, values)) / sum(weights)
                            
                    elif 'num_batches_tracked' in k:
                        # 使用最大值
                        w_tier[k] = max(values)
                else:
                    # 非BN层参数，常规加权平均
                    w_tier[k] = sum(w * model[k] for w, model in zip(normalized_weights, w_current_tier) if k in model) / sum(normalized_weights)
            
            # 更新全局模型
            for k in w_tier.keys():
                if k not in w_glob:
                    w_glob[k] = w_tier[k]
    
    print("\n联邦聚合完成")
    return w_glob


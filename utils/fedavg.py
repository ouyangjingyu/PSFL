import torch
import copy
from collections import OrderedDict

def aggregate_clients_models(cluster_clients, w_locals_client_idx, num_clusters, whether_local_loss, client_sample, idxs_users, **kwargs):
    local_v2 = kwargs.get('local_v2', False)
    
    # 初始化存储结构
    largest_client = 0
    id_largest_client = {}
    client_model = {}
    
    # # 要忽略的层
    # keys_to_remove = {
    #     'classifier.fc1.weight', 'classifier.fc1.bias', 
    #     'classifier.fc2.weight', 'classifier.fc2.bias', 
    #     'classifier.fc3.weight', 'classifier.fc3.bias',
    #     'module.classifier.fc1.weight', 'module.classifier.fc1.bias',
    #     'module.classifier.fc2.weight', 'module.classifier.fc2.bias',
    #     'module.classifier.fc3.weight', 'module.classifier.fc3.bias'
    # }
    
    # 第一步：找出每个cluster中最大的客户端模型
    for cluster_id in range(num_clusters):
        print(f"Processing cluster {cluster_id} with clients: {cluster_clients[cluster_id]}")
        
        if len(cluster_clients[cluster_id]) > 0:
            # 初始化该cluster的最大客户端
            max_size = -1
            max_client = None
            
            for client_idx in cluster_clients[cluster_id]:
                if client_idx not in w_locals_client_idx:
                    print(f"Warning: Client {client_idx} not found in w_locals_client_idx")
                    continue
                # 客户端模型去除分类器部分，其余层的权重信息保留在client_model[client_idx]中    
                if whether_local_loss and not local_v2:
                    client_model[client_idx] = {
                        k: v for k, v in w_locals_client_idx[client_idx].items()
                        if 'projection' not in k
                    }
                else:
                    client_model[client_idx] = w_locals_client_idx[client_idx].copy()
                
                current_size = len(client_model[client_idx])
                if current_size > max_size:
                    max_size = current_size
                    max_client = client_idx
            
            if max_client is not None:
                id_largest_client[cluster_id] = max_client
                # largest_client = max(largest_client, max_size)
    
    aggregated_client_state_dict = {}
    
    # 对每个cluster进行处理
    for cluster_id in range(num_clusters):
        if cluster_id not in id_largest_client:
            print(f"Warning: No valid clients found in cluster {cluster_id}")
            continue
            
        largest_client_idx = id_largest_client[cluster_id]
        aggregated_client_state_dict[cluster_id] = {}
        
        # 使用字典记录每个参数的累积和和权重和
        param_sums = {}
        client_weights_sum = {}
        
        valid_clients = 0  # 记录有效客户端数量
        
        # 对cluster中的每个客户端进行处理
        for client_idx in cluster_clients[cluster_id]:
            if client_idx not in client_model:
                print(f"Skipping client {client_idx} as it has no model")
                continue
                
            valid_clients += 1
            client_weight = client_sample[client_idx]
            
            for k, v in client_model[client_idx].items():
                if any(classifier_key in k for classifier_key in ['projection']):
                    continue

                if k not in param_sums:
                    param_sums[k] = torch.zeros_like(v)
                    client_weights_sum[k] = 0.0
                
                if param_sums[k].shape != v.shape:
                    print(f"Warning: Parameter '{k}' shape mismatch. Expected {param_sums[k].shape}, got {v.shape}")
                    continue
                
                if v.dtype == torch.int64:
                    param_sums[k] = torch.maximum(param_sums[k], v)
                else:
                    param_sums[k] += v * client_weight
                    client_weights_sum[k] += client_weight
        
        print(f"Cluster {cluster_id}: Processed {valid_clients} valid clients")
        
        # 只有当有有效客户端时才进行聚合
        if valid_clients > 0:
            # 计算最终的聚合结果
            for k in param_sums:
                if client_model[largest_client_idx][k].dtype == torch.int64:
                    aggregated_client_state_dict[cluster_id][k] = param_sums[k]
                else:
                    if client_weights_sum[k] > 0:
                        aggregated_client_state_dict[cluster_id][k] = param_sums[k] / client_weights_sum[k]
                    else:
                        aggregated_client_state_dict[cluster_id][k] = torch.zeros_like(param_sums[k])
    
    return aggregated_client_state_dict
 



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
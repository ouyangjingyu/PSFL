import torch
import copy
import numpy as np
import torch.nn as nn

def improved_model_aggregation(w_locals, client_weights):
    """
    改进的模型参数聚合算法，保持批标准化层的稳定性
    Args:
        w_locals: 本地模型参数列表
        client_weights: 客户端权重列表
    """
    if len(w_locals) == 0:
        return None
    
    # 归一化客户端权重
    total_weight = sum(client_weights)
    normalized_weights = [w/total_weight for w in client_weights]
    
    # 初始化聚合模型参数
    w_avg = copy.deepcopy(w_locals[0])
    
    # 统计参数类型信息
    is_bn_layer = {}
    for k in w_avg.keys():
        # 检测是否是批标准化层的运行统计量
        is_bn = any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked'])
        is_bn_layer[k] = is_bn
    
    # 清空初始权重值
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])
    
    # 执行加权平均
    for k in w_avg.keys():
        if not is_bn_layer[k]:  # 对于非BN层，正常进行加权平均
            for i in range(len(w_locals)):
                if k in w_locals[i]:
                    # 确保类型匹配，避免精度问题
                    w_avg[k] += w_locals[i][k].to(w_avg[k].dtype) * normalized_weights[i]
        else:  # 对于BN层，使用更稳健的聚合方法
            if 'running_mean' in k or 'running_var' in k:
                # 计算中位数而不是平均值
                values = []
                for i in range(len(w_locals)):
                    if k in w_locals[i]:
                        values.append(w_locals[i][k].cpu().numpy())
                
                if values:
                    median_value = np.median(values, axis=0)
                    w_avg[k] = torch.tensor(median_value).to(w_avg[k].device).to(w_avg[k].dtype)
            elif 'num_batches_tracked' in k:
                # 对于batch跟踪计数器，取最大值
                max_value = 0
                for i in range(len(w_locals)):
                    if k in w_locals[i]:
                        max_value = max(max_value, w_locals[i][k].item())
                w_avg[k] = torch.tensor(max_value).to(w_avg[k].device).to(w_avg[k].dtype)
    
    return w_avg

def fix_batchnorm_before_aggregation(model_weights_list):
    """
    在聚合前修复批标准化层的异常值
    """
    fixed_weights_list = []
    
    for weights in model_weights_list:
        fixed_weights = copy.deepcopy(weights)
        
        for k in fixed_weights.keys():
            # 处理批标准化层的running_mean和running_var
            if 'running_mean' in k:
                # 限制running_mean的范围
                fixed_weights[k] = torch.clamp(fixed_weights[k], min=-2.0, max=2.0)
            elif 'running_var' in k:
                # 确保方差为正且在合理范围内
                fixed_weights[k] = torch.clamp(fixed_weights[k], min=0.01, max=5.0)
        
        fixed_weights_list.append(fixed_weights)
    
    return fixed_weights_list

def verify_model_structure(model_state_dict, reference_state_dict, verbose=True):
    """
    验证模型结构与参考模型是否兼容
    """
    issues = []
    
    # 检查键是否匹配
    model_keys = set(model_state_dict.keys())
    ref_keys = set(reference_state_dict.keys())
    
    missing_keys = ref_keys - model_keys
    extra_keys = model_keys - ref_keys
    
    if missing_keys and verbose:
        issues.append(f"缺少的键: {missing_keys}")
    
    if extra_keys and verbose:
        issues.append(f"多余的键: {extra_keys}")
    
    # 检查每个共有键的形状是否匹配
    common_keys = model_keys.intersection(ref_keys)
    shape_mismatches = []
    
    for k in common_keys:
        if model_state_dict[k].shape != reference_state_dict[k].shape:
            shape_mismatches.append(
                f"键 {k}: 模型形状 {model_state_dict[k].shape} vs 参考形状 {reference_state_dict[k].shape}"
            )
    
    if shape_mismatches and verbose:
        issues.append("形状不匹配:")
        for mismatch in shape_mismatches:
            issues.append(f"  {mismatch}")
    
    if issues and verbose:
        print("模型结构验证发现以下问题:")
        for issue in issues:
            print(f"- {issue}")
    
    return len(issues) == 0, issues

def safe_model_aggregation(model_weights_list, client_weights, reference_model_state_dict=None):
    """
    安全的模型聚合，包含完整的验证和修复步骤
    """
    print("\n执行安全模型聚合...")
    
    if not model_weights_list:
        print("错误: 没有提供模型权重进行聚合")
        return None
    
    # 步骤1: 验证所有模型结构
    if reference_model_state_dict is not None:
        all_valid = True
        for i, weights in enumerate(model_weights_list):
            valid, _ = verify_model_structure(
                weights, 
                reference_model_state_dict, 
                verbose=False
            )
            if not valid:
                print(f"警告: 客户端 {i} 的模型结构与参考模型不兼容")
                all_valid = False
        
        if not all_valid:
            print("模型结构验证失败，将尝试修复以继续")
    
    # 步骤2: 修复批标准化层的异常值
    print("修复批标准化层参数...")
    fixed_weights_list = fix_batchnorm_before_aggregation(model_weights_list)
    
    # 步骤3: 执行改进的聚合
    print("执行加权平均聚合...")
    aggregated_weights = improved_model_aggregation(fixed_weights_list, client_weights)
    
    # 步骤4: 验证聚合结果是否有异常值
    if aggregated_weights:
        has_nan = False
        has_inf = False
        for k, v in aggregated_weights.items():
            if torch.isnan(v).any():
                print(f"警告: 键 {k} 包含 NaN 值")
                has_nan = True
            if torch.isinf(v).any():
                print(f"警告: 键 {k} 包含 Inf 值")
                has_inf = True
        
        if has_nan or has_inf:
            print("聚合权重包含 NaN 或 Inf 值，将替换为健康值")
            # 修复 NaN 和 Inf 值
            for k in aggregated_weights.keys():
                if torch.isnan(aggregated_weights[k]).any() or torch.isinf(aggregated_weights[k]).any():
                    if 'running_mean' in k:
                        aggregated_weights[k] = torch.zeros_like(aggregated_weights[k])
                    elif 'running_var' in k:
                        aggregated_weights[k] = torch.ones_like(aggregated_weights[k])
                    else:
                        # 用之前客户端的健康值替换
                        for weights in model_weights_list:
                            if k in weights and not torch.isnan(weights[k]).any() and not torch.isinf(weights[k]).any():
                                aggregated_weights[k] = weights[k].clone()
                                break
    
    print("模型聚合完成")
    return aggregated_weights

# 修改recalibrate_model_after_aggregation函数
def recalibrate_model_after_aggregation(model, dataloader, device, num_batches=10):
    """
    聚合后使用数据重新校准模型的批标准化层
    """
    print("\n聚合后重新校准批标准化层...")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # 验证模型参数是否在正确设备上
    for param in model.parameters():
        if param.device != device:
            print(f"警告: 发现模型参数不在{device}上，尝试修复...")
            param.data = param.data.to(device)
    
    # 设置所有批标准化层为训练模式，其他层为评估模式
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.train()
            # 重置统计量
            module.reset_running_stats()
            # 使用较小的动量参数以加快统计量更新
            module.momentum = 0.1
        else:
            module.eval()
    
    # 使用部分数据更新批标准化统计量
    with torch.no_grad():
        batch_count = 0
        for images, _ in dataloader:
            if batch_count >= num_batches:
                break
            
            # 确保数据在正确设备上
            images = images.to(device)
            
            try:
                # 前向传播
                _ = model(images)
                batch_count += 1
            except RuntimeError as e:
                print(f"前向传播出错: {str(e)}")
                print("尝试移动整个模型到设备...")
                model = model.to(device)
                # 再次尝试
                _ = model(images)
                batch_count += 1
    
    # 恢复所有层为评估模式
    model.eval()
    
    # 修复任何仍然异常的批标准化层
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'running_var'):
                with torch.no_grad():
                    # 限制方差范围以增加数值稳定性
                    module.running_var.clamp_(min=0.01, max=5.0)
            if hasattr(module, 'running_mean'):
                with torch.no_grad():
                    # 限制均值范围
                    module.running_mean.clamp_(min=-2.0, max=2.0)
    
    print("模型校准完成")
    return model

def reset_classifier_weights(model, num_classes=10, weight_scale=0.01):
    """
    重置分类器层的权重，当检测到权重分布异常时使用
    """
    # 查找可能的分类器层
    classifier_found = False
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            print(f"重置分类器层: {name}")
            # 重新初始化分类器权重
            nn.init.normal_(module.weight, mean=0.0, std=weight_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            classifier_found = True
    
    if not classifier_found:
        print("未找到匹配的分类器层")
    
    return model
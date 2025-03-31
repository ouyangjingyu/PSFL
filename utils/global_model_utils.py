import torch
import copy
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import logging
from model.resnet import resnet110_SFL_local_tier_7
from model.resnet import resnet56_SFL_local_tier_7
from model.resnet import resnet110_base
from model.resnet import resnet56_base
from model.resnet import create_classifier

def create_global_model(class_num, model_type='resnet110', device='cpu', groups_per_channel=32):
    """
    创建全局模型模板
    
    Args:
        class_num: 类别数量
        model_type: 模型类型，支持'resnet110'或'resnet56'
        device: 计算设备
        groups_per_channel: GroupNorm的每通道分组数，默认为32
        
    Returns:
        全局模型
    """
    if model_type == 'resnet110':
        # 传递 groups_per_channel 参数
        global_model = resnet110_base(classes=class_num, tier=1, groups_per_channel=groups_per_channel)
    elif model_type == 'resnet56':
        # 传递 groups_per_channel 参数
        global_model = resnet56_base(classes=class_num, tier=1, groups_per_channel=groups_per_channel)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 移动模型到指定设备
    global_model = global_model.to(device)
    
    print(f"创建全局{model_type}模型，参数数量: {sum(p.numel() for p in global_model.parameters())}")
    return global_model
    
def split_global_model(global_model, tier=1, class_num=10, model_type='resnet110', groups_per_channel=32):
    """
    将全局模型拆分成客户端和服务器模型
    
    Args:
        global_model: 全局模型
        tier: 客户端tier级别(1-7)
        class_num: 类别数量
        model_type: 模型类型，支持'resnet110'或'resnet56'
        groups_per_channel: GroupNorm的每通道分组数，默认为32
        
    Returns:
        client_model: 客户端模型
        server_model: 服务器模型
    """
    # 获取全局模型的状态字典
    if isinstance(global_model, dict):
        global_state = global_model
    else:
        global_state = global_model.state_dict()
    
    # 导入相应的模型创建函数
    if model_type == 'resnet110':
        model_func = resnet110_SFL_local_tier_7
    else:
        model_func = resnet56_SFL_local_tier_7
    
    # 创建指定tier级别的空客户端和服务器模型，传递 groups_per_channel 参数
    client_model, server_model = model_func(
        classes=class_num, 
        tier=tier, 
        local_loss=True, 
        groups_per_channel=groups_per_channel
    )
    
    # 获取模型状态字典
    client_state = client_model.state_dict()
    server_state = server_model.state_dict()
    
    # 从全局模型复制参数到客户端模型
    for key in client_state.keys():
        if key in global_state:
            client_state[key] = global_state[key].clone()
    
    # 从全局模型复制参数到服务器模型
    for key in server_state.keys():
        if key in global_state:
            server_state[key] = global_state[key].clone()
    
    # 加载状态字典到模型
    client_model.load_state_dict(client_state)
    server_model.load_state_dict(server_state)
    
    return client_model, server_model
def create_models_by_splitting(class_num, model_type='resnet110', device=None, groups_per_channel=32):
    """
    通过拆分创建全局模型，并为不同tier创建客户端和服务器模型
    
    Args:
        class_num: 类别数量
        model_type: 模型类型，支持'resnet110'或'resnet56'
        device: 计算设备，如果为None则自动选择
        groups_per_channel: GroupNorm的每通道分组数，默认为32
    
    Returns:
        client_models: 不同tier的客户端模型
        server_models: 不同tier的服务器模型
        unified_classifier: 全局分类器
        init_glob_model: 初始全局模型
        num_tiers: tier数量
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 根据模型类型创建不同的模型
    if model_type == 'resnet110':
        model_func = resnet110_SFL_local_tier_7
    elif model_type == 'resnet56':
        model_func = resnet56_SFL_local_tier_7
    else:
        model_func = resnet110_SFL_local_tier_7
    
    # 创建不同tier的客户端和服务器模型
    client_models = {}
    server_models = {}
    num_tiers = 7
    
    for tier in range(1, num_tiers+1):
        # 传递 groups_per_channel 参数
        client_model, server_model = model_func(
            class_num, 
            tier=tier, 
            local_loss=True, 
            groups_per_channel=groups_per_channel
        )
        client_models[tier] = client_model.to(device)
        server_models[tier] = server_model.to(device)
    
    # 创建用于全局模型初始化的模板 - 传递 groups_per_channel 参数
    if model_type == 'resnet110':
        init_glob_model = resnet110_base(class_num, groups_per_channel=groups_per_channel)
    else:
        init_glob_model = resnet56_base(class_num, groups_per_channel=groups_per_channel)
    init_glob_model = init_glob_model.to(device)
    
    # 创建优化的全局分类器
    final_channels = 64 * 4  # 64 * Bottleneck.expansion = 256
    unified_classifier = create_classifier(final_channels, class_num, is_global=True)
    unified_classifier = unified_classifier.to(device)
    
    return client_models, server_models, unified_classifier, init_glob_model, num_tiers

def combine_to_global_model(client_models_params, server_models_dict, client_tiers, 
                           init_glob_model, device=None):
    """
    结合客户端和服务器模型参数创建全局模型，确保设备一致性
    
    Args:
        client_models_params: 客户端模型参数字典
        server_models_dict: 服务器模型字典
        client_tiers: 客户端tier信息字典
        init_glob_model: 初始全局模型
        device: 指定的计算设备，默认自动选择
        
    Returns:
        global_model: 聚合后的全局模型状态字典
    """
    # 确定目标设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"全局模型聚合使用设备: {device}")
    
    try:
        # 初始化全局模型
        global_model = {}
        if init_glob_model is not None:
            # 复制模板模型的状态字典，确保在正确设备上
            for k, v in init_glob_model.state_dict().items():
                global_model[k] = v.clone().to(device)
        
        # 统计各tier的客户端数量
        tier_counts = {}
        for client_id, tier in client_tiers.items():
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # 按tier收集和聚合模型参数
        tier_models = {}
        
        # 1. 收集每个tier的客户端模型参数
        for client_id, params in client_models_params.items():
            tier = client_tiers.get(client_id)
            if tier is None:
                logging.warning(f"客户端 {client_id} 没有tier信息，跳过")
                continue
                
            if tier not in tier_models:
                tier_models[tier] = {'client': [], 'server': None}
            
            # 将客户端模型参数移到目标设备并添加到列表
            client_params_on_device = {}
            for k, v in params.items():
                # 跳过归一化层参数和分类器参数
                if any(x in k for x in ['norm', 'gn', 'classifier', 'projection']):
                    continue
                    
                try:
                    client_params_on_device[k] = v.to(device)
                except Exception as e:
                    logging.error(f"移动参数 {k} 到设备 {device} 失败: {str(e)}")
                    continue
            
            tier_models[tier]['client'].append(client_params_on_device)
            
            # 如果还没有收集该tier的服务器模型，则获取它
            if tier_models[tier]['server'] is None and client_id in server_models_dict:
                server_model = server_models_dict[client_id]
                server_params = server_model.state_dict()
                server_params_on_device = {}
                
                for k, v in server_params.items():
                    # 跳过归一化层参数和分类器参数
                    if any(x in k for x in ['norm', 'gn', 'classifier', 'projection']):
                        continue
                        
                    try:
                        server_params_on_device[k] = v.to(device)
                    except Exception as e:
                        logging.error(f"移动服务器参数 {k} 到设备 {device} 失败: {str(e)}")
                        continue
                
                tier_models[tier]['server'] = server_params_on_device
        
        # 2. 聚合每个tier的客户端模型，然后与对应的服务器模型组合
        feature_extraction_count = 0
        total_feature_extraction = sum(1 for k in global_model.keys() 
                                     if not any(x in k for x in ['classifier', 'projection', 'gn']))
        
        for tier, models in tier_models.items():
            client_list = models['client']
            server_model = models['server']
            
            if not client_list or not server_model:
                logging.warning(f"Tier {tier} 缺少客户端或服务器模型，跳过")
                continue
            
            # 聚合该tier的客户端模型
            try:
                agg_client_model = {}
                for key in client_list[0].keys():
                    # 初始化为零张量
                    agg_client_model[key] = torch.zeros_like(client_list[0][key], device=device)
                    
                    # 计算平均值
                    valid_models = 0
                    for client_model in client_list:
                        if key in client_model:
                            agg_client_model[key] += client_model[key]
                            valid_models += 1
                    
                    if valid_models > 0:
                        agg_client_model[key] /= valid_models
            except Exception as e:
                logging.error(f"聚合Tier {tier} 客户端模型失败: {str(e)}")
                continue
            
            # 将聚合的客户端模型和服务器模型组合到全局模型中
            for key, param in agg_client_model.items():
                if key in global_model:
                    try:
                        global_model[key] = param.clone().to(device)
                        feature_extraction_count += 1
                    except Exception as e:
                        logging.error(f"复制客户端参数 {k} 到全局模型失败: {str(e)}")
            
            for key, param in server_model.items():
                if key in global_model:
                    try:
                        global_model[key] = param.clone().to(device)
                        feature_extraction_count += 1
                    except Exception as e:
                        logging.error(f"复制服务器参数 {k} 到全局模型失败: {str(e)}")
        
        # 输出统计信息
        logging.info(f"全局聚合模型: 成功聚合了 {feature_extraction_count}/{total_feature_extraction} 个特征提取层参数")
        
        return global_model
    
    except Exception as e:
        logging.error(f"全局模型聚合失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 返回初始模型作为备选
        if init_glob_model is not None:
            return {k: v.clone().to(device) for k, v in init_glob_model.state_dict().items()}
        else:
            return {}
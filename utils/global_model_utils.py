import torch
import copy
import numpy as np
from collections import OrderedDict
import torch.nn as nn

def create_global_model(class_num, model_type='resnet110', device='cpu'):
    """
    创建全局模型模板
    
    Args:
        class_num: 类别数量
        model_type: 模型类型，支持'resnet110'或'resnet56'
        device: 计算设备
        
    Returns:
        全局模型
    """
    if model_type == 'resnet110':
        from model.resnet import resnet110_SFL_fedavg_base
        # 移除重复的local_loss参数，因为该函数内部已经设置了local_loss=True
        global_model = resnet110_SFL_fedavg_base(classes=class_num, tier=1)
    elif model_type == 'resnet56':
        from model.resnet import resnet56_base
        global_model = resnet56_base(classes=class_num, tier=1)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 移动模型到指定设备
    global_model = global_model.to(device)
    
    print(f"创建全局{model_type}模型，参数数量: {sum(p.numel() for p in global_model.parameters())}")
    return global_model
    
def split_global_model(global_model, tier=1, class_num=10, model_type='resnet110'):
    """
    将全局模型拆分成客户端和服务器模型
    
    Args:
        global_model: 全局模型
        tier: 客户端tier级别(1-7)
        class_num: 类别数量
        model_type: 模型类型，支持'resnet110'或'resnet56'
        
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
        from model.resnet import resnet110_SFL_local_tier_7
        model_func = resnet110_SFL_local_tier_7
    else:
        from model.resnet import resnet56_SFL_local_tier_7
        model_func = resnet56_SFL_local_tier_7
    
    # 创建指定tier级别的空客户端和服务器模型
    client_model, server_model = model_func(classes=class_num, tier=tier, local_loss=True)
    
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

def create_models_by_splitting(class_num, model_type='resnet110', device='cpu'):
    """
    通过拆分全局模型创建所有tier级别的客户端和服务器模型
    
    Args:
        class_num: 类别数量
        model_type: 模型类型，支持'resnet110'或'resnet56'
        device: 计算设备
        
    Returns:
        client_models: 客户端模型字典，键为tier
        server_models: 服务器模型字典，键为tier
        unified_classifier: 统一分类器
        global_model: 全局模型
        num_tiers: tier数量
    """
    # 创建全局模型
    global_model = create_global_model(class_num, model_type, device)
    
    # 提取全局分类器
    if hasattr(global_model, 'classifier'):
        unified_classifier = copy.deepcopy(global_model.classifier)
    else:
        # 如果没有直接的classifier属性，创建一个新的分类器
        from utils.enhanced_model_architecture import UnifiedClassifier
        unified_classifier = UnifiedClassifier(256, [128, 64], class_num, dropout_rate=0.5)
    
    # 初始化客户端和服务器模型字典
    client_models = {}
    server_models = {}
    num_tiers = 7  # 默认7个tier级别
    
    # 为每个tier创建客户端和服务器模型
    for tier in range(1, num_tiers + 1):
        client_model, server_model = split_global_model(global_model, tier, class_num, model_type)
        
        # 确保客户端模型有本地分类器
        if not hasattr(client_model, 'classifier') or client_model.classifier is None:
            print(f"警告: Tier {tier} 客户端模型没有分类器，可能需要确保local_loss=True")
        
        # 保存到字典
        client_models[tier] = client_model
        server_models[tier] = server_model
        
        print(f"已创建 Tier {tier} 模型")
    
    return client_models, server_models, unified_classifier, global_model, num_tiers

def combine_to_global_model(client_models_dict, server_models_dict, client_tiers, global_model_template):
    """
    将不同tier的客户端和服务器模型组合回全局模型
    
    Args:
        client_models_dict: 客户端模型字典，键为客户端ID
        server_models_dict: 服务器模型字典，键为客户端ID
        client_tiers: 客户端tier字典，键为客户端ID
        global_model_template: 全局模型模板
        
    Returns:
        global_model_params: 组合后的全局模型参数
    """
    # 获取全局模型状态字典作为模板
    if isinstance(global_model_template, dict):
        template_state = global_model_template
    else:
        template_state = global_model_template.state_dict()
    
    # 创建新的全局状态字典 - 确保在CPU上
    global_state = OrderedDict()
    for key, param in template_state.items():
        global_state[key] = torch.zeros_like(param, device='cpu')
    
    # 统计每个参数的贡献客户端数
    param_counts = {key: 0 for key in global_state.keys()}
    
    # 定义每个tier对应的层前缀
    tier_layer_map = {
        1: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'classifier', 'projection'],
        2: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'classifier', 'projection'],
        3: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'classifier', 'projection'],
        4: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'classifier', 'projection'],
        5: ['conv1', 'bn1', 'layer1', 'layer2', 'classifier', 'projection'],
        6: ['conv1', 'bn1', 'layer1', 'classifier', 'projection'],
        7: ['conv1', 'bn1', 'classifier', 'projection']
    }
    
    server_layer_map = {
        1: ['classifier', 'projection'],  # Tier 1客户端包含所有特征层
        2: ['layer6', 'classifier', 'projection'],
        3: ['layer5', 'layer6', 'classifier', 'projection'],
        4: ['layer4', 'layer5', 'layer6', 'classifier', 'projection'],
        5: ['layer3', 'layer4', 'layer5', 'layer6', 'classifier', 'projection'],
        6: ['layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'classifier', 'projection'],
        7: ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'classifier', 'projection']
    }
    
    # 将客户端和服务器模型参数组合到全局模型
    for client_id, tier in client_tiers.items():
        if client_id in client_models_dict and client_id in server_models_dict:
            client_model = client_models_dict[client_id]
            server_model = server_models_dict[client_id]
            
            # 获取状态字典
            if isinstance(client_model, dict):
                client_state = client_model
            else:
                client_state = client_model.state_dict()
                
            if isinstance(server_model, dict):
                server_state = server_model
            else:
                server_state = server_model.state_dict()
            
            # 从客户端模型获取参数
            for key in client_state.keys():
                if key in global_state:
                    prefix = key.split('.')[0] if '.' in key else key
                    if prefix in tier_layer_map[tier]:
                        # 确保参数在CPU上，避免设备不一致
                        param = client_state[key].detach().cpu()
                        global_state[key] += param
                        param_counts[key] += 1
            
            # 从服务器模型获取参数
            for key in server_state.keys():
                if key in global_state:
                    prefix = key.split('.')[0] if '.' in key else key
                    if prefix in server_layer_map[tier]:
                        # 确保参数在CPU上，避免设备不一致
                        param = server_state[key].detach().cpu()
                        global_state[key] += param
                        param_counts[key] += 1
    
    # 计算参数平均值
    for key in global_state.keys():
        if param_counts[key] > 0:
            global_state[key] = global_state[key] / param_counts[key]
        else:
            # 如果没有客户端贡献该参数，使用模板值
            global_state[key] = template_state[key].clone().cpu()
    
    return global_state
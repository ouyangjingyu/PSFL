import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import copy
from memory_utils import free_memory, safe_model_copy, safe_to_device

def train_client_with_improved_features(client_id, client_model, server_model, device, 
                         client_manager, round_idx, coordinator, local_epochs=None, split_rounds=1, global_classifier=None, **kwargs):
    """
    改进的客户端训练函数，优化特征传递和内存管理
    """
    try:
        client = client_manager.get_client(client_id)
        if client is None:
            return {'error': f"客户端 {client_id} 不存在"}
        
        # 记录原始设备
        original_device = None
        for param in client_model.parameters():
            original_device = param.device
            break
        
        # 确保客户端设备与传入的设备一致
        client.device = device
        print(f"客户端 {client_id} 使用设备: {device}")

        # 安全复制模型而不是深度复制
        client_model_copy = safe_model_copy(client_model, device)
        
        # 记录开始时间
        start_time = time.time()
        
        # 第1步：本地训练
        client_state, local_stats = client.local_train(client_model_copy, local_epochs)
        local_train_time = local_stats['time']
        
        # 释放不需要的内存
        free_memory()
        
        # 第2步：拆分学习训练 - 使用改进的特征传递
        server_model_copy = safe_model_copy(server_model, device)
        
        # 为全局分类器创建一个安全副本
        if global_classifier is not None:
            global_classifier_copy = safe_model_copy(global_classifier, device)
        else:
            global_classifier_copy = None
        
        # 使用改进的特征归一化进行拆分学习训练
        client_state_sl, server_state, sl_stats = train_split_learning_improved(
            client,
            client_model_copy, 
            server_model_copy, 
            global_classifier=global_classifier_copy,
            rounds=split_rounds
        )
        
        sl_train_time = sl_stats['time']
        
        # 释放内存
        free_memory()
        
        # 合并结果并计算总训练时间
        total_time = local_train_time + sl_train_time
        
        # 记录通信量
        communication_size_mb = sl_stats.get('data_transmitted_mb', 0)
        
        # 计算客户端数据量
        data_size = sl_stats.get('data_size', 0)
        
        # 获取学习率
        local_lr = local_stats.get('lr_final', client.learning_rate)
        
        # 合并统计信息
        merged_stats = {
            'local_train': local_stats,
            'split_learning': sl_stats,
            
            # 全局分类器结果（用于兼容旧代码）
            'loss': sl_stats['loss'],
            'accuracy': sl_stats['accuracy'],
            
            # 本地分类器的拆分学习结果
            'local_sl_loss': sl_stats.get('local_loss', 0),
            'local_sl_accuracy': sl_stats.get('local_accuracy', 0),
            
            # 全局分类器的拆分学习结果
            'global_sl_loss': sl_stats.get('global_loss', 0), 
            'global_sl_accuracy': sl_stats.get('global_accuracy', 0),
            
            # 本地训练结果
            'local_train_loss': local_stats.get('loss', 0),
            'local_train_accuracy': local_stats.get('accuracy', 0),
            
            'time': total_time,
            'local_train_time': local_train_time,
            'sl_train_time': sl_train_time,
            'data_size': data_size,
            'communication_mb': communication_size_mb,
            'lr': local_lr,
            
            # 保存训练后的模型状态
            'client_model_state': client_state_sl,
            'server_model_state': server_state
        }
        
        # 记录到wandb
        client_metrics = {
            # 客户端本地训练结果
            f"client_{client_id}/round_{round_idx}/local_train_loss": local_stats.get('loss', 0),
            f"client_{client_id}/round_{round_idx}/local_train_accuracy": local_stats.get('accuracy', 0),
            
            # 拆分学习本地分类器结果
            f"client_{client_id}/round_{round_idx}/local_sl_loss": sl_stats.get('local_loss', 0),
            f"client_{client_id}/round_{round_idx}/local_sl_accuracy": sl_stats.get('local_accuracy', 0),
            
            # 拆分学习全局分类器结果
            f"client_{client_id}/round_{round_idx}/global_sl_loss": sl_stats.get('global_loss', 0),
            f"client_{client_id}/round_{round_idx}/global_sl_accuracy": sl_stats.get('global_accuracy', 0),
            
            # 兼容旧代码
            f"client_{client_id}/round_{round_idx}/train_loss": sl_stats['loss'],
            f"client_{client_id}/round_{round_idx}/train_accuracy": sl_stats['accuracy'],
            
            f"client_{client_id}/round_{round_idx}/local_train_time": local_train_time,
            f"client_{client_id}/round_{round_idx}/sl_train_time": sl_train_time,
            f"client_{client_id}/round_{round_idx}/total_time": total_time,
            f"client_{client_id}/round_{round_idx}/communication_mb": communication_size_mb,
            f"client_{client_id}/round_{round_idx}/learning_rate": local_lr,
            f"client_{client_id}/tier": client.tier
        }
        try:
            import wandb
            wandb.log(client_metrics)
        except:
            pass
        
        # 清理并返回到原始设备
        if original_device is not None and original_device != device:
            # 将模型状态移回原始设备
            for k, v in client_state_sl.items():
                client_state_sl[k] = v.to(original_device)
            for k, v in server_state.items():
                server_state[k] = v.to(original_device)
        
        # 释放内存
        free_memory()
        
        # 返回最终的客户端和服务器模型状态以及训练统计信息
        return merged_stats
        
    except Exception as e:
        import traceback
        error_msg = f"客户端 {client_id} 训练失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # 释放内存
        free_memory()
        return {'error': error_msg}

def normalize_features(features, eps=1e-8):
    """
    归一化特征以解决尺度问题
    
    Args:
        features: 输入特征
        eps: 小常数，避免除以零
        
    Returns:
        归一化后的特征
    """
    # 计算特征尺度
    feature_scale = torch.norm(features.reshape(features.size(0), -1), dim=1, keepdim=True)
    
    # 如果是卷积特征，需要调整形状
    if len(features.shape) > 2:
        # 创建适合广播的尺度
        scale_shape = [features.size(0)] + [1] * (len(features.shape) - 1)
        feature_scale = feature_scale.reshape(scale_shape)
    
    # 避免除以零
    feature_scale = torch.max(feature_scale, torch.ones_like(feature_scale) * eps)
    
    # 归一化
    normalized_features = features / feature_scale
    
    return normalized_features

def train_split_learning_improved(client, client_model, server_model, global_classifier=None, rounds=1):
    """
    改进的拆分学习训练函数，解决特征尺度问题
    """
    device = client.device
    
    # 设置模型为训练模式
    client_model.train()
    server_model.train()
    if global_classifier is not None:
        global_classifier.train()
    
    # 初始化客户端优化器
    if not hasattr(client, 'optimizer') or client.optimizer is None:
        client.init_optimizer(client_model)
    client_optimizer = client.optimizer
    
    # 初始化服务器优化器
    server_optimizer = torch.optim.Adam(
        server_model.parameters(),
        lr=client.learning_rate,
        weight_decay=5e-4
    )
    
    # 初始化全局分类器优化器
    if global_classifier is not None:
        global_classifier_optimizer = torch.optim.Adam(
            global_classifier.parameters(),
            lr=client.learning_rate,
            weight_decay=5e-4
        )
    
    # 记录数据传输大小
    intermediate_data_size = 0
    
    # 训练统计信息
    stats = {
        # 全局分类器指标
        'global_loss': [],
        'global_accuracy': [],
        
        # 本地分类器指标
        'local_loss': [],
        'local_accuracy': [],
        
        # 其他统计信息
        'time': 0,
        'data_transmitted_mb': 0
    }
    
    # 开始计时
    time_start = time.time()
    
    for round_idx in range(rounds):
        # 本地分类器指标
        round_local_loss = 0.0
        round_local_correct = 0
        # 全局分类器指标
        round_global_loss = 0.0
        round_global_correct = 0
        # 样本数
        round_samples = 0
        
        for batch_idx, (images, labels) in enumerate(client.ldr_train):
            images, labels = images.to(device), labels.to(device)
            
            # 0. 清零梯度
            client_optimizer.zero_grad()
            server_optimizer.zero_grad()
            if global_classifier is not None:
                global_classifier_optimizer.zero_grad()
            
            # 1. 客户端前向传播
            client_outputs = client_model(images)
            
            # 客户端模型应该返回(logits, features)
            if isinstance(client_outputs, tuple):
                client_logits, client_features = client_outputs
                
                # 计算本地分类器损失和准确率
                local_loss = client.criterion(client_logits, labels)
                
                _, local_predicted = torch.max(client_logits.data, 1)
                local_batch_correct = (local_predicted == labels).sum().item()
                
                # 更新本地分类器统计
                round_local_loss += local_loss.item() * labels.size(0)
                round_local_correct += local_batch_correct
            else:
                # 如果没有返回元组，则假设只返回了特征
                client_features = client_outputs
                client_logits = None
                local_loss = 0
            
            # 2. 特征归一化 - 关键步骤，解决尺度问题
            client_features_norm = normalize_features(client_features)
            
            # 确保client_features需要梯度
            if not client_features_norm.requires_grad:
                client_features_norm = client_features_norm.clone().detach().requires_grad_(True)
            
            # 记录中间特征大小
            features_size_bytes = client_features_norm.nelement() * client_features_norm.element_size()
            intermediate_data_size += features_size_bytes + labels.nelement() * labels.element_size()
            
            # 3. 服务器前向传播
            server_features = server_model(client_features_norm)
            
            # 4. 全局分类器前向传播
            if global_classifier is not None:
                global_outputs = global_classifier(server_features)
                
                # 计算全局损失
                global_loss = client.criterion(global_outputs, labels)
                
                # 计算全局准确率
                _, global_predicted = torch.max(global_outputs.data, 1)
                global_batch_correct = (global_predicted == labels).sum().item()
                
                # 更新全局分类器统计
                round_global_loss += global_loss.item() * labels.size(0)
                round_global_correct += global_batch_correct
                
                # 5. 反向传播 - 使用全局损失
                global_loss.backward()
                
                # 更新全局分类器
                global_classifier_optimizer.step()
            else:
                # 如果没有全局分类器，则使用服务器特征作为输出
                global_outputs = server_features
                global_loss = client.criterion(global_outputs, labels)
                global_loss.backward()
            
            # 6. 获取服务器特征的梯度，用于客户端反向传播
            server_features_grad = None
            if hasattr(server_features, 'grad') and server_features.grad is not None:
                server_features_grad = server_features.grad.clone()
            
            # 7. 更新服务器模型
            server_optimizer.step()
            
            # 8. 客户端反向传播和更新
            # 如果使用了本地损失，也进行反向传播
            if local_loss > 0:
                local_loss.backward(retain_graph=True)
            
            # 从服务器传回梯度
            if server_features_grad is not None and client_features_norm.grad is None:
                client_features_norm.backward(server_features_grad)
            
            # 9. 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=5.0)
            
            # 10. 更新客户端模型
            client_optimizer.step()
            
            # 更新样本数
            round_samples += labels.size(0)
        
        # 计算每个轮次的平均值
        if round_samples > 0:
            # 本地分类器指标
            round_local_loss /= round_samples
            round_local_acc = 100.0 * round_local_correct / round_samples
            stats['local_loss'].append(round_local_loss)
            stats['local_accuracy'].append(round_local_acc)
            
            # 全局分类器指标
            round_global_loss /= round_samples
            round_global_acc = 100.0 * round_global_correct / round_samples
            stats['global_loss'].append(round_global_loss)
            stats['global_accuracy'].append(round_global_acc)
            
            print(f"客户端 {client.idx} - 拆分学习轮次 {round_idx+1}/{rounds}:")
            print(f"  本地分类器: 损失={round_local_loss:.4f}, 准确率={round_local_acc:.2f}%")
            print(f"  全局分类器: 损失={round_global_loss:.4f}, 准确率={round_global_acc:.2f}%")
    
    # 计算训练耗时
    training_time = time.time() - time_start
    
    # 计算中间数据大小（MB）
    intermediate_data_size_mb = intermediate_data_size / (1024 ** 2)
    
    # 更新训练统计信息
    avg_local_loss = np.mean(stats['local_loss']) if stats['local_loss'] else 0
    avg_local_acc = np.mean(stats['local_accuracy']) if stats['local_accuracy'] else 0
    avg_global_loss = np.mean(stats['global_loss']) if stats['global_loss'] else 0
    avg_global_acc = np.mean(stats['global_accuracy']) if stats['global_accuracy'] else 0
    
    # 更新结果统计信息
    stats['time'] = training_time
    stats['data_transmitted_mb'] = intermediate_data_size_mb
    stats['data_size'] = len(client.ldr_train.dataset) if hasattr(client.ldr_train, 'dataset') else 0
    
    # 为了兼容旧代码，保留原有的loss和accuracy字段，使用全局分类器的结果
    stats['loss'] = avg_global_loss
    stats['accuracy'] = avg_global_acc
    
    # 释放内存
    free_memory()
    
    return client_model.state_dict(), server_model.state_dict(), stats
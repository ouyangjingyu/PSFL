import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict

# 项目特定导入
from data.cifar10_eval_dataset import get_cifar10_proxy_dataset
from utils.client_clustering import extract_model_predictions


def comprehensive_evaluation_with_comparison(
    global_model, 
    init_global_model,  # 添加初始全局模型参数
    client_models_dict, 
    server_models_dict, 
    unified_classifier,
    eval_dataset,
    client_clusters,
    client_resources,
    device,
    class_num
):
    """
    全面评估不同模型配置在均衡测试数据集上的性能，并与初始全局模型进行比较
    
    Args:
        global_model: 当前全局聚合模型
        init_global_model: 初始化的全局模型
        client_models_dict: 客户端模型字典
        server_models_dict: 服务器模型字典
        unified_classifier: 全局分类器
        eval_dataset: 均衡评估数据集
        client_clusters: 客户端聚类信息
        client_resources: 客户端资源信息（包含tier信息）
        device: 计算设备
        class_num: 类别数量
    
    Returns:
        results: 包含所有评估结果的字典
    """
    # 创建均衡测试数据集的数据加载器
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, shuffle=False
    )
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    # 1. 评估当前全局模型
    global_model.eval()
    global_correct = 0
    global_total = 0
    global_loss = 0
    global_class_correct = [0] * class_num
    global_class_total = [0] * class_num
    
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            
            # 处理可能的元组输出
            if isinstance(output, tuple):
                output = output[0]
            
            loss = criterion(output, target)
            global_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            global_total += target.size(0)
            global_correct += (predicted == target).sum().item()
            
            # 每个类别的准确率
            for i in range(len(target)):
                label = target[i]
                global_class_total[label] += 1
                if predicted[i] == label:
                    global_class_correct[label] += 1
    
    # 计算全局模型指标
    global_accuracy = 100.0 * global_correct / global_total
    global_loss = global_loss / len(eval_loader)
    global_per_class_acc = [100.0 * c / t if t > 0 else 0 for c, t in zip(global_class_correct, global_class_total)]
    
    results['global'] = {
        'accuracy': global_accuracy,
        'loss': global_loss,
        'per_class_accuracy': global_per_class_acc
    }
    
    # 2. 评估初始全局模型 (新增部分)
    if init_global_model is not None:
        init_global_model.eval()
        init_global_correct = 0
        init_global_total = 0
        init_global_loss = 0
        init_global_class_correct = [0] * class_num
        init_global_class_total = [0] * class_num
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(device), target.to(device)
                output = init_global_model(data)
                
                # 处理可能的元组输出
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = criterion(output, target)
                init_global_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                init_global_total += target.size(0)
                init_global_correct += (predicted == target).sum().item()
                
                # 每个类别的准确率
                for i in range(len(target)):
                    label = target[i]
                    init_global_class_total[label] += 1
                    if predicted[i] == label:
                        init_global_class_correct[label] += 1
        
        # 计算初始全局模型指标
        init_global_accuracy = 100.0 * init_global_correct / init_global_total
        init_global_loss = init_global_loss / len(eval_loader)
        init_global_per_class_acc = [100.0 * c / t if t > 0 else 0 for c, t in zip(init_global_class_correct, init_global_class_total)]
        
        results['init_global'] = {
            'accuracy': init_global_accuracy,
            'loss': init_global_loss,
            'per_class_accuracy': init_global_per_class_acc
        }
    
    # 3. 初始化跨客户端的平均指标
    client_local_acc_sum = 0
    client_server_global_acc_sum = 0
    client_global_acc_sum = 0
    valid_client_global_count = 0
    
    # 对每个客户端评估不同的模型组合
    for client_id, client_model in client_models_dict.items():
        client_results = {}
        client_model = client_model.to(device)
        server_model = server_models_dict[client_id].to(device)
        
        # 获取客户端tier以便正确适配特征
        client_tier = client_resources[client_id]["storage_tier"]
        
        # A. 评估客户端本地模型
        client_model.eval()
        local_correct = 0
        local_total = 0
        local_loss = 0
        local_class_correct = [0] * class_num
        local_class_total = [0] * class_num
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(device), target.to(device)
                output, _ = client_model(data)  # 从本地模型获取logits
                
                loss = criterion(output, target)
                local_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                local_total += target.size(0)
                local_correct += (predicted == target).sum().item()
                
                # 每个类别的准确率
                for i in range(len(target)):
                    label = target[i]
                    local_class_total[label] += 1
                    if predicted[i] == label:
                        local_class_correct[label] += 1
        
        local_accuracy = 100.0 * local_correct / local_total
        local_loss = local_loss / len(eval_loader)
        local_per_class_acc = [100.0 * c / t if t > 0 else 0 for c, t in zip(local_class_correct, local_class_total)]
        
        client_results['local'] = {
            'accuracy': local_accuracy,
            'loss': local_loss,
            'per_class_accuracy': local_per_class_acc
        }
        
        # B. 评估客户端-服务器-全局分类器组合
        # 修改客户端-服务器-全局分类器评估部分
        client_model.eval()
        server_model.eval()
        unified_classifier.eval()
        server_global_correct = 0
        server_global_total = 0
        server_global_loss = 0
        
        # 收集预测分布，用于调试
        server_global_predictions = torch.zeros(class_num, device=device)
        
        with torch.no_grad():
            for data, target in eval_loader:
                try:
                    # 确保数据在正确设备上
                    data, target = data.to(device), target.to(device)
                    
                    # 获取客户端模型特征
                    _, client_features = client_model(data)
                    
                    # 确保特征在正确设备上
                    if client_features.device != device:
                        client_features = client_features.to(device)
                    
                    # 获取服务器模型特征
                    server_features = server_model(client_features)
                    
                    # 处理可能的元组输出
                    if isinstance(server_features, tuple):
                        server_features = server_features[0]
                    
                    # 数值稳定性处理 - 检测并处理NaN/Inf值
                    if torch.isnan(server_features).any() or torch.isinf(server_features).any():
                        # 替换NaN/Inf为0
                        server_features = torch.where(
                            torch.isnan(server_features) | torch.isinf(server_features),
                            torch.zeros_like(server_features),
                            server_features
                        )
                        logging.warning("检测到NaN/Inf特征值，已替换为零")
                    
                    # 池化（如果需要）
                    if len(server_features.shape) > 2:
                        server_features = F.adaptive_avg_pool2d(server_features, (1, 1))
                        server_features = server_features.view(server_features.size(0), -1)
                    
                    # 特征归一化处理，提高数值稳定性
                    norm = torch.norm(server_features, p=2, dim=1, keepdim=True)
                    norm = torch.clamp(norm, min=1e-7)  # 避免除零
                    server_features = server_features / norm
                    
                    # 应用全局分类器
                    global_logits = unified_classifier(server_features)
                    
                    # 梯度裁剪（模拟）- 限制logits范围，防止数值溢出
                    global_logits = torch.clamp(global_logits, -100, 100)
                    
                    # 检查数值稳定性
                    if torch.isnan(global_logits).any() or torch.isinf(global_logits).any():
                        logging.warning("检测到NaN/Inf分类器输出，跳过此批次")
                        continue
                    
                    loss = criterion(global_logits, target)
                    server_global_loss += loss.item()
                    
                    _, predicted = torch.max(global_logits.data, 1)
                    server_global_total += target.size(0)
                    server_global_correct += (predicted == target).sum().item()
                    
                    # 记录预测分布
                    for c in range(class_num):
                        server_global_predictions[c] += (predicted == c).sum().item()
                    
                except Exception as e:
                    logging.error(f"评估客户端-服务器-全局分类器时出错: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
        
        if server_global_total > 0:
            server_global_accuracy = 100.0 * server_global_correct / server_global_total
            server_global_loss = server_global_loss / len(eval_loader) if len(eval_loader) > 0 else float('inf')
            
            # 记录预测分布
            pred_distribution = server_global_predictions.cpu().numpy()
            if server_global_total > 0:
                pred_distribution = (pred_distribution / server_global_total) * 100
            
            logging.info(f"客户端-服务器-全局分类器预测分布:")
            for c in range(class_num):
                logging.info(f"  类别 {c}: {pred_distribution[c]:.2f}%")
        else:
            server_global_accuracy = 0
            server_global_loss = float('inf')
        
        client_results['server_global'] = {
            'accuracy': server_global_accuracy,
            'loss': server_global_loss,
            'prediction_distribution': pred_distribution.tolist() if 'pred_distribution' in locals() else []
        }
        
        # C. 尝试客户端直接与全局分类器结合（如果tier结构允许）
        try:
            client_model.eval()
            unified_classifier.eval()
            client_global_correct = 0
            client_global_total = 0
            client_global_loss = 0
            
            with torch.no_grad():
                for data, target in eval_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # 获取客户端模型特征
                    _, client_features = client_model(data)
                    
                    # 根据tier适配特征
                    if len(client_features.shape) > 2:  # 如果是卷积特征，需要池化
                        pooled_features = F.adaptive_avg_pool2d(client_features, (1, 1))
                        pooled_features = pooled_features.view(pooled_features.size(0), -1)
                        
                        # 根据tier，可能需要投影层将特征维度调整为分类器输入维度
                        final_features = pooled_features
                        
                        # 如果客户端模型有投影层，使用它
                        if hasattr(client_model, 'projection'):
                            final_features = client_model.projection(pooled_features)
                    else:
                        final_features = client_features
                        
                    # 应用全局分类器直接预测
                    client_global_logits = unified_classifier(final_features)
                    
                    loss = criterion(client_global_logits, target)
                    client_global_loss += loss.item()
                    _, predicted = torch.max(client_global_logits.data, 1)
                    client_global_total += target.size(0)
                    client_global_correct += (predicted == target).sum().item()
            
            if client_global_total > 0:
                client_global_accuracy = 100.0 * client_global_correct / client_global_total
                client_global_loss = client_global_loss / len(eval_loader)
                
                client_results['client_global'] = {
                    'accuracy': client_global_accuracy,
                    'loss': client_global_loss
                }
                client_global_acc_sum += client_global_accuracy
                valid_client_global_count += 1
            else:
                client_results['client_global'] = {
                    'accuracy': 'N/A - 特征维度不兼容',
                    'loss': 'N/A'
                }
        except Exception as e:
            client_results['client_global'] = {
                'accuracy': f'错误: {str(e)}',
                'loss': 'N/A'
            }
        
        # 记录该客户端的所有结果
        results[f'client_{client_id}'] = client_results
        
        # 添加到平均值
        client_local_acc_sum += local_accuracy
        client_server_global_acc_sum += server_global_accuracy
    
    # 计算客户端的平均指标
    num_clients = len(client_models_dict)
    results['averages'] = {
        'local_accuracy': client_local_acc_sum / num_clients,
        'server_global_accuracy': client_server_global_acc_sum / num_clients,
        'client_global_accuracy': client_global_acc_sum / valid_client_global_count if valid_client_global_count > 0 else 'N/A'
    }
    
    # 计算每个聚类的指标
    for cluster_id, client_ids in client_clusters.items():
        cluster_local_acc = 0
        cluster_server_global_acc = 0
        cluster_client_global_acc = 0
        cluster_client_global_count = 0
        
        for client_id in client_ids:
            if f'client_{client_id}' in results:
                client_data = results[f'client_{client_id}']
                cluster_local_acc += client_data['local']['accuracy']
                cluster_server_global_acc += client_data['server_global']['accuracy']
                
                if 'client_global' in client_data and isinstance(client_data['client_global']['accuracy'], (int, float)):
                    cluster_client_global_acc += client_data['client_global']['accuracy']
                    cluster_client_global_count += 1
        
        num_clients_in_cluster = len(client_ids)
        if num_clients_in_cluster > 0:
            results[f'cluster_{cluster_id}'] = {
                'local_accuracy': cluster_local_acc / num_clients_in_cluster,
                'server_global_accuracy': cluster_server_global_acc / num_clients_in_cluster,
                'client_global_accuracy': cluster_client_global_acc / cluster_client_global_count if cluster_client_global_count > 0 else 'N/A'
            }
    
    return results

def print_evaluation_results(eval_results, round_idx):
    """
    打印评估结果，包括初始全局模型和当前全局模型的比较
    
    Args:
        eval_results: 评估结果字典
        round_idx: 当前轮次索引
    """
    print(f"\n===== 第 {round_idx+1} 轮评估结果 =====")
    
    # 打印当前全局模型结果
    global_acc = eval_results['global']['accuracy']
    global_loss = eval_results['global']['loss']
    print(f"\n当前全局模型性能:")
    print(f"  准确率: {global_acc:.2f}%, 损失: {global_loss:.4f}")
    
    # 打印每个类别的准确率
    print("  各类别准确率:")
    per_class_acc = eval_results['global']['per_class_accuracy']
    for i, acc in enumerate(per_class_acc):
        print(f"    类别 {i}: {acc:.2f}%")
    
    # 打印初始全局模型结果(如果有)
    if 'init_global' in eval_results:
        init_global_acc = eval_results['init_global']['accuracy']
        init_global_loss = eval_results['init_global']['loss']
        print(f"\n初始全局模型性能:")
        print(f"  准确率: {init_global_acc:.2f}%, 损失: {init_global_loss:.4f}")
        
        # 打印每个类别的准确率
        print("  各类别准确率:")
        init_per_class_acc = eval_results['init_global']['per_class_accuracy']
        for i, acc in enumerate(init_per_class_acc):
            print(f"    类别 {i}: {acc:.2f}%")
        
        # 打印比较结果
        print(f"\n全局模型性能比较 (当前 vs 初始):")
        print(f"  总体准确率变化: {global_acc - init_global_acc:.2f}%")
        print("  类别准确率变化:")
        for i, (curr, init) in enumerate(zip(per_class_acc, init_per_class_acc)):
            change = curr - init
            print(f"    类别 {i}: {change:+.2f}%  ({init:.2f}% -> {curr:.2f}%)")
    
    # 打印客户端平均结果
    avg_results = eval_results['averages']
    print(f"\n客户端平均性能:")
    print(f"  本地模型准确率: {avg_results['local_accuracy']:.2f}%")
    print(f"  客户端+服务器+全局分类器准确率: {avg_results['server_global_accuracy']:.2f}%")
    
    # 打印客户端+全局分类器结果（如果可用）
    client_global_acc = avg_results['client_global_accuracy']
    if isinstance(client_global_acc, (int, float)):
        print(f"  客户端+全局分类器准确率: {client_global_acc:.2f}%")
    else:
        print(f"  客户端+全局分类器准确率: {client_global_acc}")
    
    # 打印每个聚类的结果
    print(f"\n各聚类性能:")
    for key, value in eval_results.items():
        if key.startswith('cluster_'):
            cluster_id = key.split('_')[1]
            print(f"  聚类 {cluster_id}:")
            print(f"    本地模型准确率: {value['local_accuracy']:.2f}%")
            print(f"    客户端+服务器+全局分类器准确率: {value['server_global_accuracy']:.2f}%")
            
            if isinstance(value['client_global_accuracy'], (int, float)):
                print(f"    客户端+全局分类器准确率: {value['client_global_accuracy']:.2f}%")
            else:
                print(f"    客户端+全局分类器准确率: {value['client_global_accuracy']}")
    
    print("\n" + "="*50)
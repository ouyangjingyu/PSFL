import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd
import torch.nn as nn

def analyze_batchnorm_layers(model, prefix=""):
    """分析模型中所有批标准化层的统计信息"""
    stats = {}
    issues = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            full_name = f"{prefix}.{name}" if prefix else name
            if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
                mean = module.running_mean.cpu().numpy()
                var = module.running_var.cpu().numpy()
                
                # 计算统计数据
                mean_range = (float(mean.min()), float(mean.max()))
                var_range = (float(var.min()), float(var.max()))
                mean_abs_avg = float(np.abs(mean).mean())
                var_avg = float(var.mean())
                
                stats[full_name] = {
                    'mean_range': mean_range,
                    'var_range': var_range,
                    'mean_abs_avg': mean_abs_avg,
                    'var_avg': var_avg,
                    'weight_mean': float(module.weight.mean().item()) if hasattr(module, 'weight') else None,
                    'bias_mean': float(module.bias.mean().item()) if hasattr(module, 'bias') else None
                }
                
                # 检测异常
                if mean_abs_avg > 1.0:
                    issues.append(f"{full_name}: High absolute mean average: {mean_abs_avg:.4f}")
                if var_range[1] > 5.0:
                    issues.append(f"{full_name}: High variance: max={var_range[1]:.4f}")
                if mean_range[1] - mean_range[0] > 5.0:
                    issues.append(f"{full_name}: Wide mean range: {mean_range}")
    
    return stats, issues

def analyze_relu_activations(model, dataloader, device, n_batches=5):
    """分析ReLU激活层的活跃度"""
    print(f"\n分析ReLU激活层，使用设备: {device}")
    
    # 添加这两行确保模型在正确的设备上
    model = model.to(device)
    
    # 验证模型参数是否在正确设备上
    for param in model.parameters():
        if param.device != device:
            print(f"警告: 发现模型参数不在{device}上，尝试修复...")
            param.data = param.data.to(device)
    # 存储hooks
    activation = {}
    hooks = []
    
    # 定义hook函数
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 为所有ReLU层注册hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # 收集多个批次的激活数据
    relu_stats = defaultdict(list)
    model.eval()
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break
                
            images = images.to(device)
            _ = model(images)
            
            # 分析每个ReLU层的激活
            for name, act in activation.items():
                # 计算非零激活的百分比
                non_zero_percentage = (act > 0).float().mean().item() * 100
                relu_stats[name].append(non_zero_percentage)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 计算平均激活率
    average_stats = {name: np.mean(percentages) for name, percentages in relu_stats.items()}
    
    # 检测死亡神经元
    dead_neurons = []
    for name, avg_pct in average_stats.items():
        if avg_pct < 50:  # 如果激活率低于50%
            dead_neurons.append(f"{name}: Only {avg_pct:.2f}% neurons active")
    
    return average_stats, dead_neurons

def analyze_conv_filters(model, threshold=0.1):
    """分析卷积滤波器的多样性"""
    filter_stats = {}
    low_diversity_filters = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.detach().cpu()
            
            # 计算滤波器之间的相似度
            filters = weight.view(weight.size(0), -1)
            if filters.size(0) > 1:  # 至少要有两个滤波器来计算相似度
                # 计算每个滤波器与其他滤波器的余弦相似度
                norm_filters = filters / (filters.norm(dim=1, keepdim=True) + 1e-8)
                similarity_matrix = torch.mm(norm_filters, norm_filters.t())
                
                # 忽略自身相似度 (对角线)
                mask = torch.ones_like(similarity_matrix) - torch.eye(similarity_matrix.size(0))
                masked_similarity = similarity_matrix * mask
                
                # 计算平均相似度和最大相似度
                avg_similarity = masked_similarity.sum() / (mask.sum() + 1e-8)
                max_similarity = masked_similarity.max()
                
                filter_stats[name] = {
                    'avg_similarity': float(avg_similarity.item()),
                    'max_similarity': float(max_similarity.item()),
                    'num_filters': filters.size(0)
                }
                
                # 检测低多样性
                if avg_similarity > threshold:
                    low_diversity_filters.append(f"{name}: High filter similarity: {avg_similarity:.4f}")
    
    return filter_stats, low_diversity_filters

def analyze_classifier_weights(model, num_classes=10):
    """分析分类器层权重"""
    classifier_stats = {}
    issues = []
    
    # 查找最后的全连接层(分类器)
    classifier_layer = None
    classifier_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.out_features == num_classes:  # 假设输出维度等于类别数的是分类器
                classifier_layer = module
                classifier_name = name
    
    if classifier_layer is not None:
        weight = classifier_layer.weight.detach().cpu()
        
        # 计算每个类别的权重范数
        weight_norms = torch.norm(weight, dim=1)
        
        # 计算统计数据
        norm_mean = float(weight_norms.mean().item())
        norm_std = float(weight_norms.std().item())
        norm_min = float(weight_norms.min().item())
        norm_max = float(weight_norms.max().item())
        norm_variability = float(norm_std / (norm_mean + 1e-8))
        
        classifier_stats = {
            'norm_mean': norm_mean,
            'norm_std': norm_std, 
            'norm_range': (norm_min, norm_max),
            'norm_variability': norm_variability
        }
        
        # 检测问题
        if norm_variability < 0.1:
            issues.append(f"{classifier_name}: Low weight norm variability: {norm_variability:.4f}")
    
    return classifier_stats, issues

def track_model_predictions(model, dataloader, device, num_classes=10):
    """追踪模型在数据集上的预测分布"""
    model.eval()
    prediction_counts = torch.zeros(num_classes, dtype=torch.long)
    confidence_per_class = defaultdict(list)
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 如果是元组，取第一个元素
                
            # 获取预测和置信度
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            # 更新计数
            for i in range(num_classes):
                prediction_counts[i] += (predictions == i).sum().item()
                
            # 收集每个类别的置信度
            for pred, conf, true_label in zip(predictions, confidences, labels):
                pred_class = pred.item()
                confidence_per_class[pred_class].append(conf.item())
                true_labels.append(true_label.item())
                predicted_labels.append(pred_class)
    
    # 计算混淆矩阵数据
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # 计算每个类别的平均置信度
    avg_confidence = {}
    for cls, confs in confidence_per_class.items():
        avg_confidence[cls] = np.mean(confs) if confs else 0
    
    return {
        'prediction_counts': prediction_counts.tolist(),
        'avg_confidence': avg_confidence,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }

def visualize_prediction_distribution(prediction_data, title="Prediction Distribution"):
    """可视化预测分布"""
    counts = prediction_data['prediction_counts']
    avg_conf = prediction_data['avg_confidence']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制预测分布
    classes = list(range(len(counts)))
    ax1.bar(classes, counts)
    ax1.set_title("Prediction Class Distribution")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.set_xticks(classes)
    
    # 绘制平均置信度
    conf_classes = sorted(avg_conf.keys())
    confidences = [avg_conf[c] for c in conf_classes]
    ax2.bar(conf_classes, confidences)
    ax2.set_title("Average Confidence per Class")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Average Confidence")
    ax2.set_xticks(conf_classes)
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(true_labels, predicted_labels, num_classes=10, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes))
    plt.yticks(np.arange(num_classes) + 0.5, range(num_classes))
    
    return plt.gcf()

# 更新full_model_diagnostic函数，确保内部所有调用都正确处理设备
def full_model_diagnostic(model, dataloader, device, num_classes=10, n_batches=5):
    """对模型进行全面诊断"""
    print("=" * 50)
    print("开始全面模型诊断")
    print("=" * 50)
    
    # 确保模型在正确设备上
    model = model.to(device)
    
    # 1. 分析BatchNorm层
    print("\n分析BatchNorm层...")
    bn_stats, bn_issues = analyze_batchnorm_layers(model)
    
    if bn_issues:
        print(f"发现 {len(bn_issues)} 个BatchNorm层问题:")
        for issue in bn_issues[:5]:  # 只打印前5个问题
            print(f"  - {issue}")
    else:
        print("BatchNorm层没有明显问题")
    
    # 2. 分析ReLU激活
    print("\n分析ReLU激活层...")
    try:
        relu_stats, dead_neurons = analyze_relu_activations(model, dataloader, device, n_batches)
        
        if dead_neurons:
            print(f"发现 {len(dead_neurons)} 个ReLU死亡问题:")
            for issue in dead_neurons[:5]:  # 只打印前5个问题
                print(f"  - {issue}")
        else:
            print("ReLU激活层没有明显问题")
    except Exception as e:
        print(f"分析ReLU层时出错: {str(e)}")
        relu_stats, dead_neurons = {}, []
    
    # 3. 分析卷积滤波器
    print("\n分析卷积滤波器...")
    try:
        filter_stats, low_diversity_filters = analyze_conv_filters(model)
        
        if low_diversity_filters:
            print(f"发现 {len(low_diversity_filters)} 个卷积滤波器多样性低的问题:")
            for issue in low_diversity_filters[:5]:  # 只打印前5个问题
                print(f"  - {issue}")
        else:
            print("卷积滤波器没有明显问题")
    except Exception as e:
        print(f"分析卷积滤波器时出错: {str(e)}")
        filter_stats, low_diversity_filters = {}, []
    
    # 4. 分析分类器权重
    print("\n分析分类器权重...")
    try:
        classifier_stats, classifier_issues = analyze_classifier_weights(model, num_classes)
        
        if classifier_issues:
            print(f"发现 {len(classifier_issues)} 个分类器权重问题:")
            for issue in classifier_issues:
                print(f"  - {issue}")
        else:
            print("分类器权重没有明显问题")
    except Exception as e:
        print(f"分析分类器时出错: {str(e)}")
        classifier_stats, classifier_issues = {}, []
    
    # 5. 追踪模型预测
    print("\n分析模型预测分布...")
    try:
        prediction_data = track_model_predictions(model, dataloader, device, num_classes)
        
        counts = prediction_data['prediction_counts']
        total_predictions = sum(counts)
        class_percentages = [count/total_predictions*100 for count in counts]
        
        print("预测分布:")
        for cls, count in enumerate(counts):
            print(f"  类别 {cls}: {count} 预测 ({class_percentages[cls]:.2f}%)")
        
        # 检测类别不平衡
        max_percent = max(class_percentages)
        min_percent = min(class_percentages)
        expected_percent = 100 / num_classes
        
        if max_percent > expected_percent * 2 or min_percent < expected_percent * 0.5:
            print(f"警告: 检测到严重的类别不平衡。预期每类占比约 {expected_percent:.1f}%")
            print(f"  最低类别占比: {min_percent:.2f}%")
            print(f"  最高类别占比: {max_percent:.2f}%")
    except Exception as e:
        print(f"分析预测分布时出错: {str(e)}")
        prediction_data = {'prediction_counts': [0]*num_classes, 'avg_confidence': {}}
    
    # 返回所有诊断结果
    return {
        'bn_stats': bn_stats,
        'bn_issues': bn_issues,
        'relu_stats': relu_stats,
        'dead_neurons': dead_neurons,
        'filter_stats': filter_stats,
        'low_diversity_filters': low_diversity_filters,
        'classifier_stats': classifier_stats,
        'classifier_issues': classifier_issues,
        'prediction_data': prediction_data
    }

def fix_batchnorm_statistics(model, reset_running_stats=True, momentum=0.1):
    """修复BatchNorm层的统计量"""
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            if reset_running_stats:
                module.reset_running_stats()
            # 根据需要调整动量参数
            module.momentum = momentum
            # 确保训练模式下统计量会更新
            module.track_running_stats = True
            
    return model

def calibrate_model_with_data(model, dataloader, device, num_batches=10):
    """使用一些数据校准模型(特别是BatchNorm层)"""
    # 首先确保我们在训练模式下，以便BatchNorm更新统计量
    model.train()
    
    with torch.no_grad():  # 不需要计算梯度
        for i, (data, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            # 前向传播来更新BatchNorm统计量
            data = data.to(device)
            _ = model(data)
    
    return model

def compare_models_weights(model1, model2, threshold=0.01):
    """比较两个模型的权重差异"""
    diffs = []
    
    # 确保获取原始模型而非DataParallel包装
    if isinstance(model1, torch.nn.DataParallel):
        model1 = model1.module
    if isinstance(model2, torch.nn.DataParallel):
        model2 = model2.module
        
    # 获取状态字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # 找出共有的键
    common_keys = set(state_dict1.keys()).intersection(set(state_dict2.keys()))
    
    for key in common_keys:
        w1 = state_dict1[key].cpu().flatten()
        w2 = state_dict2[key].cpu().flatten()
        
        # 确保形状匹配
        if w1.shape != w2.shape:
            diffs.append((key, "Shape mismatch", w1.shape, w2.shape))
            continue
        
        # 计算相对差异
        abs_diff = (w1 - w2).abs().mean().item()
        rel_diff = abs_diff / (w1.abs().mean().item() + 1e-8)
        
        if rel_diff > threshold:
            diffs.append((key, rel_diff, w1.mean().item(), w2.mean().item()))
    
    return diffs

def trace_gradient_flow(model, named_parameters=None):
    """
    跟踪模型中的梯度流，打印层的梯度统计信息
    """
    if named_parameters is None:
        named_parameters = model.named_parameters()
        
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            grad_abs = p.grad.abs()
            ave_grads.append(grad_abs.mean().item())
            max_grads.append(grad_abs.max().item())
        else:
            ave_grads.append(0)
            max_grads.append(0)
        layers.append(n)
    
    return {'layers': layers, 'avg_grads': ave_grads, 'max_grads': max_grads}

def print_gradient_stats(grad_stats):
    """打印梯度统计信息"""
    print("\n梯度统计:")
    
    # 计算总体统计数据
    avg_all = np.mean(grad_stats['avg_grads'])
    max_all = np.max(grad_stats['max_grads'])
    min_all = np.min([g for g in grad_stats['avg_grads'] if g > 0])
    
    print(f"  平均梯度: {avg_all:.6f}")
    print(f"  最大梯度: {max_all:.6f}")
    print(f"  最小梯度: {min_all:.6f}")
    
    # 识别显著异常的层
    for layer, avg, max_g in zip(grad_stats['layers'], grad_stats['avg_grads'], grad_stats['max_grads']):
        if max_g > max_all * 0.8:  # 接近最大值的梯度
            print(f"  注意: 层 {layer} 有较大梯度: avg={avg:.6f}, max={max_g:.6f}")
        if avg > 0 and avg < min_all * 2:  # 接近最小值的梯度
            print(f"  注意: 层 {layer} 有较小梯度: avg={avg:.6f}, max={max_g:.6f}")

def normalize_bn_for_inference(model):
    """为推理规范化BatchNorm层"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # 确保BN层处于评估模式
            m.eval()
            # 可选:限制方差的范围以避免数值不稳定
            if hasattr(m, 'running_var'):
                with torch.no_grad():
                    m.running_var.clamp_(min=0.01, max=5.0)
    return model
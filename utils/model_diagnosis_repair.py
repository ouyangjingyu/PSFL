import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt

class ModelDiagnosticTracker:
    """跟踪模型训练期间的诊断指标"""
    
    def __init__(self):
        # 历史记录
        self.bn_issues_history = []
        self.dead_neurons_history = []
        self.pred_imbalance_history = []
        self.gradient_stats_history = []
        
        # 阈值设置
        self.bn_variance_min = 0.01
        self.bn_variance_max = 5.0
        self.bn_mean_min = -2.0
        self.bn_mean_max = 2.0
        self.dead_neuron_threshold = 0.05  # 5%激活率被视为死亡
        self.pred_imbalance_threshold = 5.0  # 最大/最小预测比例
        
    def add_diagnosis_result(self, diagnosis_result):
        """添加一轮的诊断结果"""
        # 记录BN问题
        self.bn_issues_history.append(len(diagnosis_result.get('bn_issues', [])))
        
        # 记录死亡神经元问题
        self.dead_neurons_history.append(len(diagnosis_result.get('dead_neurons', [])))
        
        # 记录预测不平衡情况
        pred_data = diagnosis_result.get('prediction_data', {})
        if isinstance(pred_data, dict) and 'prediction_counts' in pred_data:
            counts = pred_data['prediction_counts']
            if isinstance(counts, (list, np.ndarray)) and len(counts) > 0:
                if max(counts) > 0 and min(counts) > 0:
                    imbalance = max(counts) / min(counts)
                    self.pred_imbalance_history.append(imbalance)
                else:
                    self.pred_imbalance_history.append(0)
            else:
                self.pred_imbalance_history.append(0)
        else:
            self.pred_imbalance_history.append(0)
        # 记录梯度统计信息
        grad_stats = diagnosis_result.get('gradient_stats', {})
        if grad_stats:
            self.gradient_stats_history.append(grad_stats)
        
    def get_trend_analysis(self):
        """分析诊断指标的趋势"""
        trends = {}
        
        # 分析BN问题趋势
        if len(self.bn_issues_history) > 1:
            bn_trend = self.bn_issues_history[-1] - self.bn_issues_history[-2]
            trends['bn_issues'] = {
                'current': self.bn_issues_history[-1],
                'previous': self.bn_issues_history[-2],
                'change': bn_trend,
                'improving': bn_trend < 0
            }
        
        # 分析死亡神经元趋势
        if len(self.dead_neurons_history) > 1:
            neuron_trend = self.dead_neurons_history[-1] - self.dead_neurons_history[-2]
            trends['dead_neurons'] = {
                'current': self.dead_neurons_history[-1],
                'previous': self.dead_neurons_history[-2],
                'change': neuron_trend,
                'improving': neuron_trend < 0
            }
        
        # 分析预测不平衡趋势
        if len(self.pred_imbalance_history) > 1:
            imbalance_trend = self.pred_imbalance_history[-1] - self.pred_imbalance_history[-2]
            trends['pred_imbalance'] = {
                'current': self.pred_imbalance_history[-1],
                'previous': self.pred_imbalance_history[-2],
                'change': imbalance_trend,
                'improving': imbalance_trend < 0
            }
        
        return trends
    
    def recommend_actions(self):
        """基于诊断历史推荐模型修复操作"""
        actions = []
        
        # 检查模型诊断历史是否足够
        if len(self.bn_issues_history) == 0:
            return ["首次诊断，暂无修复建议"]
        
        # 推荐BN层修复
        if self.bn_issues_history[-1] > 0:
            actions.append({
                'type': 'bn_fix',
                'severity': 'high' if self.bn_issues_history[-1] > 5 else 'medium',
                'message': f"修复 {self.bn_issues_history[-1]} 个BatchNorm层问题"
            })
        
        # 推荐死亡神经元修复
        if self.dead_neurons_history[-1] > 0:
            actions.append({
                'type': 'neuron_fix',
                'severity': 'high' if self.dead_neurons_history[-1] > 5 else 'medium',
                'message': f"处理 {self.dead_neurons_history[-1]} 个ReLU死亡问题"
            })
        
        # 推荐预测不平衡修复
        if len(self.pred_imbalance_history) > 0 and self.pred_imbalance_history[-1] > self.pred_imbalance_threshold:
            actions.append({
                'type': 'classifier_fix',
                'severity': 'high',
                'message': f"重置分类器以解决预测不平衡问题 (比率: {self.pred_imbalance_history[-1]:.2f})"
            })
        
        # 如果没有发现问题
        if not actions:
            actions.append({
                'type': 'maintenance',
                'severity': 'low',
                'message': "模型状态良好，建议定期进行BatchNorm校准"
            })
        
        return actions


def detect_dead_neurons(model, dataloader, device, threshold=0.05, max_batches=10):
    """
    检测模型中的死亡神经元
    
    Args:
        model: 要诊断的模型
        dataloader: 数据加载器
        device: 计算设备
        threshold: 激活率阈值
        max_batches: 最大批次数
        
    Returns:
        dead_neurons: 死亡神经元信息列表
    """
    # 存储每个ReLU层的激活信息
    activation = {}
    hooks = []
    
    # 定义hook函数
    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook
    
    # 为所有ReLU层注册hook
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # 收集激活统计信息
    activation_stats = defaultdict(list)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            data = data.to(device)
            model(data)
            
            # 分析每个ReLU层的激活
            for name, act in activation.items():
                # 计算非零元素的比例
                active_ratio = (act > 0).float().mean().item()
                activation_stats[name].append(active_ratio)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 检测死亡神经元
    dead_neurons = []
    for name, ratios in activation_stats.items():
        avg_ratio = np.mean(ratios)
        if avg_ratio < threshold:
            dead_neurons.append({
                'layer': name,
                'activation_ratio': avg_ratio,
                'severity': 'high' if avg_ratio < threshold/2 else 'medium'
            })
    
    return dead_neurons


def analyze_prediction_distribution(model, dataloader, device, num_classes=10, max_batches=10):
    """
    分析模型的预测分布
    
    Args:
        model: 要诊断的模型
        dataloader: 数据加载器
        device: 计算设备
        num_classes: 类别数量
        max_batches: 最大批次数
        
    Returns:
        prediction_stats: 预测分布统计信息
    """
    model.eval()
    prediction_counts = torch.zeros(num_classes)
    confidence_per_class = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # 获取预测和置信度
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            # 更新计数
            for i in range(num_classes):
                prediction_counts[i] += (predictions == i).sum().item()
                
            # 收集每个类别的置信度
            for pred, conf in zip(predictions, confidences):
                confidence_per_class[pred.item()].append(conf.item())
    
    # 计算每个类别的平均置信度
    avg_confidence = {}
    for cls, confs in confidence_per_class.items():
        avg_confidence[cls] = np.mean(confs) if confs else 0
    
    # 计算类别分布
    total_preds = prediction_counts.sum().item()
    if total_preds > 0:
        class_distribution = prediction_counts / total_preds
    else:
        class_distribution = torch.ones(num_classes) / num_classes
    
    # 检测预测不平衡
    if torch.max(class_distribution) > 0 and torch.min(class_distribution) > 0:
        imbalance_ratio = torch.max(class_distribution) / torch.min(class_distribution)
    else:
        imbalance_ratio = torch.tensor(1.0)
    
    most_predicted = torch.argmax(class_distribution).item()
    least_predicted = torch.argmin(class_distribution).item()
    
    return {
        'prediction_counts': prediction_counts.tolist(),
        'class_distribution': class_distribution.tolist(),
        'avg_confidence': avg_confidence,
        'imbalance_ratio': imbalance_ratio.item(),
        'most_predicted': most_predicted,
        'least_predicted': least_predicted
    }


def fix_batchnorm_statistics(model, reset_running_stats=True):
    """
    修复模型中的BatchNorm层统计量
    
    Args:
        model: 要修复的模型
        reset_running_stats: 是否重置运行统计量
        
    Returns:
        fixed_count: 修复的BatchNorm层数量
    """
    fixed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            # 重置运行统计量
            if reset_running_stats:
                module.reset_running_stats()
                fixed_count += 1
            
            # 修复方差
            if hasattr(module, 'running_var'):
                with torch.no_grad():
                    problematic_var = (module.running_var < 0.01) | (module.running_var > 5.0)
                    if problematic_var.any():
                        # 将异常值替换为1.0
                        module.running_var[problematic_var] = 1.0
                        fixed_count += problematic_var.sum().item()
            
            # 修复均值
            if hasattr(module, 'running_mean'):
                with torch.no_grad():
                    problematic_mean = (module.running_mean < -2.0) | (module.running_mean > 2.0)
                    if problematic_mean.any():
                        # 将异常值替换为0.0
                        module.running_mean[problematic_mean] = 0.0
                        fixed_count += problematic_mean.sum().item()
    
    return fixed_count


def fix_dead_relu_neurons(model, dead_neurons_info):
    """
    修复模型中的死亡ReLU神经元
    
    Args:
        model: 要修复的模型
        dead_neurons_info: 死亡神经元信息列表
        
    Returns:
        fixed_count: 修复的层数量
    """
    fixed_count = 0
    
    for neuron_info in dead_neurons_info:
        layer_name = neuron_info['layer']
        
        # 查找对应的层
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, nn.ReLU):
                # 将ReLU替换为Leaky ReLU
                parent_name = '.'.join(name.split('.')[:-1])
                parent_module = model
                
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                # 获取ReLU在父模块中的属性名
                for attr_name, attr_value in parent_module.__dict__.items():
                    if attr_value is module:
                        # 替换为LeakyReLU
                        setattr(parent_module, attr_name, nn.LeakyReLU(0.1, inplace=True))
                        fixed_count += 1
                        break
    
    return fixed_count


def reset_classifier_weights(model, num_classes=10):
    """
    重置分类器权重
    
    Args:
        model: 要修复的模型
        num_classes: 类别数量
        
    Returns:
        reset_count: 重置的分类器层数量
    """
    reset_count = 0
    classifier_found = False
    
    # 搜索分类器层
    for name, module in model.named_modules():
        # 检查是否是分类器层（最后一层线性层或明确的分类器模块）
        is_classifier = False
        
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            is_classifier = True
        elif 'classifier' in name.lower() and isinstance(module, nn.Linear):
            is_classifier = True
        
        if is_classifier:
            classifier_found = True
            # 重新初始化权重
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            reset_count += 1
    
    # 如果没有找到明确的分类器，尝试查找最后一个线性层
    if not classifier_found:
        last_linear = None
        last_linear_name = None
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                last_linear_name = name
        
        if last_linear is not None:
            # 重新初始化权重
            nn.init.kaiming_normal_(last_linear.weight, mode='fan_out', nonlinearity='relu')
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)
            reset_count += 1
    
    return reset_count


def calibrate_model_with_data(model, dataloader, device, num_batches=10):
    """
    使用数据校准模型（主要是BatchNorm层）
    
    Args:
        model: 要校准的模型
        dataloader: 数据加载器
        device: 计算设备
        num_batches: 校准批次数
        
    Returns:
        model: 校准后的模型
    """
    # 设置模型为训练模式
    model.train()
    
    # 保存每个BatchNorm层的原始状态
    bn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            bn_modules[name] = module
            module.momentum = 0.1  # 使用较小的动量参数加快统计量更新
            module.reset_running_stats()
    
    # 使用小批量数据更新BN统计量
    with torch.no_grad():  # 不需要计算梯度
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # 前向传播更新BN统计量
            data = data.to(device)
            model(data)
    
    # 将模型设回评估模式
    model.eval()
    
    # 限制BN统计量范围
    with torch.no_grad():
        for module in bn_modules.values():
            if hasattr(module, 'running_var'):
                module.running_var.clamp_(min=0.01, max=5.0)
            if hasattr(module, 'running_mean'):
                module.running_mean.clamp_(min=-2.0, max=2.0)
    
    return model


def comprehensive_model_repair(model, diagnosis_tracker, dataloader, device, num_classes=10):
    """
    综合模型修复函数，根据诊断结果自动执行所需的修复操作
    
    Args:
        model: 要修复的模型
        diagnosis_tracker: 诊断追踪器
        dataloader: 数据加载器
        device: 计算设备
        num_classes: 类别数量
        
    Returns:
        model: 修复后的模型
        repair_summary: 修复操作摘要
    """
    # 获取修复建议
    recommended_actions = diagnosis_tracker.recommend_actions()
    repair_summary = {
        'actions_taken': [],
        'bn_layers_fixed': 0,
        'dead_neurons_fixed': 0,
        'classifier_reset': False
    }
    
    # 根据建议执行修复
    for action in recommended_actions:
        if action['type'] == 'bn_fix':
            # 执行BatchNorm修复
            fixed_count = fix_batchnorm_statistics(model)
            repair_summary['bn_layers_fixed'] = fixed_count
            repair_summary['actions_taken'].append(f"修复了 {fixed_count} 个BatchNorm层")
        
        elif action['type'] == 'neuron_fix':
            # 检测死亡神经元
            dead_neurons = detect_dead_neurons(model, dataloader, device)
            if dead_neurons:
                # 修复死亡神经元
                fixed_count = fix_dead_relu_neurons(model, dead_neurons)
                repair_summary['dead_neurons_fixed'] = fixed_count
                repair_summary['actions_taken'].append(f"修复了 {fixed_count} 个死亡ReLU神经元")
        
        elif action['type'] == 'classifier_fix' and action['severity'] == 'high':
            # 重置分类器权重
            reset_count = reset_classifier_weights(model, num_classes)
            if reset_count > 0:
                repair_summary['classifier_reset'] = True
                repair_summary['actions_taken'].append(f"重置了 {reset_count} 个分类器层")
        
        elif action['type'] == 'maintenance':
            # 执行模型校准
            model = calibrate_model_with_data(model, dataloader, device)
            repair_summary['actions_taken'].append("执行了模型批标准化层校准")
    
    # 如果没有执行任何修复操作，至少进行校准
    if not repair_summary['actions_taken']:
        model = calibrate_model_with_data(model, dataloader, device)
        repair_summary['actions_taken'].append("执行了预防性模型校准")
    
    return model, repair_summary



def analyze_global_vs_local_prediction_consistency(global_model, client_models, eval_dataset, 
                                                 server_models=None, global_classifier=None, 
                                                 device='cuda', num_classes=10, client_ids=None):
    """
    Analyzes prediction consistency between global model and client models
    
    Args:
        global_model: Global model
        client_models: Dictionary of client models
        eval_dataset: Evaluation dataset
        server_models: Dictionary of server models (for split learning)
        global_classifier: Global classifier (for split learning)
        device: Computation device
        num_classes: Number of classes
        client_ids: List of client IDs to analyze (if None, use all)
        
    Returns:
        Dictionary with analysis results
    """
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # Initialize metrics
    global_predictions = []
    client_predictions = {}
    true_labels = []
    
    # Set models to evaluation mode
    global_model = global_model.to(device)
    global_model.eval()
    
    # Initialize client models if needed
    if client_ids is None:
        client_ids = list(client_models.keys())
    
    for client_id in client_ids:
        client_models[client_id] = client_models[client_id].to(device)
        client_models[client_id].eval()
        client_predictions[client_id] = []
        
        if server_models and client_id in server_models:
            server_models[client_id] = server_models[client_id].to(device)
            server_models[client_id].eval()
    
    if global_classifier is not None:
        global_classifier = global_classifier.to(device)
        global_classifier.eval()
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Get global model predictions
            outputs = global_model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, global_preds = torch.max(outputs, 1)
            global_predictions.append(global_preds.cpu().numpy())
            
            # Store true labels
            true_labels.append(target.cpu().numpy())
            
            # Get client model predictions
            for client_id in client_ids:
                if server_models and client_id in server_models:
                    # Split learning prediction
                    client_out = client_models[client_id](data)
                    if isinstance(client_out, tuple):
                        _, client_features = client_out
                    else:
                        client_features = client_out
                    
                    server_features = server_models[client_id](client_features)
                    
                    if global_classifier is not None:
                        client_outputs = global_classifier(server_features)
                    else:
                        client_outputs = server_features
                else:
                    # Local model prediction
                    client_outputs = client_models[client_id](data)
                    if isinstance(client_outputs, tuple):
                        client_outputs = client_outputs[0]
                
                _, client_preds = torch.max(client_outputs, 1)
                client_predictions[client_id].append(client_preds.cpu().numpy())
    
    # Concatenate predictions
    global_predictions = np.concatenate(global_predictions)
    true_labels = np.concatenate(true_labels)
    for client_id in client_ids:
        client_predictions[client_id] = np.concatenate(client_predictions[client_id])
    
    # Calculate agreement rates
    agreement_rates = {}
    for client_id in client_ids:
        agreement = np.mean(global_predictions == client_predictions[client_id]) * 100
        agreement_rates[client_id] = agreement
    
    # Calculate per-class agreement
    per_class_agreement = {}
    for class_idx in range(num_classes):
        class_mask = (true_labels == class_idx)
        if np.sum(class_mask) == 0:
            continue
            
        class_agreement = {}
        for client_id in client_ids:
            client_agree = np.mean(
                global_predictions[class_mask] == client_predictions[client_id][class_mask]
            ) * 100
            class_agreement[client_id] = client_agree
        
        per_class_agreement[class_idx] = class_agreement
    
    # Calculate accuracy for global and client models
    global_acc = np.mean(global_predictions == true_labels) * 100
    client_acc = {}
    for client_id in client_ids:
        client_acc[client_id] = np.mean(client_predictions[client_id] == true_labels) * 100
    
    # Calculate per-class accuracy
    global_class_acc = {}
    client_class_acc = defaultdict(dict)
    for class_idx in range(num_classes):
        class_mask = (true_labels == class_idx)
        if np.sum(class_mask) == 0:
            continue
            
        global_class_acc[class_idx] = np.mean(global_predictions[class_mask] == class_idx) * 100
        for client_id in client_ids:
            client_class_acc[client_id][class_idx] = np.mean(
                client_predictions[client_id][class_mask] == class_idx
            ) * 100
    
    # Create confusion matrices for global and client models
    global_confmat = np.zeros((num_classes, num_classes))
    client_confmat = defaultdict(lambda: np.zeros((num_classes, num_classes)))
    
    for true_class in range(num_classes):
        class_mask = (true_labels == true_class)
        if np.sum(class_mask) == 0:
            continue
            
        for pred_class in range(num_classes):
            global_confmat[true_class, pred_class] = np.mean(
                global_predictions[class_mask] == pred_class
            ) * 100
            
            for client_id in client_ids:
                client_confmat[client_id][true_class, pred_class] = np.mean(
                    client_predictions[client_id][class_mask] == pred_class
                ) * 100
    
    return {
        'agreement_rates': agreement_rates,
        'per_class_agreement': per_class_agreement,
        'global_accuracy': global_acc,
        'client_accuracy': client_acc,
        'global_class_accuracy': global_class_acc,
        'client_class_accuracy': client_class_acc,
        'global_confusion': global_confmat,
        'client_confusion': dict(client_confmat)
    }

def analyze_classifier_weights(model, device='cuda', num_classes=10):
    """
    Analyzes classifier weights to detect potential class imbalance issues
    
    Args:
        model: Model to analyze
        device: Computation device
        num_classes: Number of classes
        
    Returns:
        Dictionary with weight analysis results
    """
    model = model.to(device)
    weight_stats = {}
    
    # Find the classifier layer(s)
    classifier_found = False
    classifier_weights = []
    
    # Check if model has a dedicated classifier
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        classifier_found = True
        
        # For multi-layer classifiers, focus on the final layer
        if isinstance(classifier, torch.nn.Sequential):
            for module in reversed(classifier):
                if isinstance(module, torch.nn.Linear) and module.out_features == num_classes:
                    classifier_weights.append(module.weight.data.clone())
                    break
        elif hasattr(classifier, 'weight'):
            classifier_weights.append(classifier.weight.data.clone())
        
        # For complex classifiers, check component layers
        if hasattr(classifier, 'fc3'):
            classifier_weights.append(classifier.fc3.weight.data.clone())
    
    # If no classifier found, look for fc layers
    if not classifier_found:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == num_classes:
                classifier_weights.append(module.weight.data.clone())
                classifier_found = True
    
    # If we found classifier weights, analyze them
    if classifier_weights:
        weight = classifier_weights[0]  # Use the first one found
        
        # Calculate weight statistics
        class_weight_norms = torch.norm(weight, dim=1).cpu().numpy()
        
        # Find classes with largest/smallest weight norms
        max_norm_class = np.argmax(class_weight_norms)
        min_norm_class = np.argmin(class_weight_norms)
        
        # Calculate norm ratios and statistics
        max_norm = np.max(class_weight_norms)
        min_norm = np.min(class_weight_norms)
        mean_norm = np.mean(class_weight_norms)
        std_norm = np.std(class_weight_norms)
        
        # Calculate imbalance metrics
        if min_norm > 0:
            max_min_ratio = max_norm / min_norm
        else:
            max_min_ratio = float('inf')
        
        coefficient_of_variation = std_norm / mean_norm if mean_norm > 0 else float('inf')
        
        # Store results
        weight_stats = {
            'class_weight_norms': class_weight_norms.tolist(),
            'max_norm_class': int(max_norm_class),
            'min_norm_class': int(min_norm_class),
            'max_norm': float(max_norm),
            'min_norm': float(min_norm),
            'mean_norm': float(mean_norm),
            'std_norm': float(std_norm),
            'max_min_ratio': float(max_min_ratio),
            'coefficient_of_variation': float(coefficient_of_variation),
            'imbalance_detected': max_min_ratio > 2.0  # Threshold for imbalance
        }
    
    return weight_stats

def analyze_feature_distributions(model, dataloader, device='cuda', num_samples=500):
    """
    Analyzes feature distributions from the model's penultimate layer
    
    Args:
        model: Model to analyze
        dataloader: Data loader for input data
        device: Computation device
        num_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary with feature distribution analysis
    """
    model = model.to(device)
    model.eval()
    
    # Store features and corresponding labels/predictions
    features_list = []
    labels_list = []
    predictions_list = []
    sample_count = 0
    
    # Define hook to extract features
    activation = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook
    
    # Identify penultimate layer (typically before the final classifier)
    penultimate_layer = None
    
    # Check if model has a sequential classifier
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        modules = list(model.classifier.children())
        if len(modules) > 1:
            # Use input to the last layer
            penultimate_layer = modules[-2]  # Second to last layer
            penultimate_layer.register_forward_hook(get_activation('penultimate'))
        
    # Check if model has a classifier with multiple layers
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'fc2'):
        penultimate_layer = model.classifier.fc2
        penultimate_layer.register_forward_hook(get_activation('penultimate'))
    
    # If no suitable layer found in classifier, try avgpool or similar
    if penultimate_layer is None and hasattr(model, 'avgpool'):
        penultimate_layer = model.avgpool
        penultimate_layer.register_forward_hook(get_activation('penultimate'))
    
    # If still no layer found, use a more generic approach
    if penultimate_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'classifier' in name:
                penultimate_layer = module
                penultimate_layer.register_forward_hook(get_activation('penultimate'))
                break
    
    # If still no layer found, it's difficult to extract features
    if penultimate_layer is None:
        return {"error": "Could not identify penultimate layer for feature extraction"}
    
    with torch.no_grad():
        for data, target in dataloader:
            if sample_count >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predictions
            _, predictions = torch.max(outputs.data, 1)
            
            # Get penultimate features (batch_size x feature_dim)
            if 'penultimate' in activation:
                features = activation['penultimate']
                
                # If features are from a convolutional layer, flatten them
                if len(features.shape) > 2:
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)
                
                # Append to lists
                features_list.append(features.cpu().numpy())
                labels_list.append(target.cpu().numpy())
                predictions_list.append(predictions.cpu().numpy())
                
                sample_count += features.size(0)
    
    # If no features collected, return error
    if not features_list:
        return {"error": "No features collected"}
    
    # Combine features and labels from all batches
    features_array = np.vstack(features_list)
    labels_array = np.concatenate(labels_list)
    predictions_array = np.concatenate(predictions_list)
    
    # Calculate per-class feature statistics
    unique_classes = np.unique(labels_array)
    class_feature_stats = {}
    
    for class_idx in unique_classes:
        class_mask = (labels_array == class_idx)
        if np.sum(class_mask) == 0:
            continue
            
        class_features = features_array[class_mask]
        
        # Calculate feature statistics
        feature_mean = np.mean(class_features, axis=0)
        feature_std = np.std(class_features, axis=0)
        feature_norm = np.linalg.norm(feature_mean)
        
        # Calculate prediction accuracy for this class
        class_pred_mask = (predictions_array[class_mask] == class_idx)
        accuracy = np.mean(class_pred_mask) * 100
        
        class_feature_stats[int(class_idx)] = {
            'feature_norm': float(feature_norm),
            'feature_std_avg': float(np.mean(feature_std)),
            'num_samples': int(np.sum(class_mask)),
            'accuracy': float(accuracy)
        }
    
    # Calculate feature separation metrics
    feature_separation = {}
    if len(unique_classes) > 1:
        class_means = np.array([class_feature_stats[int(c)]['feature_norm'] for c in unique_classes])
        class_accuracy = np.array([class_feature_stats[int(c)]['accuracy'] for c in unique_classes])
        
        feature_separation = {
            'max_mean': float(np.max(class_means)),
            'min_mean': float(np.min(class_means)),
            'mean_ratio': float(np.max(class_means) / np.min(class_means) if np.min(class_means) > 0 else np.inf),
            'max_acc': float(np.max(class_accuracy)),
            'min_acc': float(np.min(class_accuracy)),
            'acc_ratio': float(np.max(class_accuracy) / np.min(class_accuracy) if np.min(class_accuracy) > 0 else np.inf)
        }
    
    return {
        'class_feature_stats': class_feature_stats,
        'feature_separation': feature_separation,
        'feature_dim': features_array.shape[1]
    }

def enhanced_analyze_prediction_distribution(model, dataloader, device='cuda', num_classes=10, max_batches=10):
    """
    Enhanced version of analyze_prediction_distribution with more detailed analysis
    
    Args:
        model: Model to analyze
        dataloader: Data loader for input data
        device: Computation device
        num_classes: Number of classes
        max_batches: Maximum number of batches to analyze
        
    Returns:
        Dictionary with enhanced prediction distribution analysis
    """
    model.eval()
    prediction_counts = torch.zeros(num_classes)
    confidence_per_class = defaultdict(list)
    confidence_correct = defaultdict(list)
    confidence_wrong = defaultdict(list)
    
    # Track confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    # Track correct and total predictions per class
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            # Update counts
            for i in range(num_classes):
                prediction_counts[i] += (predictions == i).sum().item()
                
            # Update confusion matrix and per-class statistics
            for t, p, conf in zip(targets, predictions, confidences):
                t_idx, p_idx = t.item(), p.item()
                if t_idx < num_classes and p_idx < num_classes:
                    confusion_matrix[t_idx, p_idx] += 1
                    total_per_class[t_idx] += 1
                    
                    if t_idx == p_idx:
                        correct_per_class[t_idx] += 1
                        confidence_correct[t_idx].append(conf.item())
                    else:
                        confidence_wrong[t_idx].append(conf.item())
                
                # Add confidence to appropriate class
                confidence_per_class[p_idx].append(conf.item())
    
    # Calculate class distribution and imbalance metrics
    total_preds = prediction_counts.sum().item()
    if total_preds > 0:
        class_distribution = prediction_counts / total_preds
    else:
        class_distribution = torch.ones(num_classes) / num_classes
    
    # Calculate prediction imbalance metrics
    if torch.max(class_distribution) > 0 and torch.min(class_distribution) > 0:
        imbalance_ratio = torch.max(class_distribution) / torch.min(class_distribution)
        
        # Gini coefficient for inequality
        sorted_probs = torch.sort(class_distribution)[0]
        n = sorted_probs.size(0)
        index = torch.arange(1, n + 1).float()
        gini = torch.sum((2 * index - n - 1) * sorted_probs) / (n * torch.sum(sorted_probs))
    else:
        imbalance_ratio = torch.tensor(1.0)
        gini = torch.tensor(0.0)
    
    # Calculate confusion matrix statistics
    normalized_confusion = confusion_matrix.clone()
    for i in range(num_classes):
        if total_per_class[i] > 0:
            normalized_confusion[i] = confusion_matrix[i] / total_per_class[i]
    
    # Calculate per-class accuracy
    per_class_accuracy = torch.zeros(num_classes)
    for i in range(num_classes):
        if total_per_class[i] > 0:
            per_class_accuracy[i] = correct_per_class[i] / total_per_class[i] * 100
    
    # Calculate confidence statistics
    avg_confidence = {}
    avg_correct_conf = {}
    avg_wrong_conf = {}
    for cls, confs in confidence_per_class.items():
        avg_confidence[cls] = np.mean(confs) if confs else 0
    
    for cls, confs in confidence_correct.items():
        avg_correct_conf[cls] = np.mean(confs) if confs else 0
        
    for cls, confs in confidence_wrong.items():
        avg_wrong_conf[cls] = np.mean(confs) if confs else 0
    
    # Find most/least accurate and most/least predicted classes
    most_predicted = torch.argmax(class_distribution).item()
    least_predicted = torch.argmin(class_distribution).item()
    
    most_accurate_idx = torch.argmax(per_class_accuracy).item()
    least_accurate_idx = torch.argmin(per_class_accuracy).item()
    
    return {
        'prediction_counts': prediction_counts.tolist(),
        'class_distribution': class_distribution.tolist(),
        'avg_confidence': avg_confidence,
        'avg_correct_confidence': avg_correct_conf,
        'avg_wrong_confidence': avg_wrong_conf,
        'per_class_accuracy': per_class_accuracy.tolist(),
        'imbalance_ratio': imbalance_ratio.item(),
        'gini_coefficient': gini.item(),
        'normalized_confusion': normalized_confusion.tolist(),
        'most_predicted': most_predicted,
        'least_predicted': least_predicted,
        'most_accurate': most_accurate_idx,
        'least_accurate': least_accurate_idx,
        'prediction_imbalance_detected': imbalance_ratio.item() > 3.0  # Threshold for significant imbalance
    }

def diagnose_global_model_performance(global_model, client_models, test_dataloader,
                                     balanced_dataloader, server_models=None, 
                                     global_classifier=None, device='cuda', num_classes=10):
    """
    Comprehensive diagnosis of global model performance issues
    
    Args:
        global_model: Global model
        client_models: Dictionary of client models
        test_dataloader: Test dataloader (client distribution)
        balanced_dataloader: Balanced test dataloader
        server_models: Dictionary of server models
        global_classifier: Global classifier
        device: Computation device
        num_classes: Number of classes
        
    Returns:
        Dictionary with comprehensive diagnosis results
    """
    diagnosis = {}
    
    # 1. Analyze prediction distribution on test data (client distribution)
    test_pred_dist = enhanced_analyze_prediction_distribution(
        global_model, test_dataloader, device, num_classes
    )
    diagnosis['test_prediction_distribution'] = test_pred_dist
    
    # 2. Analyze prediction distribution on balanced data
    bal_pred_dist = enhanced_analyze_prediction_distribution(
        global_model, balanced_dataloader, device, num_classes
    )
    diagnosis['balanced_prediction_distribution'] = bal_pred_dist
    
    # 3. Analyze classifier weights
    weight_stats = analyze_classifier_weights(global_model, device, num_classes)
    diagnosis['classifier_weights'] = weight_stats
    
    # 4. Analyze feature distributions on balanced data
    feature_stats = analyze_feature_distributions(global_model, balanced_dataloader, device)
    diagnosis['feature_distributions'] = feature_stats
    
    # 5. Analyze performance difference between test and balanced datasets
    if 'per_class_accuracy' in test_pred_dist and 'per_class_accuracy' in bal_pred_dist:
        test_acc = np.array(test_pred_dist['per_class_accuracy'])
        bal_acc = np.array(bal_pred_dist['per_class_accuracy'])
        
        # Calculate accuracy differences
        acc_diff = bal_acc - test_acc
        
        # Find classes with biggest differences
        max_diff_idx = np.argmax(acc_diff)
        min_diff_idx = np.argmin(acc_diff)
        
        dataset_comparison = {
            'accuracy_difference': acc_diff.tolist(),
            'max_difference_class': int(max_diff_idx),
            'max_difference_value': float(acc_diff[max_diff_idx]),
            'min_difference_class': int(min_diff_idx),
            'min_difference_value': float(acc_diff[min_diff_idx]),
            'avg_difference': float(np.mean(acc_diff)),
            'std_difference': float(np.std(acc_diff))
        }
        diagnosis['dataset_comparison'] = dataset_comparison
    
    # 6. Compare prediction distributions between datasets
    if 'class_distribution' in test_pred_dist and 'class_distribution' in bal_pred_dist:
        test_dist = np.array(test_pred_dist['class_distribution'])
        bal_dist = np.array(bal_pred_dist['class_distribution'])
        
        # Calculate distribution differences
        dist_diff = bal_dist - test_dist
        
        # Calculate KL divergence (how different are the distributions)
        epsilon = 1e-10  # Small value to avoid division by zero
        kl_div = np.sum(bal_dist * np.log((bal_dist + epsilon) / (test_dist + epsilon)))
        
        distribution_comparison = {
            'distribution_difference': dist_diff.tolist(),
            'kl_divergence': float(kl_div),
            'most_overrepresented': int(np.argmax(dist_diff)),
            'most_underrepresented': int(np.argmin(dist_diff))
        }
        diagnosis['distribution_comparison'] = distribution_comparison
    
    # 7. Analyze summary metrics
    summary = {
        'test_vs_balanced_accuracy_gap': float(np.mean(np.array(bal_pred_dist['per_class_accuracy'])) - 
                                              np.mean(np.array(test_pred_dist['per_class_accuracy']))),
        'prediction_imbalance_ratio': float(test_pred_dist['imbalance_ratio']),
        'weight_imbalance_ratio': float(weight_stats['max_min_ratio']),
        'feature_imbalance_detected': test_pred_dist['imbalance_ratio'] > 3.0 or weight_stats['max_min_ratio'] > 2.0,
        'likely_causes': []
    }
    
    # Determine likely causes of performance issues
    if summary['test_vs_balanced_accuracy_gap'] > 5.0:
        summary['likely_causes'].append("Significant accuracy gap between balanced and test datasets")
    
    if test_pred_dist['imbalance_ratio'] > 3.0:
        summary['likely_causes'].append("Prediction distribution imbalance detected")
    
    if weight_stats['max_min_ratio'] > 2.0:
        summary['likely_causes'].append("Classifier weight imbalance detected")
    
    # Add feature distribution issues if applicable
    if feature_stats.get('feature_separation', {}).get('mean_ratio', 1.0) > 2.0:
        summary['likely_causes'].append("Feature representation imbalance across classes")
    
    diagnosis['summary'] = summary
    
    return diagnosis

def generate_diagnostic_plots(diagnosis_results, save_path=None):
    """
    Generate diagnostic plots based on diagnosis results
    
    Args:
        diagnosis_results: Results from diagnose_global_model_performance
        save_path: Path to save plots (if None, plots are displayed)
        
    Returns:
        Dictionary with plot file paths (if save_path is provided)
    """
    if not diagnosis_results:
        return {"error": "No diagnosis results provided"}
    
    plot_files = {}
    
    # Create a figure for prediction distribution
    if 'test_prediction_distribution' in diagnosis_results:
        pred_dist = diagnosis_results['test_prediction_distribution']
        
        plt.figure(figsize=(12, 8))
        
        # Plot prediction distribution
        plt.subplot(2, 2, 1)
        plt.bar(range(len(pred_dist['prediction_counts'])), pred_dist['prediction_counts'])
        plt.title('Prediction Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Plot per-class accuracy
        plt.subplot(2, 2, 2)
        plt.bar(range(len(pred_dist['per_class_accuracy'])), pred_dist['per_class_accuracy'])
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        
        # Plot confusion matrix as heatmap
        plt.subplot(2, 2, 3)
        plt.imshow(pred_dist['normalized_confusion'], cmap='Blues')
        plt.colorbar()
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Plot confidence values
        plt.subplot(2, 2, 4)
        avg_conf = [pred_dist['avg_confidence'].get(i, 0) for i in range(len(pred_dist['class_distribution']))]
        plt.bar(range(len(avg_conf)), avg_conf)
        plt.title('Average Prediction Confidence')
        plt.xlabel('Class')
        plt.ylabel('Confidence')
        
        plt.tight_layout()
        
        if save_path:
            pred_dist_file = f"{save_path}/prediction_distribution.png"
            plt.savefig(pred_dist_file)
            plot_files['prediction_distribution'] = pred_dist_file
        else:
            plt.show()
        
        plt.close()
    
    # Create a figure for comparison between datasets
    if 'dataset_comparison' in diagnosis_results:
        dataset_comp = diagnosis_results['dataset_comparison']
        
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy difference
        plt.subplot(1, 2, 1)
        plt.bar(range(len(dataset_comp['accuracy_difference'])), dataset_comp['accuracy_difference'])
        plt.title('Accuracy Difference (Balanced - Test)')
        plt.xlabel('Class')
        plt.ylabel('Accuracy Difference (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        
        # Plot distribution difference
        plt.subplot(1, 2, 2)
        if 'distribution_comparison' in diagnosis_results:
            dist_comp = diagnosis_results['distribution_comparison']
            plt.bar(range(len(dist_comp['distribution_difference'])), dist_comp['distribution_difference'])
            plt.title('Distribution Difference (Balanced - Test)')
            plt.xlabel('Class')
            plt.ylabel('Distribution Difference')
            plt.axhline(y=0, color='r', linestyle='-')
        
        plt.tight_layout()
        
        if save_path:
            comp_file = f"{save_path}/dataset_comparison.png"
            plt.savefig(comp_file)
            plot_files['dataset_comparison'] = comp_file
        else:
            plt.show()
        
        plt.close()
    
    # Create a figure for classifier weights
    if 'classifier_weights' in diagnosis_results:
        weight_stats = diagnosis_results['classifier_weights']
        
        if 'class_weight_norms' in weight_stats:
            plt.figure(figsize=(10, 6))
            
            plt.bar(range(len(weight_stats['class_weight_norms'])), weight_stats['class_weight_norms'])
            plt.title(f'Classifier Weight Norms (Max/Min Ratio: {weight_stats["max_min_ratio"]:.2f})')
            plt.xlabel('Class')
            plt.ylabel('Weight Norm')
            
            plt.tight_layout()
            
            if save_path:
                weight_file = f"{save_path}/classifier_weights.png"
                plt.savefig(weight_file)
                plot_files['classifier_weights'] = weight_file
            else:
                plt.show()
            
            plt.close()
    
    # Create a figure for feature distribution stats
    if 'feature_distributions' in diagnosis_results and 'class_feature_stats' in diagnosis_results['feature_distributions']:
        feature_stats = diagnosis_results['feature_distributions']['class_feature_stats']
        
        plt.figure(figsize=(12, 6))
        
        # Plot feature norms
        plt.subplot(1, 2, 1)
        norms = [feature_stats[cls]['feature_norm'] for cls in sorted(feature_stats.keys())]
        plt.bar(range(len(norms)), norms)
        plt.title('Feature Representation Norms')
        plt.xlabel('Class')
        plt.ylabel('Feature Norm')
        
        # Plot feature standard deviations
        plt.subplot(1, 2, 2)
        stds = [feature_stats[cls]['feature_std_avg'] for cls in sorted(feature_stats.keys())]
        plt.bar(range(len(stds)), stds)
        plt.title('Feature Standard Deviations')
        plt.xlabel('Class')
        plt.ylabel('Average Std Dev')
        
        plt.tight_layout()
        
        if save_path:
            feature_file = f"{save_path}/feature_distributions.png"
            plt.savefig(feature_file)
            plot_files['feature_distributions'] = feature_file
        else:
            plt.show()
        
        plt.close()
    
    return plot_files
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


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

def verify_global_model(model, eval_dataset, device, model_name="全局模型"):
    """详细验证模型状态和特征分布"""
    print(f"\n===== {model_name}诊断 =====")
    model.eval()
    
    # 1. 分析分类器权重
    if hasattr(model, 'classifier'):
        print("分析分类器权重...")
        # 尝试获取最后一层权重
        last_layer = None
        if hasattr(model.classifier, 'fc3'):
            last_layer = model.classifier.fc3
        elif hasattr(model.classifier, 'fc'):
            last_layer = model.classifier.fc
        
        if last_layer is not None:
            weights = last_layer.weight.data
            weight_norms = torch.norm(weights, dim=1)
            max_norm = torch.max(weight_norms).item()
            min_norm = torch.min(weight_norms).item()
            imbalance = max_norm / (min_norm + 1e-6)
            
            print(f"分类器权重范数: {[f'{w:.4f}' for w in weight_norms.tolist()]}")
            print(f"权重不平衡度(最大/最小): {imbalance:.4f}")
            
            # 警告极度不平衡
            if imbalance > 10:
                print(f"警告：分类器权重严重不平衡({imbalance:.2f}倍)，可能导致分类偏向")
    
    # 2. 建立激活统计收集钩子
    activation_stats = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # 计算基本统计量
                activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }
        return hook
    
    # 为关键层注册钩子
    key_layer_types = (nn.Conv2d, nn.GroupNorm, nn.LayerNorm, nn.Linear)
    for name, module in model.named_modules():
        if isinstance(module, key_layer_types):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # 3. 运行测试数据获取激活值
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)
    class_predictions = torch.zeros(10)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(eval_loader):
            if i >= 2:  # 只用少量批次
                break
                
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # 处理可能的元组输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 统计预测分布
            _, predictions = torch.max(outputs, 1)
            for j in range(10):  # 假设10个类别
                class_predictions[j] += (predictions == j).sum().item()
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 4. 分析层激活情况
    problematic_layers = []
    for name, stats in activation_stats.items():
        if stats['has_nan'] or stats['has_inf'] or stats['std'] < 1e-4:
            problematic_layers.append(name)
            print(f"警告: {name} 层异常:")
            if stats['has_nan']:
                print("  - 包含NaN值")
            if stats['has_inf']:
                print("  - 包含Inf值")
            if stats['std'] < 1e-4:
                print(f"  - 方差极小 ({stats['std']:.6f}), 可能特征崩塌")
    
    # 5. 分析预测分布
    total_preds = class_predictions.sum().item()
    if total_preds > 0:
        class_distribution = class_predictions / total_preds
        print("\n预测类别分布:")
        for i, prob in enumerate(class_distribution):
            print(f"  类别 {i}: {prob*100:.2f}%")
        
        # 计算预测熵(越低越集中)
        distribution = class_distribution.numpy()
        entropy = -np.sum(distribution * np.log(distribution + 1e-10))
        print(f"预测熵(越低越集中): {entropy:.4f}")
        
        # 检测单一类别预测
        max_prob = torch.max(class_distribution).item()
        if max_prob > 0.9:
            print(f"警告: 预测极度集中于类别 {torch.argmax(class_distribution).item()} ({max_prob*100:.1f}%)")
    
    print(f"问题层数量: {len(problematic_layers)}/{len(activation_stats)}")
    print("="*50)
    
    # 返回诊断结果摘要
    return {
        'problematic_layers': problematic_layers,
        'prediction_distribution': class_distribution.tolist() if 'class_distribution' in locals() else None,
        'weight_imbalance': imbalance if 'imbalance' in locals() else None
    }

def diagnose_feature_matching(client_model, server_model, global_classifier, eval_dataset, device):
    """诊断客户端-服务器-分类器的特征匹配情况"""
    client_model.eval()
    server_model.eval()
    global_classifier.eval()
    
    # 创建测试数据加载器
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)
    
    # 收集特征统计信息
    client_feature_stats = []
    server_feature_stats = []
    
    with torch.no_grad():
        for data, _ in eval_loader:
            data = data.to(device)
            
            # 获取客户端特征
            _, client_features = client_model(data)
            
            # 获取服务器特征
            server_features = server_model(client_features)
            
            # 如果是卷积特征，展平
            if len(server_features.shape) > 2:
                server_features = F.adaptive_avg_pool2d(server_features, (1, 1))
                server_features = server_features.view(server_features.size(0), -1)
            
            # 计算特征统计量
            client_feat_flat = client_features.view(client_features.size(0), -1)
            client_norms = torch.norm(client_feat_flat, dim=1)
            server_norms = torch.norm(server_features, dim=1)
            
            client_feature_stats.append({
                'mean_norm': client_norms.mean().item(),
                'std_norm': client_norms.std().item(),
                'mean': client_feat_flat.mean().item(),
                'std': client_feat_flat.std().item()
            })
            
            server_feature_stats.append({
                'mean_norm': server_norms.mean().item(),
                'std_norm': server_norms.std().item(),
                'mean': server_features.mean().item(),
                'std': server_features.std().item()
            })
            
            # 只用一个批次
            break
    
    print("\n===== 客户端-服务器特征匹配诊断 =====")
    print(f"客户端特征范数: {client_feature_stats[0]['mean_norm']:.4f} ± {client_feature_stats[0]['std_norm']:.4f}")
    print(f"服务器特征范数: {server_feature_stats[0]['mean_norm']:.4f} ± {server_feature_stats[0]['std_norm']:.4f}")
    print(f"客户端特征统计: 均值={client_feature_stats[0]['mean']:.4f}, 标准差={client_feature_stats[0]['std']:.4f}")
    print(f"服务器特征统计: 均值={server_feature_stats[0]['mean']:.4f}, 标准差={server_feature_stats[0]['std']:.4f}")
    
    # 输出特征匹配评估
    norm_ratio = server_feature_stats[0]['mean_norm'] / client_feature_stats[0]['mean_norm']
    print(f"特征范数比率(服务器/客户端): {norm_ratio:.4f}")
    
    if norm_ratio < 0.1 or norm_ratio > 10:
        print("警告: 客户端和服务器特征范数差异显著，可能导致特征不匹配")
    
    std_ratio = server_feature_stats[0]['std'] / client_feature_stats[0]['std']
    if std_ratio < 0.1 or std_ratio > 10:
        print("警告: 客户端和服务器特征方差差异显著，可能导致特征不匹配")
    
    return {
        'client_features': client_feature_stats[0],
        'server_features': server_feature_stats[0],
        'norm_ratio': norm_ratio
    }
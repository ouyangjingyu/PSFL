import torch
import copy
import numpy as np
import logging
from collections import defaultdict

class LayeredAggregator:
    """分层聚合器，针对不同组件采用不同的聚合策略"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.history = {
            'shared_layers': None,
            'server_model': None,
            'global_classifier': None
        }
        
    def compute_feature_stability(self, current_params, previous_params):
        """计算特征稳定性（参数变化的倒数）"""
        if previous_params is None:
            return 1.0
        
        total_change = 0.0
        total_params = 0
        
        for key in current_params:
            if key in previous_params:
                change = torch.norm(current_params[key] - previous_params[key]).item()
                total_change += change
                total_params += current_params[key].numel()
        
        if total_params == 0:
            return 1.0
        
        avg_change = total_change / total_params
        stability = 1.0 / (1.0 + avg_change)
        return stability
    
    def aggregate_shared_layers(self, client_states, eval_results):
        """共享层：特征一致性聚合"""
        if not client_states:
            return {}
        
        # 过滤共享层参数
        shared_states = {}
        for client_id, state in client_states.items():
            shared_state = {}
            for k, v in state.items():
                if 'shared_base' in k:
                    shared_state[k] = v
            if shared_state:
                shared_states[client_id] = shared_state
        
        if not shared_states:
            return {}
        
        # 计算权重
        weights = {}
        total_weight = 0.0
        
        for client_id, state in shared_states.items():
            # 全局准确率权重
            global_acc = eval_results.get(client_id, {}).get('global_accuracy', 50.0)
            acc_weight = (global_acc / 100.0) ** 1.2  # β=1.2
            
            # 特征稳定性权重
            stability = self.compute_feature_stability(
                state, 
                self.history['shared_layers'].get(client_id) if self.history['shared_layers'] else None
            )
            stability_weight = stability ** 0.8  # γ=0.8
            
            # 组合权重
            weight = acc_weight * stability_weight
            weights[client_id] = weight
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            for client_id in weights:
                weights[client_id] /= total_weight
        else:
            # 平均权重作为备选
            num_clients = len(shared_states)
            for client_id in shared_states:
                weights[client_id] = 1.0 / num_clients
        
        # 聚合
        aggregated_state = self._weighted_average(shared_states, weights)
        
        # 更新历史
        self.history['shared_layers'] = copy.deepcopy(shared_states)
        
        logging.info(f"共享层聚合权重: {weights}")
        return aggregated_state
    
    def aggregate_server_models(self, server_states, eval_results):
        """服务器模型：性能驱动聚合"""
        if not server_states:
            return {}
        
        # 计算基于性能的权重
        weights = {}
        accuracies = []
        
        for cluster_id, state in server_states.items():
            # 从客户端结果计算聚类平均准确率
            cluster_accs = []
            for client_id, result in eval_results.items():
                if isinstance(result, dict) and result.get('cluster_id') == cluster_id:
                    cluster_accs.append(result.get('global_accuracy', 50.0))
            
            # 计算聚类平均准确率
            if cluster_accs:
                acc = sum(cluster_accs) / len(cluster_accs)
            else:
                acc = 50.0  # 默认准确率
            
            accuracies.append(acc)
        
        # 使用softmax计算权重（温度参数=5）
        temperature = 5.0
        logits = np.array(accuracies) / temperature
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        weights_array = exp_logits / np.sum(exp_logits)
        
        # 转换为字典
        for i, cluster_id in enumerate(server_states.keys()):
            weights[cluster_id] = weights_array[i]
        
        # 聚合
        aggregated_state = self._weighted_average(server_states, weights)
        
        # 更新历史
        self.history['server_model'] = copy.deepcopy(server_states)
        
        logging.info(f"服务器模型聚合权重: {weights}")
        return aggregated_state
    
    def aggregate_global_classifiers(self, classifier_states, eval_results):
        """全局分类器：保守聚合"""
        if not classifier_states:
            return {}
        
        # 先计算基础聚合结果
        weights = {}
        total_weight = 0.0
        
        for cluster_id, state in classifier_states.items():
            # 从客户端结果计算聚类平均准确率
            cluster_accs = []
            for client_id, result in eval_results.items():
                if isinstance(result, dict) and result.get('cluster_id') == cluster_id:
                    cluster_accs.append(result.get('global_accuracy', 50.0))
            
            # 计算聚类平均准确率
            if cluster_accs:
                acc = sum(cluster_accs) / len(cluster_accs)
            else:
                acc = 50.0  # 默认准确率
            
            weight = acc / 100.0
            weights[cluster_id] = weight
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            for cluster_id in weights:
                weights[cluster_id] /= total_weight
        else:
            num_clusters = len(classifier_states)
            for cluster_id in classifier_states:
                weights[cluster_id] = 1.0 / num_clusters
        
        # 基础聚合
        base_aggregated = self._weighted_average(classifier_states, weights)
        
        # 保守更新：与历史状态混合
        if self.history['global_classifier'] is not None:
            final_aggregated = {}
            for key in base_aggregated:
                if key in self.history['global_classifier']:
                    # 70%历史 + 30%新聚合
                    final_aggregated[key] = (0.7 * self.history['global_classifier'][key] + 
                                           0.3 * base_aggregated[key])
                else:
                    final_aggregated[key] = base_aggregated[key]
        else:
            final_aggregated = base_aggregated
        
        # 更新历史
        self.history['global_classifier'] = copy.deepcopy(final_aggregated)
        
        logging.info(f"全局分类器聚合权重: {weights}")
        return final_aggregated
    
    def _weighted_average(self, state_dict, weights):
        """计算加权平均"""
        if not state_dict:
            return {}
        
        # 获取第一个状态字典的键
        keys = next(iter(state_dict.values())).keys()
        
        # 初始化结果
        result = {}
        
        # 对每个参数进行加权平均
        for key in keys:
            weighted_sum = None
            total_weight = 0.0
            
            # 加权累加
            for state_id, state in state_dict.items():
                if key in state:
                    weight = weights.get(state_id, 0.0)
                    total_weight += weight
                    
                    # 累加
                    param_tensor = state[key].to(self.device)
                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor
                    else:
                        weighted_sum += weight * param_tensor
            
            # 计算平均值
            if weighted_sum is not None and total_weight > 0:
                result[key] = weighted_sum / total_weight
        
        return result
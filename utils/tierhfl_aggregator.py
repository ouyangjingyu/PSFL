import torch
import copy
import numpy as np
from collections import defaultdict
import math

# 稳定化聚合器 - 处理数据异质性的聚合问题
class StabilizedAggregator:
    def __init__(self, beta=0.5, device='cuda'):
        self.beta = beta  # 动量因子
        self.device = device
        self.previous_client_model = None  # 初始化客户端模型历史
        self.previous_server_model = None  # 初始化服务器模型历史
        self.cluster_weights = None  # 聚类权重
        self.accuracy_history = []  # 准确率历史记录
    
    # 聚合客户端模型
    def aggregate_clients(self, client_states, client_weights=None, client_clusters=None):
        """聚合客户端状态"""
        # 没有客户端状态时直接返回空字典
        if not client_states:
            return {}
            
        # 如果已提供聚类信息，使用聚类聚合
        if client_clusters:
            aggregated_model = self._clustered_aggregation(client_states, client_weights, client_clusters)
        else:
            # 否则使用直接加权平均聚合
            aggregated_model = self._weighted_average_aggregation(client_states, client_weights)
        
        # 应用动量
        if self.previous_client_model is not None:
            stabilized_model = {}
            for k in aggregated_model:
                if k in self.previous_client_model:
                    stabilized_model[k] = self.beta * self.previous_client_model[k] + (1 - self.beta) * aggregated_model[k]
                else:
                    stabilized_model[k] = aggregated_model[k]
            
            aggregated_model = stabilized_model
        
        # 保存当前模型作为下一轮的历史
        self.previous_client_model = copy.deepcopy(aggregated_model)
        
        return aggregated_model
    
    # 聚合服务器模型
    def aggregate_server(self, server_states, cluster_weights=None, eval_results=None, cluster_map=None):
        """基于性能感知的服务器状态聚合"""
        if not server_states:
            return {}
        
        # 如果提供了评估结果和聚类映射，计算基于性能的权重
        if eval_results and cluster_map:
            performance_weights = {}
            total_weight = 0.0
            
            for cluster_id, model_state in server_states.items():
                # 计算该聚类中客户端的平均全局准确率
                cluster_clients = cluster_map.get(cluster_id, [])
                valid_clients = [c for c in cluster_clients if c in eval_results]
                
                if valid_clients:
                    avg_acc = sum(eval_results[c].get('global_accuracy', 0) 
                            for c in valid_clients) / len(valid_clients)
                    
                    # 使用sigmoid函数映射准确率到权重
                    weight = 1.0 / (1.0 + math.exp(-0.1 * (avg_acc - 50)))
                    performance_weights[cluster_id] = weight
                    total_weight += weight
                else:
                    performance_weights[cluster_id] = 0.5
                    total_weight += 0.5
            
            # 归一化权重
            if total_weight > 0:
                for cluster_id in performance_weights:
                    performance_weights[cluster_id] /= total_weight
                cluster_weights = performance_weights
                
        # 默认使用平均权重
        if cluster_weights is None:
            cluster_weights = {i: 1.0/len(server_states) for i in server_states}
            
        # 使用加权平均聚合
        aggregated_model = self._weighted_average(server_states, cluster_weights)
        
        # 应用动量
        if self.previous_server_model is not None:
            stabilized_model = {}
            for k in aggregated_model:
                if k in self.previous_server_model:
                    stabilized_model[k] = self.beta * self.previous_server_model[k] + (1 - self.beta) * aggregated_model[k]
                else:
                    stabilized_model[k] = aggregated_model[k]
            
            aggregated_model = stabilized_model
        
        self.previous_server_model = copy.deepcopy(aggregated_model)
        
        return aggregated_model
        
     # 添加自适应动量方法
    def adjust_beta(self, round_idx, global_acc, prev_global_acc):
        """基于训练进度和性能动态调整动量因子"""
        # 前期使用较小beta促进充分探索
        if round_idx < 15:
            self.beta = max(0.3, 0.8 - round_idx * 0.03)
        # 性能停滞时降低beta增加新信息
        elif global_acc - prev_global_acc < 0.2:
            self.beta = max(0.2, self.beta * 0.9)
        # 性能增长明显时可适度增加beta提高稳定性
        elif global_acc - prev_global_acc > 1.0:
            self.beta = min(0.7, self.beta * 1.05)
    def set_cluster_weights(self, weights):
        """设置聚类权重"""
        self.cluster_weights = weights
        
    def aggregate(self, client_states, client_weights=None, client_clusters=None):
        """聚合客户端状态"""
        # 没有客户端状态时直接返回空字典
        if not client_states:
            return {}
            
        # 如果已提供聚类信息，使用聚类聚合
        if client_clusters:
            return self._clustered_aggregation(client_states, client_weights, client_clusters)
        
        # 否则使用直接加权平均聚合
        return self._weighted_average_aggregation(client_states, client_weights)
    
    def _clustered_aggregation(self, client_states, client_weights, client_clusters):
        """聚类导向的聚合策略"""
        # 第一步：聚类内聚合
        cluster_models = {}
        
        for cluster_id, client_ids in client_clusters.items():
            # 收集该聚类中的客户端状态
            cluster_states = {}
            cluster_client_weights = {}
            
            for client_id in client_ids:
                if client_id in client_states:
                    # 获取共享参数（排除分类器参数）
                    shared_state = self._filter_personalized_params(client_states[client_id])
                    cluster_states[client_id] = shared_state
                    
                    # 获取该客户端的权重
                    if client_weights and client_id in client_weights:
                        cluster_client_weights[client_id] = client_weights[client_id]
                    else:
                        cluster_client_weights[client_id] = 1.0
            
            # 如果该聚类有客户端状态，进行聚类内聚合
            if cluster_states:
                # 使用归一化的权重
                total_weight = sum(cluster_client_weights.values())
                if total_weight > 0:
                    for client_id in cluster_client_weights:
                        cluster_client_weights[client_id] /= total_weight
                        
                # 聚类内聚合
                cluster_models[cluster_id] = self._weighted_average(cluster_states, cluster_client_weights)
        
        # 第二步：跨聚类聚合
        if not cluster_models:
            return {}
            
        # 如果只有一个聚类，直接返回
        if len(cluster_models) == 1:
            return next(iter(cluster_models.values()))
            
        # 否则进行跨聚类聚合
        # 计算聚类权重（默认平均权重）
        if self.cluster_weights is None:
            cluster_weights = {cluster_id: 1.0 / len(cluster_models) for cluster_id in cluster_models}
        else:
            # 使用提供的聚类权重
            cluster_weights = {}
            total_weight = 0.0
            for cluster_id in cluster_models:
                if cluster_id in self.cluster_weights:
                    cluster_weights[cluster_id] = self.cluster_weights[cluster_id]
                    total_weight += self.cluster_weights[cluster_id]
                else:
                    cluster_weights[cluster_id] = 1.0
                    total_weight += 1.0
            
            # 归一化
            for cluster_id in cluster_weights:
                cluster_weights[cluster_id] /= total_weight
                
        # 将聚类模型转换为字典格式
        cluster_states = {cluster_id: model for cluster_id, model in cluster_models.items()}
        
        # 聚合
        return self._weighted_average(cluster_states, cluster_weights)
    
    def _weighted_average_aggregation(self, client_states, client_weights):
        """简单的加权平均聚合"""
        # 过滤掉个性化参数
        filtered_states = {}
        for client_id, state in client_states.items():
            filtered_states[client_id] = self._filter_personalized_params(state)
        
        # 如果没有提供权重，使用均等权重
        if client_weights is None:
            client_weights = {client_id: 1.0 / len(filtered_states) for client_id in filtered_states}
        
        # 聚合
        return self._weighted_average(filtered_states, client_weights)
        
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
            # 初始化累加器
            weighted_sum = None
            total_weight = 0.0
            
            # 加权累加
            for client_id, state in state_dict.items():
                if key in state:
                    weight = weights.get(client_id, 0.0)
                    total_weight += weight
                    
                    # 累加
                    if weighted_sum is None:
                        weighted_sum = weight * state[key].clone().to(self.device)
                    else:
                        weighted_sum += weight * state[key].clone().to(self.device)
            
            # 计算平均值
            if weighted_sum is not None and total_weight > 0:
                result[key] = weighted_sum / total_weight
        
        return result
        
    def _filter_personalized_params(self, state_dict):
        """过滤掉个性化参数"""
        filtered_state = {}
        for k, v in state_dict.items():
            # 排除分类器参数
            if not any(name in k for name in ['classifier', 'local_head', 'fc', 'linear']):
                filtered_state[k] = v
        return filtered_state
        
    def update_accuracy_history(self, accuracy):
        """更新准确率历史并调整beta"""
        self.accuracy_history.append(accuracy)
        
        # 动态调整beta值
        if len(self.accuracy_history) >= 2:
            acc_diff = self.accuracy_history[-1] - self.accuracy_history[-2]
            
            # 准确率下降或停滞时，降低beta
            if acc_diff < 0.1:
                self.beta = max(0.1, self.beta * 0.9)
            # 准确率提升时，提高beta
            elif acc_diff > 1.0:
                self.beta = min(0.6, self.beta * 1.05)

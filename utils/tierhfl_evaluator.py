import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from collections import defaultdict

class TierHFLEvaluator:
    """TierHFL评估工具"""
    def __init__(self, client_manager, central_server, test_loader, device='cuda'):
        self.client_manager = client_manager
        self.central_server = central_server
        self.test_loader = test_loader  # 均衡测试集
        self.device = device
        
        # 评估历史
        self.evaluation_history = []
    
    def evaluate_client_local(self, client_id, client_model):
        """评估客户端本地性能
        
        Args:
            client_id: 客户端ID
            client_model: 客户端模型
            
        Returns:
            评估结果
        """
        client = self.client_manager.get_client(client_id)
        if client is None:
            return None
        
        # 获取客户端所在的服务器组
        server_group = self.central_server.get_client_group(client_id)
        if server_group is None:
            return None
        
        # 执行评估
        eval_result = client.evaluate(
            server_group.server_model, server_group.global_classifier)
        
        return eval_result
    
    def evaluate_global_generalization(self, client_models):
        """评估全局模型在均衡测试集上的泛化性能
        
        Args:
            client_models: 客户端模型字典
            
        Returns:
            评估结果
        """
        # 按Tier分组统计
        tier_results = defaultdict(list)
        tier_models = defaultdict(list)
        
        for client_id, model in client_models.items():
            client = self.client_manager.get_client(client_id)
            if client:
                tier = client.tier
                tier_models[tier].append((client_id, model))
        
        # 对每个Tier选择代表模型评估
        tier_accuracies = {}
        for tier, models in tier_models.items():
            if models:
                # 选择第一个模型作为代表
                client_id, model = models[0]
                
                # 获取服务器组
                server_group = self.central_server.get_client_group(client_id)
                if server_group:
                    # 执行评估
                    accuracy = self._evaluate_on_test_set(
                        model, server_group.server_model, server_group.global_classifier)
                    tier_accuracies[tier] = accuracy
        
        # 计算全局平均准确率
        global_accuracy = np.mean(list(tier_accuracies.values())) if tier_accuracies else 0
        
        return {
            'global_accuracy': global_accuracy,
            'tier_accuracy': tier_accuracies
        }
    
    def evaluate_cross_client(self, client_models):
        """评估模型在跨客户端数据上的性能
        
        创建客户端间性能矩阵，分析知识迁移能力
        
        Args:
            client_models: 客户端模型字典
            
        Returns:
            评估结果
        """
        client_ids = list(client_models.keys())
        n_clients = len(client_ids)
        performance_matrix = np.zeros((n_clients, n_clients))
        
        for i, client_id1 in enumerate(client_ids):
            client1 = self.client_manager.get_client(client_id1)
            if client1 is None:
                continue
                
            # 获取客户端所在的服务器组
            server_group1 = self.central_server.get_client_group(client_id1)
            if server_group1 is None:
                continue
                
            model1 = client_models[client_id1].to(self.device)
            
            for j, client_id2 in enumerate(client_ids):
                if i == j:
                    # 自身性能
                    eval_result = client1.evaluate(
                        server_group1.server_model, server_group1.global_classifier)
                    performance_matrix[i, j] = eval_result['global_accuracy']
                else:
                    # 跨客户端性能
                    client2 = self.client_manager.get_client(client_id2)
                    if client2 is None:
                        continue
                        
                    # 使用客户端1的模型和服务器处理客户端2的数据
                    accuracy = self._evaluate_on_client_data(
                        model1, server_group1.server_model, 
                        server_group1.global_classifier, client2.test_data)
                    performance_matrix[i, j] = accuracy
        
        # 计算每个客户端的平均跨客户端性能
        client_avg_perf = []
        for i, client_id in enumerate(client_ids):
            # 排除自身性能
            cross_perf = []
            for j in range(n_clients):
                if i != j:
                    cross_perf.append(performance_matrix[i, j])
            
            avg_perf = np.mean(cross_perf) if cross_perf else 0
            client_avg_perf.append((client_id, avg_perf))
        
        # 计算总体平均跨客户端性能
        overall_avg = np.mean([p for _, p in client_avg_perf]) if client_avg_perf else 0
        
        return {
            'performance_matrix': performance_matrix,
            'client_avg_performance': client_avg_perf,
            'overall_avg_performance': overall_avg
        }
    
    def _evaluate_on_test_set(self, client_model, server_model, global_classifier):
        """在测试集上评估模型性能
        
        Args:
            client_model: 客户端模型
            server_model: 服务器模型
            global_classifier: 全局分类器
            
        Returns:
            准确率
        """
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        global_classifier.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 客户端前向传播
                _, global_features, _ = client_model(data)
                
                # 服务器处理
                server_features = server_model(global_features)
                global_logits = global_classifier(server_features)
                
                # 计算准确率
                _, predicted = global_logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy
    
    def _evaluate_on_client_data(self, client_model, server_model, 
                               global_classifier, test_data):
        """在特定客户端数据上评估模型性能
        
        Args:
            client_model: 客户端模型
            server_model: 服务器模型
            global_classifier: 全局分类器
            test_data: 测试数据
            
        Returns:
            准确率
        """
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        global_classifier.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                # 客户端前向传播
                _, global_features, _ = client_model(data)
                
                # 服务器处理
                server_features = server_model(global_features)
                global_logits = global_classifier(server_features)
                
                # 计算准确率
                _, predicted = global_logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy
    
    def conduct_comprehensive_evaluation(self, client_models, round_idx):
        """进行全面评估
        
        Args:
            client_models: 客户端模型字典
            round_idx: 当前训练轮次
            
        Returns:
            评估结果
        """
        start_time = time.time()
        
        # 本地评估
        local_results = {}
        for client_id, model in client_models.items():
            result = self.evaluate_client_local(client_id, model)
            if result:
                local_results[client_id] = result
        
        # 全局泛化性能评估
        global_result = self.evaluate_global_generalization(client_models)
        
        # 跨客户端评估 - 仅在特定轮次执行(计算开销大)
        cross_client_result = None
        if round_idx % 5 == 0 or round_idx == 0:
            cross_client_result = self.evaluate_cross_client(client_models)
        
        evaluation_time = time.time() - start_time
        
        # 整合结果
        result = {
            'round': round_idx,
            'local_results': local_results,
            'global_result': global_result,
            'cross_client_result': cross_client_result,
            'evaluation_time': evaluation_time,
            'timestamp': time.time()
        }
        
        # 保存评估历史
        self.evaluation_history.append(result)
        
        # 更新中央服务器的评估历史
        self.central_server.update_evaluation_history(round_idx, {
            'global_accuracy': global_result['global_accuracy'],
            'group_accuracy': global_result.get('tier_accuracy', {}),
            'cross_client_accuracy': cross_client_result['overall_avg_performance'] 
                                    if cross_client_result else None
        })
        
        return result
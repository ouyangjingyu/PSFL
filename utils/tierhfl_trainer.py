import torch
import threading
import queue
import time
import copy
import numpy as np
from collections import defaultdict
import logging

class TierHFLTrainer:
    """TierHFL并行训练框架"""
    def __init__(self, client_manager, central_server, grouping_strategy, 
                 loss_fn, gradient_guide, max_workers=None):
        self.client_manager = client_manager
        self.central_server = central_server
        self.grouping_strategy = grouping_strategy
        self.loss_fn = loss_fn
        self.gradient_guide = gradient_guide
        
        # 设置最大并行工作线程数
        if max_workers is None:
            if torch.cuda.is_available():
                max_workers = torch.cuda.device_count()
            else:
                import multiprocessing
                max_workers = max(1, multiprocessing.cpu_count() // 2)
        
        self.max_workers = max(1, max_workers)
        
        # 初始化日志
        self.logger = logging.getLogger("TierHFLTrainer")
        self.logger.setLevel(logging.INFO)
        
        # 训练统计
        self.stats = defaultdict(list)
    
    def _train_client(self, client_id, client_model, server_model, 
                     global_classifier, round_idx, results_queue):
        """训练单个客户端的工作函数"""
        try:
            # 获取客户端
            client = self.client_manager.get_client(client_id)
            if client is None:
                self.logger.warning(f"客户端 {client_id} 不存在，跳过")
                return
            
            # 记录开始时间
            start_time = time.time()
            
            # 训练客户端
            train_result, model_state = client.train(
                server_model, global_classifier, self.loss_fn, 
                self.gradient_guide, round_idx)
            
            # 评估客户端
            eval_result = client.evaluate(server_model, global_classifier)
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 结果加入队列
            results_queue.put({
                'client_id': client_id,
                'train_result': train_result,
                'eval_result': eval_result,
                'model_state': model_state,
                'training_time': training_time
            })
            
            self.logger.info(f"客户端 {client_id} 训练完成，"
                           f"耗时: {training_time:.2f}秒，"
                           f"准确率: {eval_result['global_accuracy']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 训练失败: {str(e)}")
            
            # 报告错误
            results_queue.put({
                'client_id': client_id,
                'error': str(e)
            })
    
    def execute_parallel_training(self, client_models, round_idx):
        """执行并行训练
        
        Args:
            client_models: 客户端模型字典
            round_idx: 当前训练轮次
            
        Returns:
            训练结果
        """
        start_time = time.time()
        
        # 客户端分组
        client_groups = self.grouping_strategy.group_clients(
            self.client_manager, client_models, round_idx)
        
        # 更新中央服务器中的客户端分组
        for group_id, client_ids in client_groups.items():
            for client_id in client_ids:
                self.central_server.assign_client_to_group(client_id, group_id)
        
        # 打印分组信息
        self.logger.info(f"轮次 {round_idx} 客户端分组:")
        for group_id, client_ids in client_groups.items():
            self.logger.info(f"组 {group_id}: {client_ids}")
        
        # 创建结果队列
        results_queue = queue.Queue()
        
        # 创建线程
        threads = []
        for client_id, model in client_models.items():
            # 获取客户端所在的服务器组
            group = self.central_server.get_client_group(client_id)
            if group is None:
                self.logger.warning(f"客户端 {client_id} 未分配服务器组，跳过")
                continue
            
            thread = threading.Thread(
                target=self._train_client,
                args=(client_id, model, group.server_model, 
                     group.global_classifier, round_idx, results_queue)
            )
            threads.append(thread)
        
        # 控制并行度
        active_threads = []
        
        for thread in threads:
            # 等待有可用线程槽
            while len(active_threads) >= self.max_workers:
                # 检查已完成的线程
                active_threads = [t for t in active_threads if t.is_alive()]
                if len(active_threads) >= self.max_workers:
                    time.sleep(0.1)
            
            # 启动新线程
            thread.start()
            active_threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        train_results = {}
        eval_results = {}
        client_states = {}
        time_stats = {}
        
        while not results_queue.empty():
            result = results_queue.get()
            
            # 检查是否有错误
            if 'error' in result:
                self.logger.error(f"客户端 {result['client_id']} 返回错误: {result['error']}")
                continue
            
            # 保存结果
            client_id = result['client_id']
            train_results[client_id] = result['train_result']
            eval_results[client_id] = result['eval_result']
            client_states[client_id] = result['model_state']
            time_stats[client_id] = {
                'training_time': result['training_time']
            }
        
        # 聚合结果
        grouped_states = defaultdict(list)
        grouped_weights = defaultdict(list)
        
        for client_id, state in client_states.items():
            group_id = self.grouping_strategy.get_client_group(client_id)
            if group_id is not None:
                grouped_states[group_id].append(state)
                
                # 计算权重 - 基于全局准确率
                accuracy = eval_results[client_id]['global_accuracy']
                grouped_weights[group_id].append(max(0.1, accuracy / 100.0))
        
        # 更新每个组的服务器模型
        server_states = {}
        classifier_states = {}
        
        for group_id, states in grouped_states.items():
            if states:
                weights = grouped_weights[group_id]
                # 归一化权重
                weights = [w / sum(weights) for w in weights]
                
                # 聚合该组的模型
                server_group = self.central_server.get_server_group(group_id)
                if server_group:
                    # 聚合服务器模型参数
                    aggregated_server = self._aggregate_models(
                        [copy.deepcopy(server_group.server_model.state_dict())] + states, 
                        [0.3] + [0.7/len(states)] * len(states))
                    
                    # 更新服务器组模型
                    server_group.server_model.load_state_dict(aggregated_server)
                    server_states[group_id] = aggregated_server
                    
                    # 收集分类器状态
                    classifier_states[group_id] = server_group.global_classifier.state_dict()
        
        # 更新客户端模型
        for client_id, state in client_states.items():
            # 提取共享基础层参数
            shared_base_params = {}
            for key, value in state.items():
                if 'shared_base' in key:
                    shared_base_params[key] = value
            
            # 更新客户端模型
            model = client_models[client_id]
            for key, value in shared_base_params.items():
                if key in model.state_dict():
                    model.state_dict()[key].copy_(value)
        
        training_time = time.time() - start_time
        self.logger.info(f"轮次 {round_idx} 并行训练完成，"
                       f"耗时: {training_time:.2f}秒")
        
        # 更新统计信息
        self.stats['training_time'].append(training_time)
        
        # 计算平均准确率
        avg_local_acc = np.mean([r['local_accuracy'] for r in eval_results.values()])
        avg_global_acc = np.mean([r['global_accuracy'] for r in eval_results.values()])
        self.stats['avg_local_acc'].append(avg_local_acc)
        self.stats['avg_global_acc'].append(avg_global_acc)
        
        return {
            'train_results': train_results,
            'eval_results': eval_results,
            'client_states': client_states,
            'server_states': server_states,
            'classifier_states': classifier_states,
            'time_stats': time_stats,
            'training_time': training_time,
            'avg_local_acc': avg_local_acc,
            'avg_global_acc': avg_global_acc
        }
    
    def _aggregate_models(self, model_states, weights=None):
        """聚合模型参数
        
        Args:
            model_states: 模型状态列表
            weights: 权重列表，如果为None则平均聚合
            
        Returns:
            聚合后的模型状态
        """
        if not model_states:
            return {}
        
        # 如果没有提供权重，使用平均权重
        if weights is None:
            weights = [1.0 / len(model_states)] * len(model_states)
        
        # 确保权重长度正确
        assert len(weights) == len(model_states), "权重数量必须等于模型数量"
        
        # 初始化聚合结果
        aggregated = {}
        for key in model_states[0].keys():
            aggregated[key] = torch.zeros_like(model_states[0][key])
        
        # 加权聚合
        for w, state in zip(weights, model_states):
            for key in aggregated.keys():
                if key in state:
                    aggregated[key] += w * state[key]
        
        return aggregated
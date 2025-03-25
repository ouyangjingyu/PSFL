import torch
import torch.multiprocessing as mp
import copy
import time
import numpy as np
from collections import defaultdict
import logging
import threading
import queue

import torch.nn as nn
import torch.nn.functional as F

class ParallelClusterTrainer:
    """
    并行聚类训练器，使不同聚类组能够并行进行训练
    """
    
    def __init__(self, cluster_map, client_models, server_models, 
                 shared_classifier=None, device_map=None, max_workers=None):
        """
        初始化并行聚类训练器
        
        Args:
            cluster_map: 聚类映射，键为聚类ID，值为客户端ID列表
            client_models: 客户端模型字典，键为客户端ID，值为模型
            server_models: 服务器模型字典，键为客户端ID，值为模型
            shared_classifier: 共享分类器（可选）
            device_map: 设备映射，键为聚类ID，值为设备
            max_workers: 最大并行工作线程数
        """
        self.cluster_map = cluster_map
        self.client_models = client_models
        self.server_models = server_models
        self.shared_classifier = shared_classifier
        
        # 如果未提供设备映射，默认所有聚类使用同一设备
        self.device_map = device_map or {}
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for cluster_id in cluster_map.keys():
            if cluster_id not in self.device_map:
                self.device_map[cluster_id] = default_device
        
        # 设置最大并行工作线程数
        if max_workers is None:
            # 默认使用可用GPU数量或CPU核心数的一半
            if torch.cuda.is_available():
                max_workers = torch.cuda.device_count()
            else:
                import multiprocessing
                max_workers = max(1, multiprocessing.cpu_count() // 2)
        
        self.max_workers = max_workers
        
        # 初始化结果存储
        self.cluster_results = {}
        self.client_training_stats = defaultdict(dict)
    
    def train_cluster(self, cluster_id, client_ids, train_fn, eval_fn, **kwargs):
        """
        训练单个聚类中的所有客户端（串行）
        
        Args:
            cluster_id: 聚类ID
            client_ids: 客户端ID列表
            train_fn: 训练函数，接受客户端ID、模型和其他参数
            eval_fn: 评估函数，接受客户端ID、模型和其他参数
            **kwargs: 传递给训练和评估函数的其他参数
            
        Returns:
            聚类训练结果
        """
        device = self.device_map.get(cluster_id, 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 收集该聚类的训练结果
        cluster_result = {
            'client_models': {},
            'server_models': {},
            'training_metrics': {},
            'evaluation_metrics': {}
        }
        
        # 创建全局分类器的完整副本，确保它完全在正确设备上
        if 'global_classifier' in kwargs and kwargs['global_classifier'] is not None:
            # 使用深拷贝创建完全独立的副本
            import copy
            original_classifier = kwargs['global_classifier']
            cluster_classifier = copy.deepcopy(original_classifier)
            
            # 将整个分类器移至设备
            cluster_classifier = cluster_classifier.to(device)
            
            # 额外确保所有参数和缓冲区也在正确设备上
            for param in cluster_classifier.parameters():
                param.data = param.data.to(device)
            for buffer in cluster_classifier.buffers():
                buffer.data = buffer.data.to(device)
                
            # 替换原始分类器
            kwargs['global_classifier'] = cluster_classifier
        
        # 在该聚类中串行训练每个客户端
        for client_id in client_ids:
            # 如果该客户端模型不存在，跳过
            if client_id not in self.client_models:
                print(f"客户端 {client_id} 模型不存在，跳过")
                continue
            
            # 获取客户端和服务器模型，并确保深度复制和完全转移到正确设备
            client_model = copy.deepcopy(self.client_models[client_id])
            server_model = copy.deepcopy(self.server_models[client_id])
            
            # 将模型彻底移至指定设备
            client_model = client_model.to(device)
            server_model = server_model.to(device)
            
            # 修改：确保模型的所有参数都在正确设备上（解决深层嵌套模块问题）
            for param in client_model.parameters():
                param.data = param.data.to(device)
            for param in server_model.parameters():
                param.data = param.data.to(device)
            
            # 如果存在共享分类器，使用共享分类器替换服务器模型的分类器
            # 注意：在新架构中，服务器可能没有分类器，此处代码保留以兼容旧代码
            if self.shared_classifier is not None:
                if hasattr(server_model, 'classifier'):
                    server_classifier = copy.deepcopy(self.shared_classifier).to(device)
                    # 确保分类器的所有参数都在正确设备上
                    for param in server_classifier.parameters():
                        param.data = param.data.to(device)
                    server_model.classifier = server_classifier
                elif hasattr(server_model, 'fc'):
                    server_fc = copy.deepcopy(self.shared_classifier).to(device)
                    # 确保分类器的所有参数都在正确设备上
                    for param in server_fc.parameters():
                        param.data = param.data.to(device)
                    server_model.fc = server_fc
            
            # 训练客户端
            try:
                # 修改：确保device参数也正确传递
                train_result = train_fn(
                    client_id=client_id,
                    client_model=client_model,
                    server_model=server_model,
                    device=device,
                    **kwargs
                )
                
                # 存储训练后的模型和指标
                cluster_result['client_models'][client_id] = client_model.cpu()
                cluster_result['server_models'][client_id] = server_model.cpu()
                cluster_result['training_metrics'][client_id] = train_result
                
                # 更新客户端训练统计信息
                self.client_training_stats[client_id].update({
                    'cluster_id': cluster_id,
                    'training_time': train_result.get('time', 0),
                    'device': device,
                    'success': True
                })
                
                # 评估客户端
                if eval_fn is not None:
                    # 创建评估专用的参数字典，移除训练特有的参数
                    eval_kwargs = kwargs.copy()
                    if 'local_epochs' in eval_kwargs:
                        del eval_kwargs['local_epochs']  # 移除评估函数不需要的参数
                    if 'split_rounds' in eval_kwargs:
                        del eval_kwargs['split_rounds']  # 移除评估函数不需要的参数
                    
                    # 确保global_classifier参数在正确的设备上
                    if 'global_classifier' in eval_kwargs and eval_kwargs['global_classifier'] is not None:
                        eval_kwargs['global_classifier'] = eval_kwargs['global_classifier'].to(device)
                    
                    eval_result = eval_fn(
                        client_id=client_id,
                        client_model=client_model,
                        server_model=server_model,
                        device=device,
                        **eval_kwargs
                    )
                    cluster_result['evaluation_metrics'][client_id] = eval_result
            
            except Exception as e:
                print(f"训练客户端 {client_id} 时出错: {str(e)}")
                self.client_training_stats[client_id].update({
                    'cluster_id': cluster_id,
                    'error': str(e),
                    'success': False
                })
        
        return cluster_result
    
    def _worker(self, cluster_id, client_ids, train_fn, eval_fn, kwargs, result_queue):
        """
        工作线程函数，用于并行训练不同聚类
        """
        try:
            # 创建线程专用的kwargs副本，避免跨线程干扰
            thread_kwargs = kwargs.copy()
            # 调试打印
            print(f"聚类 {cluster_id} 收到的kwargs键: {list(kwargs.keys())}")
            thread_kwargs = kwargs.copy()
            
            # 更多调试
            if 'global_classifier' in thread_kwargs:
                print(f"聚类 {cluster_id} global_classifier类型: {type(thread_kwargs['global_classifier'])}")

            result = self.train_cluster(cluster_id, client_ids, train_fn, eval_fn, **thread_kwargs)
            result_queue.put((cluster_id, result))
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"聚类 {cluster_id} 训练时出错: {str(e)}\n详细错误:\n{error_trace}")
            result_queue.put((cluster_id, {'error': str(e), 'traceback': error_trace}))
    
    def train_all_clusters_parallel(self, train_fn, eval_fn=None, **kwargs):
        """
        并行训练所有聚类
        
        Args:
            train_fn: 训练函数
            eval_fn: 评估函数（可选）
            **kwargs: 其他参数
            
        Returns:
            所有聚类的训练结果
        """
        start_time = time.time()
        
        # 创建结果队列
        result_queue = queue.Queue()
        
        # 创建并启动工作线程
        threads = []
        for cluster_id, client_ids in self.cluster_map.items():
            thread = threading.Thread(
                target=self._worker,
                args=(cluster_id, client_ids, train_fn, eval_fn, kwargs, result_queue)
            )
            threads.append(thread)
        
        # 控制并行度，每次最多启动max_workers个线程
        active_threads = []
        for thread in threads:
            # 如果活动线程达到最大值，等待某个线程完成
            while len(active_threads) >= self.max_workers:
                for t in active_threads[:]:
                    if not t.is_alive():
                        active_threads.remove(t)
                if len(active_threads) >= self.max_workers:
                    time.sleep(0.1)
            
            # 启动新线程
            thread.start()
            active_threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        while not result_queue.empty():
            cluster_id, result = result_queue.get()
            self.cluster_results[cluster_id] = result
        
        total_time = time.time() - start_time
        
        return self.cluster_results, self.client_training_stats, total_time

class GlobalClassifierService:
    """全局分类器服务，处理所有聚类的特征请求"""
    
    def __init__(self, classifier, device='cpu'):
        """
        初始化全局分类器服务
        
        Args:
            classifier: 全局分类器模型
            device: 运行设备
        """
        self.classifier = classifier.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def process_batch(self, features_batch, labels_batch, cluster_ids=None):
        """
        批量处理来自不同聚类的特征
        
        Args:
            features_batch: 特征张量列表或字典
            labels_batch: 标签张量列表或字典
            cluster_ids: 聚类ID列表
            
        Returns:
            gradients_dict: 每个输入对应的梯度字典
            loss_dict: 每个输入对应的损失字典
            metrics_dict: 准确率等指标字典
        """
        self.optimizer.zero_grad()
        
        # 准备批处理结果存储
        gradients_dict = {}
        loss_dict = {}
        metrics_dict = {}
        
        # 如果输入是字典形式
        if isinstance(features_batch, dict):
            # 批量处理所有特征
            all_features = []
            all_labels = []
            batch_indices = {}
            start_idx = 0
            
            # 将不同聚类的特征和标签组合成批次
            for cluster_id, features in features_batch.items():
                if cluster_id not in labels_batch:
                    continue
                    
                features = features.to(self.device)
                labels = labels_batch[cluster_id].to(self.device)
                
                batch_size = features.size(0)
                all_features.append(features)
                all_labels.append(labels)
                
                # 记录每个聚类数据在批次中的位置
                batch_indices[cluster_id] = (start_idx, start_idx + batch_size)
                start_idx += batch_size
            
            if all_features:
                # 连接所有特征和标签
                all_features = torch.cat(all_features, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                
                # 前向传播
                outputs = self.classifier(all_features)
                loss = self.criterion(outputs, all_labels)
                
                # 反向传播计算梯度
                loss.backward()
                
                # 优化器步进
                self.optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == all_labels).float().mean().item() * 100
                
                # 为每个聚类提取梯度和损失
                for cluster_id, (start_idx, end_idx) in batch_indices.items():
                    cluster_features = all_features[start_idx:end_idx]
                    cluster_labels = all_labels[start_idx:end_idx]
                    cluster_outputs = outputs[start_idx:end_idx]
                    
                    # 计算聚类损失
                    cluster_loss = self.criterion(cluster_outputs, cluster_labels).item()
                    loss_dict[cluster_id] = cluster_loss
                    
                    # 提取聚类梯度（对于输入特征）
                    if cluster_features.grad is not None:
                        gradients_dict[cluster_id] = cluster_features.grad.clone().detach()
                    
                    # 计算聚类准确率
                    cluster_predicted = predicted[start_idx:end_idx]
                    cluster_accuracy = (cluster_predicted == cluster_labels).float().mean().item() * 100
                    metrics_dict[cluster_id] = {'accuracy': cluster_accuracy}
            
        # 如果输入是列表形式
        elif isinstance(features_batch, list):
            for i, features in enumerate(features_batch):
                if i >= len(labels_batch):
                    continue
                    
                cluster_id = cluster_ids[i] if cluster_ids and i < len(cluster_ids) else i
                features = features.to(self.device)
                labels = labels_batch[i].to(self.device)
                
                # 前向传播
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                
                # 反向传播计算梯度
                loss.backward(retain_graph=(i < len(features_batch) - 1))
                
                # 提取梯度和损失
                if features.grad is not None:
                    gradients_dict[cluster_id] = features.grad.clone().detach()
                loss_dict[cluster_id] = loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).float().mean().item() * 100
                metrics_dict[cluster_id] = {'accuracy': accuracy}
            
            # 优化器步进
            self.optimizer.step()
        
        return gradients_dict, loss_dict, metrics_dict

class ResourceAwareScheduler:
    """
    资源感知调度器，根据客户端的计算能力和网络速度进行优化调度
    """
    
    def __init__(self, client_resources, cluster_map=None):
        """
        初始化资源感知调度器
        
        Args:
            client_resources: 客户端资源信息字典，包含计算能力和网络速度
            cluster_map: 初始聚类映射（可选）
        """
        self.client_resources = client_resources
        self.cluster_map = cluster_map or {}
        
        # 初始化设备映射
        self.device_map = {}
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
        else:
            self.gpu_count = 0
    
    def optimize_cluster_assignment(self, clients, n_clusters=3):
        """
        优化聚类分配，考虑客户端资源异构性
        
        Args:
            clients: 客户端ID列表
            n_clusters: 聚类数量
            
        Returns:
            优化后的聚类映射
        """
        # 根据资源进行排序
        def compute_resource_score(client_id):
            resource = self.client_resources.get(client_id, {})
            compute_power = resource.get('compute_power', 1.0)
            network_speed = resource.get('network_speed', 1.0)
            return compute_power * 0.7 + network_speed * 0.3
        
        # 对客户端按资源分数排序
        client_scores = [(client_id, compute_resource_score(client_id)) for client_id in clients]
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 分配客户端到聚类
        optimized_clusters = {i: [] for i in range(n_clusters)}
        
        # 使用贪心算法：先分配资源多的客户端到不同聚类
        for i, (client_id, _) in enumerate(client_scores):
            cluster_id = i % n_clusters
            optimized_clusters[cluster_id].append(client_id)
        
        return optimized_clusters
    
    def allocate_devices(self, optimized_clusters):
        """
        为聚类分配计算设备
        
        Args:
            optimized_clusters: 优化后的聚类映射
            
        Returns:
            设备映射字典
        """
        device_map = {}
        
        # 如果有GPU可用，优先分配GPU
        if self.gpu_count > 0:
            # 根据聚类规模排序
            cluster_sizes = [(cluster_id, len(clients)) for cluster_id, clients in optimized_clusters.items()]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # 将最大的聚类分配到GPU
            for i, (cluster_id, _) in enumerate(cluster_sizes):
                if i < self.gpu_count:
                    device_map[cluster_id] = f'cuda:{i}'
                else:
                    device_map[cluster_id] = 'cpu'
        else:
            # 如果没有GPU，所有聚类使用CPU
            for cluster_id in optimized_clusters.keys():
                device_map[cluster_id] = 'cpu'
        
        return device_map
    
    def schedule_training(self, cluster_map=None, client_models=None):
        """
        调度训练任务，优化设备分配和训练顺序
        
        Args:
            cluster_map: 聚类映射（可选）
            client_models: 客户端模型字典（可选）
            
        Returns:
            优化后的训练调度
        """
        # 使用提供的聚类映射或已有的
        cluster_map = cluster_map or self.cluster_map
        
        # 优化设备分配
        device_map = self.allocate_devices(cluster_map)
        
        # 对每个聚类内的客户端进行优化排序
        optimized_order = {}
        
        for cluster_id, clients in cluster_map.items():
            # 按资源评分排序
            def compute_task_score(client_id):
                resource = self.client_resources.get(client_id, {})
                compute_power = resource.get('compute_power', 1.0)
                model_size = 1.0
                
                # 如果提供了模型，考虑模型大小
                if client_models and client_id in client_models:
                    if hasattr(client_models[client_id], 'size'):
                        model_size = client_models[client_id].size
                
                # 优先处理计算能力小、模型小的客户端
                return compute_power * model_size
            
            # 排序客户端
            sorted_clients = sorted(clients, key=compute_task_score)
            optimized_order[cluster_id] = sorted_clients
        
        return {
            'device_map': device_map,
            'optimized_order': optimized_order
        }


class TrainingCoordinator:
    """
    训练协调器，管理整个训练过程，包括模型的分发和聚合
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练协调器
        
        Args:
            device: 默认计算设备
        """
        self.device = device
        self.client_models = {}
        self.server_models = {}
        self.client_resources = {}
        self.shared_classifier = None
        self.global_model = None
        self.cluster_models = {}
        self.scheduler = None
        self.trainer = None
        
        # 初始化诊断追踪器
        self.diagnostic_tracker = None
        
        # 设置日志记录
        self.logger = logging.getLogger("TrainingCoordinator")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def register_client(self, client_id, client_model, server_model, resources=None):
        """
        注册客户端及其模型
        
        Args:
            client_id: 客户端ID
            client_model: 客户端模型
            server_model: 对应的服务器模型
            resources: 客户端资源信息（可选）
        """
        self.client_models[client_id] = client_model
        self.server_models[client_id] = server_model
        
        if resources:
            self.client_resources[client_id] = resources
    
    def set_shared_classifier(self, classifier):
        """
        设置共享分类器
        
        Args:
            classifier: 统一分类器模型
        """
        self.shared_classifier = classifier
    
    def set_global_model(self, global_model):
        """
        设置全局模型
        
        Args:
            global_model: 全局模型参数
        """
        self.global_model = global_model
    
    def set_cluster_models(self, cluster_models):
        """
        设置聚类模型
        
        Args:
            cluster_models: 聚类模型字典
        """
        self.cluster_models = cluster_models
    
    def set_diagnostic_tracker(self, tracker):
        """
        设置诊断追踪器
        
        Args:
            tracker: 模型诊断追踪器
        """
        self.diagnostic_tracker = tracker
    
    def setup_training(self, cluster_map, max_workers=None):
        """
        设置训练环境
        
        Args:
            cluster_map: 聚类映射
            max_workers: 最大并行工作线程数
        """
        # 初始化资源感知调度器
        self.scheduler = ResourceAwareScheduler(self.client_resources, cluster_map)
        
        # 获取优化的训练调度
        schedule = self.scheduler.schedule_training(cluster_map, self.client_models)
        device_map = schedule['device_map']
        
        # 使用优化的聚类顺序更新聚类映射
        optimized_cluster_map = {}
        for cluster_id, clients in schedule['optimized_order'].items():
            optimized_cluster_map[cluster_id] = clients
        
        # 初始化并行聚类训练器
        self.trainer = ParallelClusterTrainer(
            optimized_cluster_map,
            self.client_models,
            self.server_models,
            self.shared_classifier,
            device_map,
            max_workers
        )
        
        self.logger.info(f"训练设置完成，聚类数量: {len(optimized_cluster_map)}")
        for cluster_id, clients in optimized_cluster_map.items():
            self.logger.info(f"聚类 {cluster_id}: {len(clients)} 客户端，设备: {device_map[cluster_id]}")
    
    def execute_training(self, train_fn, eval_fn=None, **kwargs):
        """
        执行训练过程
        
        Args:
            train_fn: 客户端训练函数
            eval_fn: 客户端评估函数（可选）
            **kwargs: 其他参数
            
        Returns:
            训练结果和统计信息
        """
        print(f"执行训练，shared_classifier: {self.shared_classifier is not None}")
        
        if self.trainer is None:
            self.logger.error("训练器未初始化，请先调用setup_training")
            return None, None, 0
        
        # 添加全局分类器到kwargs - 确保使用kwargs['global_classifier']
        if self.shared_classifier is not None:
            kwargs['global_classifier'] = self.shared_classifier
        
        # 执行并行训练
        start_time = time.time()
        self.logger.info("开始并行训练...")
        
        cluster_results, client_stats, training_time = self.trainer.train_all_clusters_parallel(
            train_fn, eval_fn, **kwargs
        )
        
        self.logger.info(f"并行训练完成，耗时: {training_time:.2f} 秒")
        
        # 统计每个聚类的训练情况
        for cluster_id, result in cluster_results.items():
            # 过滤掉错误
            if 'error' in result:
                self.logger.error(f"聚类 {cluster_id} 训练出错: {result['error']}")
                continue
                
            # 统计成功训练的客户端数量
            success_count = len(result.get('client_models', {}))
            metrics = result.get('training_metrics', {})
            avg_loss = np.mean([m.get('loss', 0) for m in metrics.values() if 'loss' in m])
            avg_acc = np.mean([m.get('accuracy', 0) for m in metrics.values() if 'accuracy' in m])
            
            self.logger.info(f"聚类 {cluster_id}: {success_count} 客户端完成训练, 平均损失: {avg_loss:.4f}, 平均准确率: {avg_acc:.2f}%")
        
        total_time = time.time() - start_time
        
        # 返回训练结果
        return cluster_results, client_stats, total_time
        
    def collect_trained_models(self, cluster_results):
        """
        收集训练后的模型
        
        Args:
            cluster_results: 聚类训练结果
            
        Returns:
            client_models_params: 客户端模型参数字典
            client_weights: 客户端权重字典
        """
        client_models_params = {}
        client_weights = {}
        
        for cluster_id, result in cluster_results.items():
            # 过滤掉错误
            if 'error' in result:
                continue
            
            # 收集训练指标
            training_metrics = result.get('training_metrics', {})
            
            # 收集每个客户端的训练后模型状态
            for client_id, metrics in training_metrics.items():
                # 如果训练成功并保存了模型状态
                if 'client_model_state' in metrics and 'server_model_state' in metrics:
                    # 使用训练后保存的模型状态
                    client_models_params[client_id] = metrics['client_model_state']
                    
                    # 基于训练数据量设置权重
                    data_size = metrics.get('data_size', 1.0)
                    client_weights[client_id] = float(data_size)
                elif not isinstance(metrics, dict):
                    # 跳过非字典类型的指标
                    continue
                else:
                    # 如果没有直接保存模型状态但有模型
                    if client_id in result.get('client_models', {}):
                        # 获取模型状态字典
                        model = result['client_models'][client_id]
                        if hasattr(model, 'state_dict'):
                            client_models_params[client_id] = model.state_dict()
                        else:
                            client_models_params[client_id] = model
                        
                        # 基于训练数据量设置权重
                        data_size = metrics.get('data_size', 1.0)
                        client_weights[client_id] = float(data_size)
        
        print(f"收集到 {len(client_models_params)} 个客户端的训练后模型")
        return client_models_params, client_weights

class TrainingCoordinatorWithGlobalClassifier(TrainingCoordinator):
    """扩展训练协调器，支持全局分类器串行处理"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TrainingCoordinatorWithGlobalClassifier, self).__init__(device)
        self.global_classifier_service = None
        self.features_queue = {}
        self.labels_queue = {}
        self.results_queue = {}
        self.batch_size = 16  # 默认批处理大小
        
    def set_shared_classifier(self, classifier):
        """设置共享分类器并创建服务"""
        super().set_shared_classifier(classifier)
        self.global_classifier_service = GlobalClassifierService(
            classifier, 
            device=self.device
        )
    
    def register_features(self, cluster_id, features, labels):
        """注册聚类的特征和标签到队列"""
        if cluster_id not in self.features_queue:
            self.features_queue[cluster_id] = []
            self.labels_queue[cluster_id] = []
        
        self.features_queue[cluster_id].append(features)
        self.labels_queue[cluster_id].append(labels)
        
        # 检查是否可以进行批处理
        self._try_batch_process()
    
    def _try_batch_process(self):
        """尝试批量处理队列中的特征"""
        # 检查是否有足够的数据进行批处理
        total_batches = sum(len(queue) for queue in self.features_queue.values())
        
        if total_batches >= self.batch_size or (total_batches > 0 and all(len(queue) > 0 for queue in self.features_queue.values())):
            # 准备批处理数据
            batch_features = {}
            batch_labels = {}
            
            for cluster_id, features_list in self.features_queue.items():
                if not features_list:
                    continue
                    
                # 获取该聚类的一批特征和标签
                features = torch.cat(features_list, dim=0)
                labels = torch.cat(self.labels_queue[cluster_id], dim=0)
                
                batch_features[cluster_id] = features
                batch_labels[cluster_id] = labels
                
                # 清空队列
                self.features_queue[cluster_id] = []
                self.labels_queue[cluster_id] = []
            
            # 执行批处理
            if batch_features:
                gradients, losses, metrics = self.global_classifier_service.process_batch(
                    batch_features, batch_labels
                )
                
                # 保存结果
                for cluster_id in batch_features.keys():
                    if cluster_id not in self.results_queue:
                        self.results_queue[cluster_id] = []
                    
                    result = {
                        'gradients': gradients.get(cluster_id),
                        'loss': losses.get(cluster_id),
                        'metrics': metrics.get(cluster_id)
                    }
                    
                    self.results_queue[cluster_id].append(result)
    
    def get_results(self, cluster_id):
        """获取指定聚类的处理结果"""
        if cluster_id in self.results_queue and self.results_queue[cluster_id]:
            return self.results_queue[cluster_id].pop(0)
        
        return None
    
    def process_features_sync(self, cluster_id, features, labels):
        """同步处理特征（非队列模式）"""
        if self.global_classifier_service is None:
            raise ValueError("全局分类器服务未初始化")
        
        batch_features = {cluster_id: features}
        batch_labels = {cluster_id: labels}
        
        gradients, losses, metrics = self.global_classifier_service.process_batch(
            batch_features, batch_labels
        )
        
        return {
            'gradients': gradients.get(cluster_id),
            'loss': losses.get(cluster_id),
            'metrics': metrics.get(cluster_id)
        }


def create_training_framework(client_models, server_models, client_resources=None):
    """
    创建完整的训练框架
    
    Args:
        client_models: 客户端模型字典
        server_models: 服务器模型字典
        client_resources: 客户端资源信息字典（可选）
        
    Returns:
        训练协调器
    """
    # 初始化训练协调器
    coordinator = TrainingCoordinator()
    
    # 注册客户端
    for client_id, client_model in client_models.items():
        server_model = server_models.get(client_id)
        if server_model is not None:
            resources = None
            if client_resources and client_id in client_resources:
                resources = client_resources[client_id]
            coordinator.register_client(client_id, client_model, server_model, resources)
    
    return coordinator

# 添加新的框架创建函数
def create_training_framework_with_global_classifier(client_models, server_models, client_resources=None):
    """
    创建带有全局分类器服务的训练框架
    
    Args:
        client_models: 客户端模型字典
        server_models: 服务器模型字典
        client_resources: 客户端资源信息字典（可选）
        
    Returns:
        训练协调器
    """
    # 初始化扩展的训练协调器
    coordinator = TrainingCoordinatorWithGlobalClassifier()
    
    # 注册客户端
    for client_id, client_model in client_models.items():
        server_model = server_models.get(client_id)
        if server_model is not None:
            resources = None
            if client_resources and client_id in client_resources:
                resources = client_resources[client_id]
            coordinator.register_client(client_id, client_model, server_model, resources)
    
    return coordinator
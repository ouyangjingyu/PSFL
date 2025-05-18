import torch
import threading
import queue
import time
import copy
import numpy as np
from collections import defaultdict
import logging
import random
import torch.nn.functional as F
import math
# 替换为：
import sys
import os
# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from model.resnet import EnhancedServerModel, ImprovedGlobalClassifier

class UnifiedParallelTrainer:
    """新的统一并行训练器 - 简化并行训练流程"""
    def __init__(self, client_manager, server_model, global_classifier, 
                loss_fn=None, device="cuda", max_workers=None, training_controller=None):
        self.client_manager = client_manager
        self.server_model = server_model
        self.global_classifier = global_classifier
        self.default_device = device
        
        # 处理损失函数
        if loss_fn is None:
            self.loss_fn = EnhancedUnifiedLoss()
        else:
            self.loss_fn = loss_fn
        
        # 保存训练控制器
        self.training_controller = training_controller
        
        # 设置最大并行工作线程数
        if max_workers is None:
            if torch.cuda.is_available():
                max_workers = torch.cuda.device_count()
            else:
                import multiprocessing
                max_workers = max(1, multiprocessing.cpu_count() // 2)
        
        self.max_workers = max(1, max_workers)
        
        # 初始化日志
        self.logger = logging.getLogger("UnifiedParallelTrainer")
        self.logger.setLevel(logging.INFO)
        
        # 客户端模型字典
        self.client_models = {}
        
        # 聚类和设备映射
        self.cluster_map = {}
        self.device_map = {}
        
        # 客户端梯度缓存
        self.client_gradients = {}
        
        # 批处理大小 - 根据设备内存调整
        self.client_batch_size = 2  # 处理客户端的批量大小
        self.feature_batch_size = 32  # 特征处理的批量大小
        
    def register_client_models(self, client_models_dict):
        """注册客户端模型"""
        self.client_models.update(client_models_dict)
    
    def setup_training(self, cluster_map, device_map=None):
        """设置训练环境"""
        self.cluster_map = cluster_map
        
        # 如果没有设备映射，创建一个简单的映射
        if device_map is None:
            device_map = {}
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i, cluster_id in enumerate(cluster_map.keys()):
                    device_map[cluster_id] = f"cuda:{i % gpu_count}"
            else:
                for cluster_id in cluster_map.keys():
                    device_map[cluster_id] = "cpu"
        
        self.device_map = device_map
        self.logger.info(f"训练设置完成，聚类数量: {len(cluster_map)}")
        
    def execute_training(self, round_idx=0):
        """执行统一的并行训练流程"""
        start_time = time.time()
        
        # 结果容器
        train_results = {}
        eval_results = {}
        shared_states = {}
        time_stats = {}
        
        # 1. 并行处理各聚类
        cluster_results = self._parallel_process_clusters(round_idx)
        
        # 2. 合并结果
        for result in cluster_results:
            if 'train_results' in result:
                train_results.update(result['train_results'])
            if 'eval_results' in result:
                eval_results.update(result['eval_results'])
            if 'shared_states' in result:
                shared_states.update(result['shared_states'])
            if 'time_stats' in result:
                time_stats.update(result['time_stats'])
        
        # 3. 更新损失函数历史
        if hasattr(self.loss_fn, 'update_history'):
            self.loss_fn.update_history(eval_results)
        
        training_time = time.time() - start_time
        self.logger.info(f"统一训练完成，总耗时: {training_time:.2f}秒")
        
        return train_results, eval_results, shared_states, time_stats, training_time
    
    def _parallel_process_clusters(self, round_idx):
        """并行处理各聚类"""
        results = []
        threads = []
        results_queue = queue.Queue()
        
        # 创建聚类处理线程
        for cluster_id, client_ids in self.cluster_map.items():
            thread = threading.Thread(
                target=self._process_cluster,
                args=(cluster_id, client_ids, round_idx, results_queue)
            )
            threads.append(thread)
        
        # 控制并行度
        active_threads = []
        for thread in threads:
            # 等待有空闲槽位
            while len(active_threads) >= self.max_workers:
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
        while not results_queue.empty():
            results.append(results_queue.get())
        
        return results
    
    def _process_cluster(self, cluster_id, client_ids, round_idx, results_queue):
        """处理单个聚类"""
        try:
            # 获取设备
            device = self.device_map.get(cluster_id, self.default_device)
            
            # 直接创建新模型实例，传入必要参数但不依赖实例属性
            server_model = EnhancedServerModel(
                model_type='resnet56',  # 直接硬编码或从配置中获取
                feature_dim=128
            ).to(device)
            
            # 加载状态
            if hasattr(self, 'cluster_server_models') and cluster_id in self.cluster_server_models:
                server_model.load_state_dict(self.cluster_server_models[cluster_id])
            else:
                server_model.load_state_dict(self.server_model.state_dict())
            
            # 同理处理分类器
            classifier = ImprovedGlobalClassifier(
                feature_dim=128,
                num_classes=10  # 根据您的数据集调整或从全局分类器推断
            ).to(device)
            classifier.load_state_dict(self.global_classifier.state_dict())
            
            # 结果容器
            cluster_train_results = {}
            cluster_eval_results = {}
            cluster_shared_states = {}
            cluster_time_stats = {}
            
            # 1. 并行处理客户端个性化层（分批处理以节省内存）
            client_batches = [client_ids[i:i+self.client_batch_size] 
                              for i in range(0, len(client_ids), self.client_batch_size)]
            
            for batch in client_batches:
                # 并行处理客户端个性化层
                personal_results = {}
                for client_id in batch:
                    client = self.client_manager.get_client(client_id)
                    if not client or client_id not in self.client_models:
                        continue
                    
                    # 获取客户端模型
                    client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                    
                    # 设置临时设备
                    original_device = client.device
                    client.device = device
                    client.model = client_model
                    
                    # 训练个性化层
                    try:
                        result = self._train_client_personal_layers(client, client_model, round_idx)
                        personal_results[client_id] = result
                    except Exception as e:
                        self.logger.error(f"客户端 {client_id} 个性化层训练失败: {str(e)}")
                    
                    # 恢复原始设备
                    client.device = original_device
                    
                    # 保存更新后的模型
                    self.client_models[client_id] = client_model.cpu()
                    
                    # 释放内存
                    torch.cuda.empty_cache()
                
                # 2. 串行处理服务器训练
                for client_id, result in personal_results.items():
                    client = self.client_manager.get_client(client_id)
                    client_model = self.client_models[client_id].to(device)
                    
                    # 设置临时设备
                    client.device = device
                    client.model = client_model
                    
                    # 处理服务器训练
                    features = result['features']
                    local_loss = result['local_loss']
                    time_cost = result['time_cost']
                    
                    # 进行服务器训练，获取共享层梯度
                    shared_grads, train_stats = self._train_server_step(
                        server_model, classifier, client_model, features, local_loss, client_id, round_idx, device)
                    
                    # 应用共享层梯度
                    self._apply_shared_grads(client_model, shared_grads)
                    
                    # 保存结果
                    cluster_train_results[client_id] = train_stats
                    cluster_time_stats[client_id] = {'training_time': time_cost + train_stats.get('time_cost', 0)}
                    
                    # 评估客户端
                    eval_result = self._evaluate_client(
                        client, client_model, server_model, classifier)
                    cluster_eval_results[client_id] = eval_result
                    
                    # 保存共享层状态
                    shared_state = {}
                    for name, param in client_model.named_parameters():
                        if 'shared_base' in name:
                            shared_state[name] = param.data.clone()
                    cluster_shared_states[client_id] = shared_state
                    
                    # 恢复设备
                    client.device = original_device
                    
                    # 更新客户端模型
                    self.client_models[client_id] = client_model.cpu()
                    
                    # 释放内存
                    torch.cuda.empty_cache()
            
            # 保存训练后的服务器模型状态
            if not hasattr(self, 'cluster_server_models'):
                self.cluster_server_models = {}
            self.cluster_server_models[cluster_id] = server_model.state_dict()
            
            # 将结果添加到队列
            results_queue.put({
                'cluster_id': cluster_id,
                'server_model': server_model.state_dict(),  # 添加服务器模型状态
                'train_results': cluster_train_results,
                'eval_results': cluster_eval_results,
                'shared_states': cluster_shared_states,
                'time_stats': cluster_time_stats
            })
            
        except Exception as e:
            import traceback
            self.logger.error(f"聚类 {cluster_id} 处理失败: {str(e)}\n{traceback.format_exc()}")
            results_queue.put({'cluster_id': cluster_id, 'error': str(e)})
    
    def _train_client_personal_layers(self, client, client_model, round_idx):
        """训练客户端个性化层"""
        # 记录开始时间
        start_time = time.time()
        
        # 调用客户端的个性化层训练方法
        stats, collected_features = client.train_personalized_layers(round_idx, self.rounds)
        
        # 计算训练时间
        time_cost = time.time() - start_time
        
        return {
            'features': collected_features,
            'local_loss': stats['local_loss'],
            'local_accuracy': stats['local_accuracy'],
            'time_cost': time_cost
        }
    
    def _train_server_step(self, server_model, classifier, client_model, features, local_loss, client_id, round_idx, device=None):
        """执行服务器训练步骤 - 使用传入的损失函数"""
        if device is None:
            # 回退方案：从参数获取设备
            device = next(server_model.parameters()).device
        
        # 设置训练模式
        server_model.train()
        classifier.train()
        
        # 设置共享层为可训练
        for name, param in client_model.named_parameters():
            if 'shared_base' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 创建优化器
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=0.0005
        )
        
        # 统计信息
        stats = {
            'total_loss': 0.0,
            'global_loss': 0.0,
            'local_loss': local_loss,
            'feature_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 批处理特征数据以节省内存
        num_samples = len(features)
        num_batches = (num_samples + self.feature_batch_size - 1) // self.feature_batch_size
        
        for i in range(num_batches):
            start_idx = i * self.feature_batch_size
            end_idx = min(start_idx + self.feature_batch_size, num_samples)
            batch_features = features[start_idx:end_idx]
            
            # 合并批次内的特征
            shared_features_batch = torch.cat([f['shared_features'] for f in batch_features])
            personal_features_batch = torch.cat([f['personal_features'] for f in batch_features]) if 'personal_features' in batch_features[0] else None
            targets_batch = torch.cat([f['targets'] for f in batch_features])
            
            # 清除梯度
            server_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            shared_optimizer.zero_grad()
            
            # 服务器前向传播
            server_features = server_model(shared_features_batch)
            global_logits = classifier(server_features)
            
            # 使用传入的损失函数计算损失
            total_rounds = getattr(self, 'rounds', 100)
            
            if personal_features_batch is not None and hasattr(self.loss_fn, 'forward'):
                # 使用完整的UnifiedLoss
                total_loss, global_loss, feature_loss = self.loss_fn(
                    global_logits=global_logits,
                    targets=targets_batch,
                    local_loss=local_loss,
                    personal_features=personal_features_batch,
                    server_features=server_features,
                    round_idx=round_idx,
                    total_rounds=total_rounds
                )
            else:
                # 简化损失计算
                global_loss = F.cross_entropy(global_logits, targets_batch)
                # 如果有AdaptiveTrainingController，使用它调整权重
                if hasattr(self, 'training_controller'):
                    weights = self.training_controller.get_current_weights()
                    alpha, beta = weights['alpha'], weights['beta']
                else:
                    # 默认权重
                    progress = round_idx / total_rounds
                    alpha = 0.3 + 0.4 * progress
                    beta = 0.1 * (1 - abs(2 * progress - 1))
                
                # 简单特征对齐损失
                feature_loss = torch.tensor(0.0, device=device)
                total_loss = (1 - alpha) * global_loss + alpha * local_loss + beta * feature_loss
            
            # 反向传播
            total_loss.backward()
            
            # 更新参数
            server_optimizer.step()
            classifier_optimizer.step()
            shared_optimizer.step()
            
            # 更新统计信息
            stats['total_loss'] += total_loss.item()
            stats['global_loss'] += global_loss.item()
            stats['feature_loss'] += feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss
            stats['batch_count'] += 1
            
            # 计算准确率
            _, pred = global_logits.max(1)
            stats['correct'] += pred.eq(targets_batch).sum().item()
            stats['total'] += targets_batch.size(0)
            
            # 释放内存
            del shared_features_batch, targets_batch
            if personal_features_batch is not None:
                del personal_features_batch
            torch.cuda.empty_cache()
        
        # 计算平均值
        if stats['batch_count'] > 0:
            for key in ['total_loss', 'global_loss', 'feature_loss']:
                stats[key] /= stats['batch_count']
        
        # 计算全局准确率
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
            
        # 计算训练时间
        stats['time_cost'] = time.time() - start_time
        
        # 获取共享层梯度
        shared_grads = {}
        for name, param in client_model.named_parameters():
            if 'shared_base' in name and param.grad is not None:
                shared_grads[name] = param.grad.clone()
        
        return shared_grads, stats
    
    def _apply_shared_grads(self, client_model, shared_grads):
        """应用共享层梯度到客户端模型"""
        # 创建优化器
        optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n],
            lr=0.0005
        )
        
        # 应用梯度
        for name, param in client_model.named_parameters():
            if 'shared_base' in name and name in shared_grads:
                param.grad = shared_grads[name]
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
    
    def _evaluate_client(self, client, client_model, server_model, classifier):
        """评估客户端模型"""
        device = client.device
        
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        classifier.eval()
        
        # 统计信息
        local_correct = 0
        global_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in client.test_data:
                # 移至设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                local_logits, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                global_logits = classifier(server_features)
                
                # 计算准确率
                _, local_pred = local_logits.max(1)
                _, global_pred = global_logits.max(1)
                
                local_correct += local_pred.eq(target).sum().item()
                global_correct += global_pred.eq(target).sum().item()
                total += target.size(0)
        
        # 计算准确率
        local_accuracy = 100.0 * local_correct / max(1, total)
        global_accuracy = 100.0 * global_correct / max(1, total)
        
        return {
            'local_accuracy': local_accuracy,
            'global_accuracy': global_accuracy,
            'total_samples': total
        }

    def collect_server_models(self):
        """收集所有聚类的服务器模型状态"""
        server_states = {}
        
        # 创建临时服务器模型用于收集状态
        for cluster_id in self.cluster_map.keys():
            # 获取设备
            device = self.device_map.get(cluster_id, self.default_device)
            
            # 创建临时模型用于收集状态
            temp_server = copy.deepcopy(self.server_model).to(device)
            
            # 尝试从缓存中获取服务器模型状态
            if hasattr(self, 'cluster_server_models') and cluster_id in self.cluster_server_models:
                temp_server.load_state_dict(self.cluster_server_models[cluster_id])
                server_states[cluster_id] = temp_server.state_dict()
            else:
                self.logger.warning(f"未找到聚类 {cluster_id} 的服务器模型状态")
                # 使用主服务器模型状态
                server_states[cluster_id] = self.server_model.state_dict()
        
        return server_states

    def update_server_models(self, aggregated_server_model):
        """更新所有聚类的服务器模型"""
        # 确保存在缓存字典
        if not hasattr(self, 'cluster_server_models'):
            self.cluster_server_models = {}
        
        # 更新每个聚类的服务器模型状态
        for cluster_id in self.cluster_map.keys():
            self.cluster_server_models[cluster_id] = copy.deepcopy(aggregated_server_model)
        
        return True



        
class ModelFeatureClusterer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clustering_history = []
    
    # 在ModelFeatureClusterer类中，优化cluster_clients方法，考虑无法获取原始数据的情况
    def cluster_clients(self, client_models, client_ids, eval_dataset=None, device='cuda'):
        """基于模型特征的聚类方法，考虑服务器不能获取原始数据的限制"""
        clusters = {i: [] for i in range(self.num_clusters)}
        
        # 提取模型特征 - 使用共享层参数而非原始数据
        client_features = {}
        feature_dims = []
        
        # 第一步：提取特征
        for client_id in client_ids:
            if client_id in client_models:
                model = client_models[client_id]
                features = []
                
                # 只提取共享层参数作为特征
                for name, param in model.named_parameters():
                    if 'shared_base' in name and 'weight' in name:
                        # 提取统计信息而非原始参数
                        param_data = param.detach().cpu()
                        # 只收集标量特征，避免形状不一致
                        features.extend([
                            param_data.mean().item(),
                            param_data.std().item(),
                            param_data.abs().max().item(),
                            (param_data > 0).float().mean().item()  # 正值比例
                        ])
                
                if features:
                    # 确保features是一维数组
                    features_array = np.array(features, dtype=np.float32)
                    client_features[client_id] = features_array
                    feature_dims.append(len(features_array))
        
        # 检查所有特征向量的维度是否一致
        if feature_dims and len(set(feature_dims)) > 1:
            # 如果维度不一致，找出最常见的维度
            from collections import Counter
            dim_counter = Counter(feature_dims)
            common_dim = dim_counter.most_common(1)[0][0]
            
            print(f"发现不同维度的特征向量: {dict(dim_counter)}，使用最常见维度: {common_dim}")
            
            # 处理维度不一致的特征向量
            for client_id in list(client_features.keys()):
                feat = client_features[client_id]
                if len(feat) != common_dim:
                    if len(feat) < common_dim:
                        # 如果特征太短，使用填充
                        client_features[client_id] = np.pad(feat, (0, common_dim - len(feat)), 'constant')
                    else:
                        # 如果特征太长，进行裁剪
                        client_features[client_id] = feat[:common_dim]
        
        # 尝试K-means聚类
        if len(client_features) >= self.num_clusters:
            try:
                from sklearn.cluster import KMeans
                # 转换为矩阵
                feature_client_ids = list(client_features.keys())
                features_matrix = np.vstack([client_features[cid] for cid in feature_client_ids])
                
                # 标准化特征
                mean = np.mean(features_matrix, axis=0)
                std = np.std(features_matrix, axis=0) + 1e-8
                features_matrix = (features_matrix - mean) / std
                
                # 执行K-means
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
                kmeans.fit(features_matrix)
                
                # 构建聚类映射
                for i, label in enumerate(kmeans.labels_):
                    client_id = feature_client_ids[i]
                    clusters[label].append(client_id)
                
                # 处理没有特征的客户端 - 平均分配
                remaining_clients = [cid for cid in client_ids if cid not in client_features]
                for i, client_id in enumerate(remaining_clients):
                    target_cluster = i % self.num_clusters
                    clusters[target_cluster].append(client_id)
                    
            except Exception as e:
                print(f"K-means聚类失败: {str(e)}，使用备选方案")
                # 备用方案 - 均匀分配
                for i, client_id in enumerate(client_ids):
                    cluster_idx = i % self.num_clusters
                    clusters[cluster_idx].append(client_id)
        else:
            # 备用方案：均匀分配
            for i, client_id in enumerate(client_ids):
                cluster_idx = i % self.num_clusters
                clusters[cluster_idx].append(client_id)
        
        # 记录聚类结果
        self.clustering_history.append({
            'timestamp': time.time(),
            'clusters': copy.deepcopy(clusters),
            'num_clients': len(client_ids)
        })
            
        return clusters
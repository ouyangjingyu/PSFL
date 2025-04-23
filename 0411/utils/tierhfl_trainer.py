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


# 簇感知并行训练器 - 支持聚类并行训练
class ClusterAwareParallelTrainer:
    def __init__(self, client_manager, server_model, global_classifier=None, device="cuda", max_workers=None, loss_fn=None):
        """初始化训练器"""
        self.client_manager = client_manager
        self.server_model = server_model
        self.global_classifier = global_classifier
        self.default_device = device
        self.loss_fn = loss_fn  # 保存损失函数
        
        # 设置最大并行工作线程数
        if max_workers is None:
            if torch.cuda.is_available():
                max_workers = torch.cuda.device_count()
            else:
                import multiprocessing
                max_workers = max(1, multiprocessing.cpu_count() // 2)
        
        self.max_workers = max(1, max_workers)  # 至少1个工作线程
        
        # 初始化日志
        self.logger = logging.getLogger("ClusterAwareParallelTrainer")
        self.logger.setLevel(logging.INFO)
        
        # 客户端模型字典
        self.client_models = {}
        
        # 聚类和设备映射
        self.cluster_map = {}
        self.device_map = {}
    
    def register_client_model(self, client_id, model):
        """注册客户端模型"""
        self.client_models[client_id] = model
    
    def register_client_models(self, client_models_dict):
        """批量注册客户端模型"""
        self.client_models.update(client_models_dict)
    
    def setup_training(self, cluster_map, device_map=None):
        """设置训练环境"""
        self.cluster_map = cluster_map
        
        # 如果没有设备映射，创建一个简单的映射
        if device_map is None:
            device_map = {}
            
            # 检查可用的GPU数量
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i, cluster_id in enumerate(cluster_map.keys()):
                    # 循环分配GPU
                    if gpu_count > 0:
                        device_map[cluster_id] = f"cuda:{i % gpu_count}"
                    else:
                        device_map[cluster_id] = "cpu"
            else:
                # 如果没有GPU，所有聚类都用CPU
                for cluster_id in cluster_map.keys():
                    device_map[cluster_id] = "cpu"
        
        self.device_map = device_map
        self.logger.info(f"训练设置完成，聚类数量: {len(cluster_map)}")
        for cluster_id, clients in cluster_map.items():
            self.logger.info(f"聚类 {cluster_id}: {len(clients)} 个客户端，设备: {device_map.get(cluster_id, 'default')}")
    
    
    # # 添加跨客户端评估方法
    # def _evaluate_cross_client(self, client_id, client_model, server_model, global_classifier):
    #     """评估客户端模型在其他客户端数据上的表现"""
    #     # 确保使用同一个设备 - 使用固定设备cuda:0或者默认设备
    #     # 获取当前可用的第一个设备，避免设备混合使用
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda:0')  # 固定使用第一个GPU
    #     else:
    #         device = torch.device('cpu')
        
    #     # 确保所有模型都在同一设备上
    #     client_model = client_model.to(device)
    #     server_model = server_model.to(device)
    #     global_classifier = global_classifier.to(device)
        
    #     # 设置为评估模式
    #     client_model.eval()
    #     server_model.eval()
    #     global_classifier.eval()
        
    #     all_results = []
        
    #     # 获取当前客户端
    #     current_client = self.client_manager.get_client(client_id)
    #     if current_client is None:
    #         return None
        
    #     # 获取其他客户端(最多取3个)
    #     other_clients = []
    #     for other_id, client in self.client_manager.clients.items():
    #         if other_id != client_id:
    #             other_clients.append(client)
    #             if len(other_clients) >= 3:
    #                 break
        
    #     if not other_clients:
    #         return None
        
    #     # 评估每个其他客户端的数据
    #     for other_client in other_clients:
    #         test_data = other_client.test_data
            
    #         # 准备统计
    #         correct = 0
    #         total = 0
    #         loss_sum = 0.0
    #         batch_count = 0
            
    #         with torch.no_grad():
    #             for data, target in test_data:
    #                 # 确保数据在正确的设备上
    #                 data, target = data.to(device), target.to(device)
                    
    #                 # 前向传播
    #                 try:
    #                     # 使用新的模型接口，处理三个返回值
    #                     local_logits, shared_features, _ = client_model(data)
                        
    #                     # 确保特征在正确设备上
    #                     shared_features = shared_features.to(device)
                        
    #                     # 服务器处理
    #                     server_features = server_model(shared_features)
                        
    #                     # 确保特征在正确设备上
    #                     server_features = server_features.to(device)
                        
    #                     # 全局分类
    #                     global_logits = global_classifier(server_features)
                        
    #                     # 计算损失
    #                     loss = F.cross_entropy(global_logits, target)
    #                     loss_sum += loss.item()
    #                     batch_count += 1
                        
    #                     # 计算准确率
    #                     _, predicted = global_logits.max(1)
    #                     total += target.size(0)
    #                     correct += predicted.eq(target).sum().item()
    #                 except Exception as e:
    #                     self.logger.error(f"跨客户端评估错误: {str(e)}")
    #                     # 打印更多调试信息
    #                     if torch.cuda.is_available():
    #                         self.logger.error(f"当前设备: {device}, 模型设备: client={next(client_model.parameters()).device}, server={next(server_model.parameters()).device}, classifier={next(global_classifier.parameters()).device}")
    #                     continue
                    
    #                 # 限制评估批次数量，节省时间
    #                 if batch_count >= 5:
    #                     break
            
    #         # 计算该客户端数据的结果
    #         if total > 0:
    #             accuracy = 100.0 * correct / total
    #             avg_loss = loss_sum / max(1, batch_count)
                
    #             all_results.append({
    #                 'other_client_id': other_client.client_id,
    #                 'accuracy': accuracy,
    #                 'loss': avg_loss
    #             })
        
    #     # 计算平均结果
    #     if all_results:
    #         avg_accuracy = sum(r['accuracy'] for r in all_results) / len(all_results)
    #         avg_loss = sum(r['loss'] for r in all_results) / len(all_results)
            
    #         return {
    #             'accuracy': avg_accuracy,
    #             'loss': avg_loss,
    #             'details': all_results
    #         }
        
    #     return None
    
    # 在utils/tierhfl_trainer.py中
    def execute_parallel_training(self, round_idx=0):
        """执行两阶段训练：客户端并行，全局分类器串行"""
        start_time = time.time()
        
        # 阶段一：客户端模型和服务器模型并行训练
        # 这里不训练全局分类器
        train_results, eval_results, server_models, shared_states, time_stats = self._execute_parallel_client_training(round_idx)
        
        # 阶段二：全局分类器串行训练
        # 选择单一设备进行全局分类器训练
        global_training_time = self._train_global_classifier(train_results, round_idx)
        
        # 阶段二评估：评估全局分类器性能
        global_eval_results = self._eval_global_classifier(eval_results)
        
        # 更新评估结果，合并全局分类器的评估结果
        for client_id, global_result in global_eval_results.items():
            if client_id in eval_results:
                eval_results[client_id].update(global_result)
        
        total_time = time.time() - start_time
        self.logger.info(f"两阶段训练完成，总耗时: {total_time:.2f}秒")
        
        return train_results, eval_results, server_models, shared_states, time_stats, total_time

    def _execute_parallel_client_training(self, round_idx=0):
        """阶段一：仅训练客户端模型和服务器模型"""
        start_time = time.time()
        
        # 没有聚类映射时返回空结果
        if not self.cluster_map:
            self.logger.warning("没有设置聚类映射，无法执行训练")
            return {}, {}, {}, {}, {}
        
        # 创建结果队列
        results_queue = queue.Queue()
        
        # 创建线程
        threads = []
        for cluster_id, client_ids in self.cluster_map.items():
            thread = threading.Thread(
                target=self._train_cluster_phase1,
                args=(cluster_id, client_ids, round_idx, results_queue)
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
        server_models = {}
        shared_states = {}
        time_stats = {}
        
        while not results_queue.empty():
            result = results_queue.get()
            
            # 检查是否有错误
            if 'error' in result:
                self.logger.error(f"聚类 {result['cluster_id']} 返回错误: {result['error']}")
                continue
            
            # 保存结果
            cluster_id = result['cluster_id']
            if 'server_model' in result:
                server_models[cluster_id] = result['server_model']
            
            # 收集客户端状态和特征
            if 'client_features' in result:
                for client_id, features in result['client_features'].items():
                    train_results[client_id] = train_results.get(client_id, {})
                    train_results[client_id]['features'] = features
            
            # 收集共享层状态
            if 'shared_states' in result:
                for client_id, shared_state in result['shared_states'].items():
                    shared_states[client_id] = shared_state
            
            # 合并训练结果
            if 'train_results' in result:
                for client_id, client_result in result['train_results'].items():
                    train_results[client_id] = train_results.get(client_id, {})
                    train_results[client_id].update(client_result)
            
            # 合并评估结果
            if 'eval_results' in result:
                for client_id, eval_result in result['eval_results'].items():
                    eval_results[client_id] = eval_result
                
            # 合并时间统计
            if 'time_stats' in result:
                for client_id, client_time in result['time_stats'].items():
                    time_stats[client_id] = client_time
        
        training_time = time.time() - start_time
        self.logger.info(f"阶段一并行训练完成，耗时: {training_time:.2f}秒")
        
        return train_results, eval_results, server_models, shared_states, time_stats

    def _train_global_classifier(self, train_results, round_idx=0):
        """阶段二：在单一设备上串行训练全局分类器"""
        start_time = time.time()
        
        # 选择单一设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"开始在设备 {device} 上训练全局分类器")
        
        # 确保模型在正确设备上
        self.global_classifier = self.global_classifier.to(device)
        
        # 使用Adam优化器，略微增大学习率以加速收敛
        optimizer = torch.optim.Adam(self.global_classifier.parameters(), lr=0.002)
        
        # 统计信息
        stats = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 收集所有客户端的特征和标签
        all_features = []
        all_targets = []
        
        # 从训练结果中提取保存的特征和标签
        for client_id, result in train_results.items():
            if 'features' in result:
                client_features = result['features']
                for batch_features, batch_targets in client_features:
                    # 确保特征和标签在正确设备上
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    all_features.append((batch_features, batch_targets))
        
        # 如果没有足够数据，则跳过训练
        if len(all_features) == 0:
            self.logger.warning("没有可用于训练全局分类器的特征数据")
            return 0.0
        
        # 创建数据集
        batch_size = 64
        
        # 动态调整训练轮次 - 早期多训练几轮，后期减少轮次
        if round_idx < 10:
            num_epochs = 5  # 早期多训练，建立良好基础
        elif round_idx < 30:
            num_epochs = 3  # 中期保持适度训练
        else:
            num_epochs = 2  # 后期减少轮次，避免过拟合
        
        # 训练循环
        for epoch in range(num_epochs):
            # 随机打乱数据，每轮使用不同顺序
            random.shuffle(all_features)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_features, batch_targets in all_features:
                # 前向传播
                optimizer.zero_grad()
                logits = self.global_classifier(batch_features)
                loss = F.cross_entropy(logits, batch_targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 更新统计
                epoch_loss += loss.item()
                stats['batch_count'] += 1
                
                # 计算准确率
                _, preds = logits.max(1)
                epoch_total += batch_targets.size(0)
                epoch_correct += preds.eq(batch_targets).sum().item()
            
            # 每轮结束后记录性能
            epoch_accuracy = 100.0 * epoch_correct / max(1, epoch_total)
            self.logger.info(f"全局分类器训练 - 轮次 {epoch+1}/{num_epochs}, 损失: {epoch_loss/len(all_features):.4f}, 准确率: {epoch_accuracy:.2f}%")
        
        # 计算平均损失和准确率
        if stats['batch_count'] > 0:
            avg_loss = stats['loss'] / stats['batch_count']
            accuracy = 100.0 * stats['correct'] / max(1, stats['total'])
            self.logger.info(f"全局分类器训练完成。总损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")
        
        # 确保全局分类器回到CPU
        self.global_classifier = self.global_classifier.cpu()
        
        training_time = time.time() - start_time
        return training_time

    def _train_cluster_phase1(self, cluster_id, client_ids, round_idx, results_queue):
        """第一阶段：训练客户端和服务器模型，同时保存特征用于第二阶段"""
        try:
            # 获取当前聚类的设备
            device = self.device_map.get(cluster_id, self.default_device)
            self.logger.info(f"聚类 {cluster_id} 开始阶段一训练，设备: {device}")
            
            # 计时
            model_load_start = time.time()
            
            # 将服务器模型移到正确设备
            server_model = copy.deepcopy(self.server_model).to(device)
            
            # 创建损失函数
            loss_fn = None
            if hasattr(self, 'loss_fn'):
                loss_fn = copy.deepcopy(self.loss_fn).to(device)
            
            model_load_time = time.time() - model_load_start
            
            # 记录结果
            cluster_results = {}
            cluster_eval_results = {}
            cluster_time_stats = {}
            shared_states = {}
            client_features = {}  # 保存客户端特征用于第二阶段
            
            total_rounds = getattr(self, 'rounds', 100)
            
            # 对每个客户端执行训练
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if client is None or client_id not in self.client_models:
                    self.logger.warning(f"客户端 {client_id} 不存在或没有模型，跳过")
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                
                # 临时设置客户端设备和模型
                original_device = client.device
                client.device = device
                client.model = client_model
                
                # 执行阶段一训练，保存特征
                try:
                    # 训练客户端和服务器，同时收集特征
                    train_stats, shared_state, server_state, features_data = self._train_phase1(
                        client, client_model, server_model, loss_fn, round_idx, total_rounds)
                    
                    cluster_results[client_id] = train_stats
                    shared_states[client_id] = shared_state
                    client_features[client_id] = features_data
                    
                    # 记录时间统计
                    cluster_time_stats[client_id] = {
                        "training_time": train_stats.get('training_time', 0),
                        "total_time": train_stats.get('training_time', 0) + 0.1
                    }
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 阶段一训练失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                # 评估客户端
                try:
                    # 这里只评估客户端和服务器，不使用全局分类器
                    eval_result = self._eval_phase1(client, client_model, server_model)
                    cluster_eval_results[client_id] = eval_result
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 评估失败: {str(e)}")
                
                # 恢复客户端设备
                client.device = original_device
                
                # 更新本地模型
                self.client_models[client_id] = client_model.cpu()
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 聚合该聚类的服务器模型，这里暂不处理全局分类器
            agg_server_model = self._aggregate_states(
                {client_id: server_state for client_id, server_state in client_features.items() if 'server_state' in server_state}
            )
                
            # 将结果添加到队列
            results_queue.put({
                'cluster_id': cluster_id,
                'server_model': agg_server_model,
                'train_results': cluster_results,
                'eval_results': cluster_eval_results,
                'shared_states': shared_states,
                'client_features': client_features,  # 包含用于第二阶段的特征
                'time_stats': cluster_time_stats,
                'model_load_time': model_load_time
            })
            
            self.logger.info(f"聚类 {cluster_id} 阶段一训练完成")
            
        except Exception as e:
            import traceback
            error_msg = f"聚类 {cluster_id} 阶段一训练失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            # 报告错误
            results_queue.put({
                'cluster_id': cluster_id,
                'error': error_msg
            })

    def _train_phase1(self, client, client_model, server_model, loss_fn, round_idx, total_rounds):
        """客户端阶段一训练：训练客户端和服务器模型，同时收集特征"""
        # 确保模型在训练模式
        client_model.train()
        server_model.train()
        
        device = client.device
        
        # 获取不同类型的参数
        shared_params = []
        personalized_params = []
        
        for name, param in client_model.named_parameters():
            if 'shared_base' in name:
                shared_params.append(param)
            else:
                personalized_params.append(param)
        
        # 差异化学习率设置
        base_lr = client.lr
        progress_factor = round_idx / max(1, total_rounds)
        
        # 学习率因子计算
        server_lr_factor = 1.5 - 0.7 * progress_factor
        shared_lr_factor = 1.0 - 0.5 * progress_factor
        personal_lr_factor = 0.8 + 0.4 * progress_factor
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(shared_params, lr=base_lr * shared_lr_factor)
        personalized_optimizer = torch.optim.Adam(personalized_params, lr=base_lr * personal_lr_factor)
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=base_lr * server_lr_factor)
        
        # 统计信息
        stats = {
            'total_loss': 0.0,
            'local_loss': 0.0,
            'feature_loss': 0.0,
            'correct_local': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 动态平衡因子
        alpha = 0.3 + 0.4 * (round_idx / total_rounds)
        
        # 保存特征数据用于第二阶段
        features_data = []
        
        # 训练循环
        for epoch in range(client.local_epochs):
            for batch_idx, (data, target) in enumerate(client.train_data):
                # 移到设备
                data, target = data.to(device), target.to(device)
                
                # 清除所有梯度
                shared_optimizer.zero_grad()
                personalized_optimizer.zero_grad()
                server_optimizer.zero_grad()
                
                # 前向传播
                local_logits, shared_features, personal_features = client_model(data)
                server_features = server_model(shared_features)
                
                # 计算损失
                local_loss = F.cross_entropy(local_logits, target)
                
                # 动态特征对齐
                if hasattr(loss_fn, 'dynamic_feature_alignment'):
                    feature_loss = loss_fn.dynamic_feature_alignment(
                        personal_features, server_features, round_idx, total_rounds)
                else:
                    # 简单对齐损失
                    feature_loss = 1.0 - F.cosine_similarity(
                        F.adaptive_avg_pool2d(personal_features, (1, 1)).flatten(1),
                        server_features, dim=1).mean()
                    feature_weight = max(0.05, 0.3 - 0.25 * (round_idx / total_rounds))
                    feature_loss = feature_loss * feature_weight
                
                # 第一阶段总损失 - 没有全局损失
                total_loss = alpha * local_loss + feature_loss
                
                # 反向传播
                total_loss.backward()
                
                # 更新参数
                shared_optimizer.step()
                personalized_optimizer.step()
                server_optimizer.step()
                
                # 更新统计
                stats['total_loss'] += total_loss.item()
                stats['local_loss'] += local_loss.item()
                stats['feature_loss'] += feature_loss.item()
                stats['batch_count'] += 1
                
                # 计算本地准确率
                _, local_pred = local_logits.max(1)
                stats['total'] += target.size(0)
                stats['correct_local'] += local_pred.eq(target).sum().item()
                
                # 保存特征和标签用于第二阶段
                # 注意：我们克隆并分离特征以避免内存泄漏
                features_data.append((
                    server_features.clone().detach(),
                    target.clone().detach()
                ))
        
        # 计算平均值
        train_stats = {
            'loss': stats['total_loss'] / max(1, stats['batch_count']),
            'local_loss': stats['local_loss'] / max(1, stats['batch_count']),
            'feature_loss': stats['feature_loss'] / max(1, stats['batch_count']),
            'local_accuracy': 100.0 * stats['correct_local'] / max(1, stats['total']),
            'training_time': time.time() - start_time
        }
        
        # 获取共享层状态
        shared_state = {}
        for name, param in client_model.named_parameters():
            if 'shared_base' in name:
                shared_state[name] = param.data.clone()
        
        # 第一阶段结果：不包含全局分类器状态，但包含保存的特征用于第二阶段
        return train_stats, shared_state, server_model.state_dict(), features_data

    def _eval_phase1(self, client, client_model, server_model):
        """阶段一评估：只评估本地模型，不使用全局分类器"""
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        
        device = client.device
        
        # 统计信息
        local_correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in client.test_data:
                # 移到设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                local_logits, _, _ = client_model(data)
                
                # 计算损失
                loss = F.cross_entropy(local_logits, target)
                test_loss += loss.item()
                
                # 计算本地准确率
                _, local_pred = local_logits.max(1)
                local_correct += local_pred.eq(target).sum().item()
                total += target.size(0)
        
        # 计算平均损失和准确率
        test_loader_len = len(client.test_data)
        avg_loss = test_loss / max(1, test_loader_len)
        local_accuracy = 100.0 * local_correct / max(1, total)
        
        return {
            'test_loss': avg_loss,
            'local_accuracy': local_accuracy,
            # 注意：此时不包含全局准确率，将在第二阶段评估
        }
        
    def _eval_global_classifier(self, eval_results):
        """评估全局分类器在各客户端测试数据上的性能"""
        self.logger.info("开始评估全局分类器性能...")
        
        # 选择在CPU上评估，以便后续在不同GPU上使用
        device = torch.device('cpu')
        self.global_classifier = self.global_classifier.to(device)
        self.server_model = self.server_model.to(device)
        
        # 设置为评估模式
        self.global_classifier.eval()
        self.server_model.eval()
        
        # 收集评估结果
        global_eval_results = {}
        
        for client_id in self.client_models:
            client = self.client_manager.get_client(client_id)
            if client is None:
                continue
                
            # 将客户端模型移到评估设备
            client_model = self.client_models[client_id].to(device)
            client_model.eval()
            
            # 统计信息
            stats = {
                'global_loss': 0.0,
                'global_correct': 0,
                'total': 0,
                'batch_count': 0
            }
            
            with torch.no_grad():
                for data, target in client.test_data:
                    # 移到设备
                    data, target = data.to(device), target.to(device)
                    
                    # 前向传播
                    _, shared_features, _ = client_model(data)
                    server_features = self.server_model(shared_features)
                    global_logits = self.global_classifier(server_features)
                    
                    # 计算全局损失
                    global_loss = F.cross_entropy(global_logits, target)
                    stats['global_loss'] += global_loss.item()
                    stats['batch_count'] += 1
                    
                    # 计算全局准确率
                    _, global_pred = global_logits.max(1)
                    stats['total'] += target.size(0)
                    stats['global_correct'] += global_pred.eq(target).sum().item()
            
            # 计算平均损失和准确率
            test_loader_len = len(client.test_data)
            global_loss = stats['global_loss'] / max(1, stats['batch_count'])
            global_accuracy = 100.0 * stats['global_correct'] / max(1, stats['total'])
            
            # 更新评估结果
            if client_id in eval_results:
                eval_results[client_id]['global_loss'] = global_loss
                eval_results[client_id]['global_accuracy'] = global_accuracy
            else:
                eval_results[client_id] = {
                    'global_loss': global_loss,
                    'global_accuracy': global_accuracy
                }
            
            global_eval_results[client_id] = {
                'global_loss': global_loss,
                'global_accuracy': global_accuracy
            }
        
        # 计算平均性能
        avg_global_acc = np.mean([res['global_accuracy'] for res in global_eval_results.values()])
        self.logger.info(f"全局分类器平均准确率: {avg_global_acc:.2f}%")
        
        # 将模型移回CPU
        self.global_classifier = self.global_classifier.cpu()
        self.server_model = self.server_model.cpu()
        
        return global_eval_results

    def diagnose_client6_features(self, round_idx=0):
        """针对客户端6提取并保存中间特征，用于问题诊断"""
        # 只有在客户端6存在于集群中时才进行诊断
        client6_exists = False
        for cluster_id, clients in self.cluster_map.items():
            if 6 in clients:
                client6_exists = True
                client6_cluster = cluster_id
                break
        
        if not client6_exists:
            self.logger.warning("客户端6不存在，无法诊断")
            return
        
        self.logger.info(f"开始诊断客户端6 (聚类 {client6_cluster})...")
        
        # 获取客户端6和其模型
        client6 = self.client_manager.get_client(6)
        if client6 is None or 6 not in self.client_models:
            self.logger.warning("无法获取客户端6或其模型")
            return
            
        client6_model = copy.deepcopy(self.client_models[6]).to(self.default_device)
        server_model = copy.deepcopy(self.server_model).to(self.default_device)
        global_classifier = copy.deepcopy(self.global_classifier).to(self.default_device)
        
        # 设置为评估模式
        client6_model.eval()
        server_model.eval()
        global_classifier.eval()
        
        # 创建一个特征提取函数
        def extract_features(data_loader, n_batches=5):
            features_dict = {
                'client_inputs': [],
                'client_features': [],
                'server_features': [],
                'global_logits': [],
                'targets': []
            }
            
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(data_loader):
                    if i >= n_batches:
                        break
                        
                    inputs = inputs.to(self.default_device)
                    targets = targets.to(self.default_device)
                    
                    # 客户端前向传播
                    local_logits, client_features = client6_model(inputs)
                    
                    # 服务器前向传播
                    try:
                        server_features = server_model(client_features, tier=client6.tier)
                        global_logits = global_classifier(server_features)
                    except Exception as e:
                        self.logger.error(f"特征提取错误: {str(e)}")
                        continue
                    
                    # 保存特征
                    features_dict['client_inputs'].append(inputs.cpu().numpy())
                    features_dict['client_features'].append(client_features.cpu().numpy())
                    features_dict['server_features'].append(server_features.cpu().numpy())
                    features_dict['global_logits'].append(global_logits.cpu().numpy())
                    features_dict['targets'].append(targets.cpu().numpy())
            
            # 合并批次数据
            for k in features_dict:
                if features_dict[k]:
                    features_dict[k] = np.concatenate(features_dict[k], axis=0)
            
            return features_dict
        
        # 提取训练和测试集特征
        try:
            import numpy as np
            
            self.logger.info("提取客户端6测试集特征...")
            test_features = extract_features(client6.test_data)
            
            # 保存诊断信息
            import os
            os.makedirs('diagnostics', exist_ok=True)
            
            # 保存特征到numpy文件
            np.savez_compressed(
                f'diagnostics/client6_features_round{round_idx}.npz',
                **test_features
            )
            
            self.logger.info(f"客户端6特征已保存到 diagnostics/client6_features_round{round_idx}.npz")
            
            # 打印一些统计信息
            if len(test_features['global_logits']) > 0:
                preds = np.argmax(test_features['global_logits'], axis=1)
                unique_preds = np.unique(preds)
                
                self.logger.info(f"全局分类器预测统计:")
                self.logger.info(f"- 样本数量: {len(preds)}")
                self.logger.info(f"- 不同类别数量: {len(unique_preds)}")
                self.logger.info(f"- 预测的类别: {unique_preds}")
                
                if len(unique_preds) == 1:
                    self.logger.warning(f"警告: 全局分类器对所有样本预测相同类别 ({unique_preds[0]})")
                    
                    # 分析服务器特征
                    server_feat = test_features['server_features']
                    self.logger.info(f"服务器特征统计:")
                    self.logger.info(f"- 均值: {np.mean(server_feat):.4f}")
                    self.logger.info(f"- 标准差: {np.std(server_feat):.4f}")
                    self.logger.info(f"- 最小值: {np.min(server_feat):.4f}")
                    self.logger.info(f"- 最大值: {np.max(server_feat):.4f}")
                    self.logger.info(f"- 零值比例: {np.mean(np.abs(server_feat) < 1e-5)*100:.2f}%")
        except Exception as e:
            self.logger.error(f"诊断过程出错: {str(e)}")
            import traceback
            traceback.print_exc()

# 自适应训练控制器 - 动态调整训练参数
class AdaptiveTrainingController:
    def __init__(self, initial_alpha=0.5, initial_lambda=0.1):
        """初始化控制器"""
        self.alpha = initial_alpha  # 个性化与全局平衡因子
        self.lambda_feature = initial_lambda  # 特征对齐损失权重
        
        # 历史性能记录
        self.history = {
            'local_accuracy': [],
            'global_accuracy': [],
            'global_imbalance': []
        }
    
    def update_history(self, eval_results):
        """更新历史记录"""
        if not eval_results:
            return
            
        local_accs = []
        global_accs = []
        imbalances = []
        
        for result in eval_results.values():
            if 'local_accuracy' in result:
                local_accs.append(result['local_accuracy'])
            if 'global_accuracy' in result:
                global_accs.append(result['global_accuracy'])
            if 'global_imbalance' in result:
                imbalances.append(result['global_imbalance'])
        
        # 计算平均值
        if local_accs:
            self.history['local_accuracy'].append(sum(local_accs) / len(local_accs))
        if global_accs:
            self.history['global_accuracy'].append(sum(global_accs) / len(global_accs))
        if imbalances:
            # 过滤掉infinity
            valid_imbalances = [i for i in imbalances if i != float('inf')]
            if valid_imbalances:
                self.history['global_imbalance'].append(sum(valid_imbalances) / len(valid_imbalances))
    
    def adjust_parameters(self):
        """调整训练参数，更加动态的策略"""
        if len(self.history['local_accuracy']) < 3:
            return {'alpha': self.alpha, 'lambda_feature': self.lambda_feature}
        
        # 获取最近几轮的性能指标
        window_size = min(5, len(self.history['local_accuracy']))
        recent_local_acc = self.history['local_accuracy'][-window_size:]
        recent_global_acc = self.history['global_accuracy'][-window_size:]
        
        # 计算趋势
        local_trend = recent_local_acc[-1] - recent_local_acc[-3]
        global_trend = recent_global_acc[-1] - recent_global_acc[-3]
        
        # 当前性能
        current_local_acc = recent_local_acc[-1]
        current_global_acc = recent_global_acc[-1]
        
        # 计算不平衡度趋势
        if len(self.history['global_imbalance']) >= 3:
            recent_imbalance = self.history['global_imbalance'][-3:]
            imbalance_trend = recent_imbalance[-1] - recent_imbalance[0]
        else:
            imbalance_trend = 0
        
        # 新策略：根据性能差距和趋势调整alpha
        acc_gap = current_local_acc - current_global_acc
        
        

        # 调整alpha - 个性化与全局平衡，更加倾向全局性能
        if global_trend < -1.0 and local_trend > 0:
            # 全局性能下降但本地性能上升，适度增加个性化权重
            self.alpha = min(0.6, self.alpha + 0.03)  # 限制最大值并减小增量
        elif global_trend > 0.5 or (global_trend > 0 and local_trend < 0):
            # 更积极地降低alpha以促进全局学习
            self.alpha = max(0.1, self.alpha - 0.05)

        # 增强特征对齐的调整
        if global_trend < 0 or imbalance_trend > 0.2:
            # 当全局性能下降或不平衡度增加时，增强特征对齐
            self.lambda_feature = min(0.8, self.lambda_feature + 0.1)
        elif global_trend > 2.0 and imbalance_trend < 0:
            # 全局性能显著上升且不平衡度下降，适当减弱特征对齐
            self.lambda_feature = max(0.2, self.lambda_feature - 0.05)
        
        
        return {
            'alpha': self.alpha,
            'lambda_feature': self.lambda_feature
        }
        
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
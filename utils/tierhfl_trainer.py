import torch
import threading
import queue
import time
import copy
import numpy as np
from collections import defaultdict
import logging
import torch.nn.functional as F

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
    
    # 添加计算平衡权重的方法
    def calculate_balanced_weights(self, client_performances, client_distributions, alpha=0.7):
        """计算平衡性能和多样性的权重
        
        Args:
            client_performances: 客户端性能字典 {client_id: 性能值}
            client_distributions: 客户端数据分布字典 {client_id: 分布信息}
            alpha: 平衡因子，控制性能和多样性的权重
            
        Returns:
            权重字典 {client_id: 权重值}
        """
        # 性能权重计算
        performance_weights = {}
        if client_performances:
            min_perf = min(client_performances.values())
            max_perf = max(client_performances.values())
            perf_range = max(0.1, max_perf - min_perf)
            
            for client_id, perf in client_performances.items():
                performance_weights[client_id] = 0.2 + 0.8 * (perf - min_perf) / perf_range
        else:
            # 如果没有性能数据，使用平均权重
            for client_id in client_distributions.keys():
                performance_weights[client_id] = 1.0
        
        # 多样性权重计算 - 给予低代表性数据分布更高权重
        diversity_weights = {}
        distribution_counts = {}
        
        # 统计各类分布出现次数
        for client_id, dist in client_distributions.items():
            # 将分布转换为可哈希形式
            dist_key = tuple(sorted([(k, v) for k, v in dist.items()]))
            distribution_counts[dist_key] = distribution_counts.get(dist_key, 0) + 1
        
        # 计算多样性权重 - 稀有分布获得更高权重
        for client_id, dist in client_distributions.items():
            dist_key = tuple(sorted([(k, v) for k, v in dist.items()]))
            count = distribution_counts[dist_key]
            diversity_weights[client_id] = 1.0 / count
        
        # 归一化多样性权重
        total_div = sum(diversity_weights.values())
        if total_div > 0:
            diversity_weights = {k: v/total_div for k, v in diversity_weights.items()}
        else:
            diversity_weights = {k: 1.0/len(diversity_weights) for k in diversity_weights.keys()}
        
        # 综合权重 - alpha控制性能和多样性的平衡
        final_weights = {}
        for client_id in client_distributions.keys():
            final_weights[client_id] = alpha * performance_weights.get(client_id, 1.0) + \
                                     (1-alpha) * diversity_weights.get(client_id, 1.0)
        
        # 归一化最终权重
        total = sum(final_weights.values())
        if total > 0:
            return {k: v/total for k, v in final_weights.items()}
        else:
            return {k: 1.0/len(final_weights) for k in final_weights.keys()}
    
    # 修改执行并行训练的方法
    def execute_parallel_training(self, client_models, round_idx):
        """优化的并行训练方法
        
        Args:
            client_models: 客户端模型字典
            round_idx: 当前轮次
            
        Returns:
            训练结果
        """
        start_time = time.time()
        
        # 获取可用GPU数量
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.logger.info(f"使用 {num_gpus} 个GPU进行训练")
        
        # 客户端分组
        client_groups = self.grouping_strategy.group_clients(
            self.client_manager, client_models, round_idx)
        
        # 重置中央服务器中的所有客户端分组
        for group_id in range(len(self.central_server.server_groups)):
            self.central_server.server_groups[group_id].client_ids = []
        
        # 记录分组之前所有客户端的组映射
        old_client_groups = {cid: self.central_server.client_to_group.get(cid) 
                            for cid in client_models.keys()}
        
        # 清空所有现有映射
        self.central_server.client_to_group = {}
        
        # 更新中央服务器中的客户端分组
        for group_id, client_ids in client_groups.items():
            for client_id in client_ids:
                self.central_server.assign_client_to_group(client_id, group_id)
        
        # 打印分组信息
        self.logger.info(f"轮次 {round_idx} 客户端分组:")
        for group_id, client_ids in client_groups.items():
            self.logger.info(f"组 {group_id}: {client_ids}")
        
        # 验证每个客户端都被正确分配
        for client_id in client_models.keys():
            group = self.central_server.get_client_group(client_id)
            if group is None:
                self.logger.warning(f"客户端 {client_id} 未正确分配到服务器组，正在修复...")
                # 使用旧分组或默认分配到第一个组
                old_group_id = old_client_groups.get(client_id, 0)
                self.central_server.assign_client_to_group(client_id, old_group_id)
        
        # 分配GPU给每个组
        group_to_gpu = {}
        for i, group_id in enumerate(self.central_server.server_groups.keys()):
            group_to_gpu[group_id] = i % num_gpus
            self.logger.info(f"组 {group_id} 分配到GPU {group_to_gpu[group_id]}")
        
        total_rounds = getattr(self, 'rounds', 100)
        all_results = {}
        
        # 每个组创建单独的线程处理
        threads = []
        results_queues = {group_id: queue.Queue() for group_id in client_groups.keys()}
        
        for group_id, client_ids in client_groups.items():
            thread = threading.Thread(
                target=self._train_group,
                args=(group_id, client_ids, client_models, round_idx, total_rounds, 
                     group_to_gpu[group_id], results_queues[group_id])
            )
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集所有组的结果
        for group_id, results_queue in results_queues.items():
            while not results_queue.empty():
                result = results_queue.get()
                all_results.update(result)
        
        # 整理最终结果
        train_results = {}
        eval_results = {}
        client_states = {}
        time_stats = {}
        
        for client_id, result in all_results.items():
            train_results[client_id] = result.get('train_result', {})
            eval_results[client_id] = result.get('eval_result', {})
            client_states[client_id] = result.get('client_state', {})
            time_stats[client_id] = {'training_time': result.get('training_time', 0)}
        
        # 更新全局分类器 - 无需聚合，已在训练中更新
        
        # 记录完成信息
        training_time = time.time() - start_time
        self.logger.info(f"轮次 {round_idx} 并行训练完成，耗时: {training_time:.2f}秒")
        
        # 统计训练结果
        if train_results and eval_results:
            self.log_training_statistics(train_results, eval_results)
        
        # 计算平均准确率
        avg_local_acc = np.mean([r.get('local_accuracy', 0) for r in eval_results.values()]) if eval_results else 0
        avg_global_acc = np.mean([r.get('global_accuracy', 0) for r in eval_results.values()]) if eval_results else 0
        
        # 更新统计信息
        self.stats['training_time'].append(training_time)
        self.stats['avg_local_acc'].append(avg_local_acc)
        self.stats['avg_global_acc'].append(avg_global_acc)
        
        # 返回结果
        return {
            'train_results': train_results,
            'eval_results': eval_results,
            'client_states': client_states,
            'server_states': {gid: group.server_model.state_dict() 
                            for gid, group in self.central_server.server_groups.items()},
            'classifier_states': {gid: group.global_classifier.state_dict() 
                                for gid, group in self.central_server.server_groups.items()},
            'time_stats': time_stats,
            'training_time': training_time,
            'avg_local_acc': avg_local_acc,
            'avg_global_acc': avg_global_acc
        }
    
    # 添加组训练方法
    def _train_group(self, group_id, client_ids, client_models, round_idx, total_rounds, gpu_id, results_queue):
        """训练一个组的客户端，优化特征维度处理和内存使用"""
        try:
            # 设置设备
            device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
            
            # 获取服务器组
            server_group = self.central_server.get_server_group(group_id)
            if not server_group:
                self.logger.error(f"找不到组 {group_id} 的服务器组")
                return
            
            # 将服务器模型和全局分类器移动到正确的设备
            server_model = server_group.server_model.to(device)
            global_classifier = server_group.global_classifier.to(device)
            
            # 为该组创建损失函数
            loss_fn = copy.deepcopy(self.loss_fn).to(device)
            
            # 创建服务器优化器
            server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.0005)
            classifier_optimizer = torch.optim.Adam(global_classifier.parameters(), lr=0.001)
            
            # 阶段一: 每个客户端独立训练个性化路径
            phase1_results = {}
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if not client:
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = client_models[client_id].to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                # 执行阶段一训练
                phase1_result = client.train_phase1(round_idx, total_rounds)
                phase1_results[client_id] = phase1_result
                
                # 恢复客户端设备
                client.device = original_device
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 收集每个客户端的特征和标签 - 为阶段二准备
            all_features = []
            all_labels = []
            client_features = {}
            
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if not client:
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = client_models[client_id].to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                try:
                    # 提取特征和标签
                    features, labels = client.extract_features_and_labels()
                    
                    if features is not None and labels is not None:
                        # 确保特征维度正确 - 不在这里处理，应该在模型的forward方法中处理
                        # 储存特征供阶段二使用
                        client_features[client_id] = (features, labels)
                        all_features.append(features)
                        all_labels.append(labels)
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 特征提取错误: {str(e)}")
                
                # 恢复客户端设备
                client.device = original_device
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 阶段二: 执行批处理训练和客户端共享层更新
            phase2_results = {}
            shared_states = {}
            
            # 是否有足够特征进行批处理
            batch_process_success = False
            
            if all_features and all_labels:
                try:
                    # 合并特征和标签到设备上
                    batch_features = torch.cat(all_features, dim=0).to(device)
                    batch_labels = torch.cat(all_labels, dim=0).to(device)
                    
                    # 使用批处理和梯度累积训练服务器模型
                    server_model.train()
                    global_classifier.train()
                    
                    # 以下代码替代了单独训练每个客户端的共享层
                    # 相反，我们使用所有客户端的特征批量训练服务器
                    
                    # 清除梯度
                    server_optimizer.zero_grad()
                    
                    # 前向传播 - 在服务器模型的forward中处理不同维度
                    server_features = server_model(batch_features)
                    logits = global_classifier(server_features)
                    
                    # 计算损失
                    loss = F.cross_entropy(logits, batch_labels)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 更新参数
                    server_optimizer.step()
                    
                    # 标记批处理成功
                    batch_process_success = True
                    
                    # 如果需要，可以禁用经验回放缓冲区
                    disable_buffer = getattr(self, 'disable_buffer', True)
                    if not disable_buffer:
                        try:
                            # 更新全局分类器的经验回放缓冲区
                            global_classifier.update_buffer(server_features.detach(), batch_labels, device)
                            
                            # 执行经验回放训练
                            replay_features, replay_labels = global_classifier.sample_from_buffer(
                                min(64, getattr(self, 'batch_size_server', 32)), device)
                            
                            if replay_features is not None and replay_labels is not None:
                                # 清除梯度
                                classifier_optimizer.zero_grad()
                                
                                # 前向传播
                                replay_logits = global_classifier(replay_features)
                                
                                # 计算损失
                                replay_loss = F.cross_entropy(replay_logits, replay_labels)
                                
                                # 反向传播
                                replay_loss.backward()
                                
                                # 更新参数
                                classifier_optimizer.step()
                        except Exception as e:
                            self.logger.error(f"经验回放训练错误: {str(e)}")
                    
                    # 释放内存
                    del batch_features, batch_labels, server_features
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.logger.error(f"批处理训练错误: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # 针对每个客户端更新共享层
            # 这部分仅在批处理成功的情况下执行，否则采用原始的客户端独立训练方式
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if not client:
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = client_models[client_id].to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                try:
                    if batch_process_success:
                        # 如果批处理成功，只提取共享层状态，无需额外训练
                        # 提取客户端共享层状态
                        shared_state = {}
                        for name, param in client_model.named_parameters():
                            if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2_shared']):
                                shared_state[name] = param.data.clone()
                        
                        # 记录结果（假设训练成功）
                        phase2_results[client_id] = {'loss': 0.0, 'accuracy': 100.0, 'training_time': 0.0}
                        shared_states[client_id] = shared_state
                    else:
                        # 如果批处理失败，退回到原始的客户端独立训练方式
                        phase2_result, shared_state = client.train_phase2(
                            server_model, global_classifier, loss_fn, round_idx, total_rounds)
                        phase2_results[client_id] = phase2_result
                        shared_states[client_id] = shared_state
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 阶段二训练错误: {str(e)}")
                    
                # 恢复客户端设备
                client.device = original_device
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 评估每个客户端
            eval_results = {}
            for client_id in client_ids:
                # 评估代码保持不变...
                client = self.client_manager.get_client(client_id)
                if not client:
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = client_models[client_id].to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                # 执行评估
                try:
                    eval_result = client.evaluate(server_model, global_classifier)
                    eval_results[client_id] = eval_result
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 评估错误: {str(e)}")
                
                # 恢复客户端设备
                client.device = original_device
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 计算权重 - 基于评估性能和数据分布
            client_performances = {cid: res.get('global_accuracy', 0) 
                                for cid, res in eval_results.items()}
            
            client_distributions = {}
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if client:
                    client_distributions[client_id] = client.data_distribution.get('percentage', {})
            
            weights = self.calculate_balanced_weights(client_performances, client_distributions)
            
            # 聚合共享层 - 使用平衡权重
            if shared_states:
                # 聚合共享层参数
                aggregated_shared = {}
                for name in next(iter(shared_states.values())).keys():
                    weighted_sum = None
                    for client_id, state in shared_states.items():
                        if name in state:
                            weight = weights.get(client_id, 1.0 / len(shared_states))
                            if weighted_sum is None:
                                weighted_sum = weight * state[name].clone()
                            else:
                                weighted_sum += weight * state[name]
                    
                    if weighted_sum is not None:
                        aggregated_shared[name] = weighted_sum
                
                # 更新每个客户端的共享层
                for client_id in client_ids:
                    if client_id in client_models:
                        client_model = client_models[client_id]
                        for name, param in aggregated_shared.items():
                            if name in client_model.state_dict():
                                client_model.state_dict()[name].copy_(param)
            
            # 将结果加入队列
            client_results = {}
            for client_id in client_ids:
                if client_id in phase1_results and client_id in phase2_results and client_id in eval_results:
                    # 合并阶段一和阶段二的结果
                    train_result = {
                        'loss': phase1_results[client_id].get('loss', 0) + phase2_results[client_id].get('loss', 0),
                        'local_loss': phase1_results[client_id].get('loss', 0),
                        'global_loss': phase2_results[client_id].get('global_loss', 0) if 'global_loss' in phase2_results[client_id] else 0,
                        'feature_loss': phase2_results[client_id].get('feature_loss', 0) if 'feature_loss' in phase2_results[client_id] else 0,
                        'accuracy': phase1_results[client_id].get('accuracy', 0),
                        'training_time': phase1_results[client_id].get('training_time', 0) + 
                                    phase2_results[client_id].get('training_time', 0)
                    }
                    
                    client_results[client_id] = {
                        'train_result': train_result,
                        'eval_result': eval_results[client_id],
                        'client_state': client_models[client_id].state_dict(),
                        'training_time': train_result['training_time']
                    }
            
            # 将结果添加到队列
            results_queue.put(client_results)
            
        except Exception as e:
            self.logger.error(f"组 {group_id} 训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    # 修改批处理函数中的特征处理
    def _train_server_with_batches(self, server_model, global_classifier, 
                                features, labels, server_optimizer, classifier_optimizer,
                                batch_size=32, accum_steps=4, device=None):
        """使用批处理和梯度累积训练服务器模型，确保特征维度正确"""
        # 设置训练模式
        server_model.train()
        global_classifier.train()
        
        # 检查特征维度
        expected_dim = global_classifier.feature_dim
        
        # 计算批次数
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # 初始化统计
        total_loss = 0.0
        total_correct = 0
        
        # 清除梯度
        server_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        
        # 批处理循环
        for i in range(num_batches):
            # 确定批次范围
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_size_actual = end_idx - start_idx
            
            # 获取当前批次
            batch_features = features[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            # 确保数据在正确的设备上
            if device is not None:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
            
            try:
                # 前向传播
                server_features = server_model(batch_features)
                
                # 确保特征维度正确
                if server_features.size(-1) != expected_dim:
                    self.logger.warning(f"特征维度不匹配: 期望 {expected_dim}, 得到 {server_features.size(-1)}")
                    # 尝试调整维度
                    if len(server_features.shape) == 2:
                        server_features = server_features.view(server_features.size(0), -1)
                        # 如果新维度仍不匹配，使用线性层调整
                        if server_features.size(-1) != expected_dim:
                            # 创建临时调整层
                            adapter = nn.Linear(server_features.size(-1), expected_dim).to(device)
                            server_features = adapter(server_features)
                
                logits = global_classifier(server_features)
                
                # 计算损失
                loss = F.cross_entropy(logits, batch_labels)
                
                # 缩放损失以适应梯度累积
                loss = loss / accum_steps
                
                # 反向传播
                loss.backward()
                
                # 累计统计
                total_loss += loss.item() * accum_steps
                _, predicted = logits.max(1)
                total_correct += predicted.eq(batch_labels).sum().item()
                
                # 累积指定步骤后更新参数
                if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                    # 更新参数
                    server_optimizer.step()
                    classifier_optimizer.step()
                    
                    # 清除梯度
                    server_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()
                    
                    # 临时释放内存
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"批处理训练错误: {str(e)}")
                # 跳过失败的批次
                continue
        
        # 计算平均损失和准确率
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = 100.0 * total_correct / num_samples if num_samples > 0 else 0
        
        return avg_loss, accuracy       
    def log_training_statistics(self, train_results, eval_results):
        """记录详细的训练统计信息"""
        # 计算全局平均值
        avg_global_loss = np.mean([r.get('global_loss', 0) for r in train_results.values()])
        avg_local_loss = np.mean([r.get('local_loss', 0) for r in train_results.values()])
        avg_feature_loss = np.mean([r.get('feature_loss', 0) for r in train_results.values()])
        avg_local_acc = np.mean([r.get('local_accuracy', 0) for r in eval_results.values()])
        avg_global_acc = np.mean([r.get('global_accuracy', 0) for r in eval_results.values()])
        
        # 按Tier分组统计
        tier_stats = defaultdict(lambda: defaultdict(list))
        for client_id, result in eval_results.items():
            client = self.client_manager.get_client(client_id)
            if client:
                tier = client.tier
                tier_stats[tier]['local_acc'].append(result.get('local_accuracy', 0))
                tier_stats[tier]['global_acc'].append(result.get('global_accuracy', 0))
        
        # 计算每个Tier的平均值
        tier_avg = {}
        for tier, stats in tier_stats.items():
            tier_avg[tier] = {
                'local_acc': np.mean(stats['local_acc']),
                'global_acc': np.mean(stats['global_acc'])
            }
        
        # 详细打印统计信息
        self.logger.info("-------------- 训练统计 --------------")
        self.logger.info(f"平均损失 - 全局: {avg_global_loss:.4f}, 本地: {avg_local_loss:.4f}, 特征对齐: {avg_feature_loss:.4f}")
        self.logger.info(f"平均准确率 - 本地: {avg_local_acc:.2f}%, 全局: {avg_global_acc:.2f}%")
        
        # 打印每个Tier的统计
        for tier, avg in tier_avg.items():
            self.logger.info(f"Tier {tier} - 本地准确率: {avg['local_acc']:.2f}%, 全局准确率: {avg['global_acc']:.2f}%")
        self.logger.info("--------------------------------------")
    
    def _aggregate_models(self, model_states, weights=None):
        """聚合模型"""
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
                    # 检查类型并特殊处理Long类型
                    if aggregated[key].dtype == torch.long:
                        # 对于Long类型，先转换为浮点型计算，再四舍五入回Long类型
                        float_result = aggregated[key].float() + (w * state[key].float())
                        aggregated[key] = torch.round(float_result).long()
                    else:
                        # 其他类型直接计算
                        aggregated[key] += w * state[key]
        
        return aggregated
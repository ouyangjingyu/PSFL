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
    
    def _train_client(self, client_id, client_model, server_model, global_classifier, 
                round_idx, results_queue, gpu_id=None, total_rounds=100,
                loss_fn=None, gradient_guide=None):
        """修复的客户端训练函数，接受组专用的loss_fn"""
        try:
            # 指定GPU设备
            if gpu_id is not None and torch.cuda.is_available():
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cpu')
                
            # 确保模型在正确的设备上
            client_model = client_model.to(device)
            server_model = server_model.to(device)
            global_classifier = global_classifier.to(device)
            
            # 使用传入的loss_fn或默认的loss_fn
            if loss_fn is None:
                loss_fn = self.loss_fn
            # 确保loss_fn在正确的设备上
            loss_fn = loss_fn.to(device)
            
            # 使用传入的gradient_guide或默认的gradient_guide
            if gradient_guide is None:
                gradient_guide = self.gradient_guide
            
            # 获取客户端
            client = self.client_manager.get_client(client_id)
            if client is None:
                self.logger.warning(f"客户端 {client_id} 不存在，跳过")
                return
                
            # 记录开始时间
            start_time = time.time()
            
            # 临时设置客户端设备
            original_device = client.device
            client.device = device
            
            # 训练客户端，使用组专用的loss_fn
            train_result, model_state = client.train(
                server_model, global_classifier, loss_fn, 
                gradient_guide, round_idx, total_rounds)
            
            # 评估客户端
            eval_result = client.evaluate(server_model, global_classifier)
            
            # 恢复客户端设备设置
            client.device = original_device
            
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
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 报告错误
            results_queue.put({
                'client_id': client_id,
                'error': str(e)
            })
            
    def execute_parallel_training(self, client_models, round_idx):
        """优化的并行训练方法，按组分配GPU"""
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
        
        # 修改：按组逐个处理，而不是所有组并行，避免GPU冲突
        all_results = {}
        total_rounds = getattr(self, 'rounds', 100)

        # 分配GPU给每个组
        group_to_gpu = {}
        for i, group_id in enumerate(self.central_server.server_groups.keys()):
            group_to_gpu[group_id] = i % num_gpus
            self.logger.info(f"组 {group_id} 分配到GPU {group_to_gpu[group_id]}")
        
        # 每个组使用专用的loss_fn实例，确保在正确设备上
        group_loss_fns = {}
        # 按组处理
        for group_id, client_ids in client_groups.items():
            self.logger.info(f"开始处理组 {group_id} 的 {len(client_ids)} 个客户端")
            
            group = self.central_server.get_server_group(group_id)
            if not group:
                continue
                
            # 获取分配的GPU
            gpu_id = group_to_gpu[group_id]
            device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_str)
        
            # 为该组创建专用的loss_fn和gradient_guide实例
            group_loss_fn = copy.deepcopy(self.loss_fn).to(device)
            group_gradient_guide = copy.deepcopy(self.gradient_guide)
            group_loss_fns[group_id] = group_loss_fn
            
            # 创建结果队列
            results_queue = queue.Queue()
            
            # 为每个客户端创建服务器模型和分类器的副本
            server_model_templates = {}
            global_classifier_templates = {}
            
            # 预先创建所有模型副本
            for client_id in client_ids:
                if client_id not in client_models:
                    continue
                    
                # 为每个客户端创建独立副本并移到正确的设备上
                server_model_templates[client_id] = copy.deepcopy(group.server_model).to(device)
                global_classifier_templates[client_id] = copy.deepcopy(group.global_classifier).to(device)
            
            # 为组内每个客户端创建线程
            threads = []
            for client_id in client_ids:
                if client_id not in client_models:
                    continue
                    
                thread = threading.Thread(
                    target=self._train_client,
                    args=(client_id, client_models[client_id], 
                        server_model_templates[client_id], 
                        global_classifier_templates[client_id], 
                        round_idx, results_queue, gpu_id, total_rounds,
                        group_loss_fn, group_gradient_guide)  # 传递组专用的loss_fn
                )
                threads.append(thread)
            
            # 启动该组的所有线程
            for thread in threads:
                thread.start()
            
            # 等待该组的所有线程完成
            for thread in threads:
                thread.join()
            
            # 收集当前组的结果
            group_results = {}
            while not results_queue.empty():
                result = results_queue.get()
                if 'error' in result:
                    self.logger.error(f"客户端 {result['client_id']} 返回错误: {result['error']}")
                else:
                    group_results[result['client_id']] = result
            
            # 将结果保存到总结果字典
            all_results.update(group_results)
            
            # 聚合该组的模型
            if group_results:
                # 收集所有模型状态
                server_states = [server_model_templates[cid].state_dict() 
                                for cid in group_results.keys()]
                classifier_states = [global_classifier_templates[cid].state_dict() 
                                for cid in group_results.keys()]
                client_states = {cid: res['model_state'] for cid, res in group_results.items()}
                
                # 计算权重
                weights = {}
                for client_id, result in group_results.items():
                    accuracy = result['eval_result']['global_accuracy']
                    weights[client_id] = max(0.1, accuracy / 100.0)
                
                # 归一化权重
                total_weight = sum(weights.values())
                if total_weight > 0:
                    normalized_weights = {k: v/total_weight for k, v in weights.items()}
                else:
                    normalized_weights = {k: 1.0/len(weights) for k in weights.keys()}
                
                # 聚合服务器模型 (如果有结果)
                if server_states:
                    # 聚合服务器模型
                    weight_list = [0.3] + [0.7 * normalized_weights.get(cid, 1.0/len(client_states)) 
                                        for cid in group_results.keys()]
                    aggregated_server = self._aggregate_models(
                        [group.server_model.state_dict()] + server_states,
                        weight_list
                    )
                    
                    # 更新原始服务器模型
                    group.server_model.load_state_dict(aggregated_server)
                
                # 聚合全局分类器 (如果有结果)
                if classifier_states:
                    # 简单平均聚合分类器
                    aggregated_classifier = self._aggregate_models(
                        classifier_states, 
                        [1.0/len(classifier_states)] * len(classifier_states)
                    )
                    
                    # 更新原始分类器
                    group.global_classifier.load_state_dict(aggregated_classifier)
            
            # 清理模型副本，释放内存
            del server_model_templates
            del global_classifier_templates
            torch.cuda.empty_cache()
        
        # 整理最终结果
        train_results = {}
        eval_results = {}
        client_states = {}
        time_stats = {}
        
        for client_id, result in all_results.items():
            train_results[client_id] = result['train_result']
            eval_results[client_id] = result['eval_result']
            client_states[client_id] = result['model_state']
            time_stats[client_id] = {'training_time': result['training_time']}
        
        # 更新客户端模型
        for client_id, state in client_states.items():
            # 提取共享基础层参数
            shared_base_params = {}
            for key, value in state.items():
                if any(name in key for name in ['conv1', 'bn1', 'layer1', 'layer2', 'global_adapter']):
                    shared_base_params[key] = value
                
            # 更新客户端模型
            model = client_models[client_id]
            for key, value in shared_base_params.items():
                if key in model.state_dict():
                    model.state_dict()[key].copy_(value)
        
        # 记录完成信息
        training_time = time.time() - start_time
        self.logger.info(f"轮次 {round_idx} 并行训练完成，耗时: {training_time:.2f}秒")
        
        # 统计训练结果
        if train_results and eval_results:
            self.log_training_statistics(train_results, eval_results)
        
        # 计算平均准确率
        avg_local_acc = np.mean([r['local_accuracy'] for r in eval_results.values()]) if eval_results else 0
        avg_global_acc = np.mean([r['global_accuracy'] for r in eval_results.values()]) if eval_results else 0
        
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
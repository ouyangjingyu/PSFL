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
    
    def _train_cluster(self, cluster_id, client_ids, round_idx, results_queue):
        """训练单个聚类的工作函数 - 支持双阶段训练"""
        try:
            # 获取当前聚类的设备
            device = self.device_map.get(cluster_id, self.default_device)
            self.logger.info(f"聚类 {cluster_id} 开始训练，设备: {device}")
            
            # 计时
            model_load_start = time.time()
            
            # 将服务器模型和全局分类器移到正确设备
            server_model = copy.deepcopy(self.server_model).to(device)
            global_classifier = copy.deepcopy(self.global_classifier).to(device) if self.global_classifier else None
            
            # 创建损失函数
            loss_fn = None
            if hasattr(self, 'loss_fn'):
                loss_fn = copy.deepcopy(self.loss_fn).to(device)
            
            model_load_time = time.time() - model_load_start
            
            # 记录结果
            cluster_results = {}
            phase1_results = {}
            phase2_results = {}
            shared_states = {}
            cluster_eval_results = {}
            cluster_time_stats = {}
            
            total_rounds = getattr(self, 'rounds', 100)
            
            # 阶段一: 每个客户端独立训练个性化路径
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if client is None or client_id not in self.client_models:
                    self.logger.warning(f"客户端 {client_id} 不存在或没有模型，跳过")
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                # 执行阶段一训练
                try:
                    phase1_result = client.train_phase1(round_idx, total_rounds)
                    phase1_results[client_id] = phase1_result
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 阶段一训练失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                # 恢复客户端设备
                client.device = original_device
                
                # 更新本地模型
                self.client_models[client_id] = client_model.cpu()
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 阶段二: 执行共享层和服务器训练
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if client is None or client_id not in self.client_models:
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                # 执行阶段二训练
                try:
                    phase2_result, shared_state = client.train_phase2(
                        server_model, global_classifier, loss_fn, round_idx, total_rounds)
                    phase2_results[client_id] = phase2_result
                    shared_states[client_id] = shared_state
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 阶段二训练失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                # 恢复客户端设备
                client.device = original_device
                
                # 更新本地模型
                self.client_models[client_id] = client_model.cpu()
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 评估每个客户端
            eval_results = {}
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if client is None or client_id not in self.client_models:
                    continue
                
                # 将客户端模型移到正确的设备
                client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                
                # 临时设置客户端设备
                original_device = client.device
                client.device = device
                client.model = client_model
                
                # 执行评估
                try:
                    # 标准评估
                    eval_result = client.evaluate(client_model, server_model, global_classifier)  # 添加global_classifier参数
                    eval_results[client_id] = eval_result
                    
                    # 添加跨客户端评估
                    if hasattr(self, '_evaluate_cross_client'):
                        cross_result = self._evaluate_cross_client(
                            client_id, client_model, server_model, global_classifier)
                        
                        if cross_result:
                            eval_result['cross_client_accuracy'] = cross_result.get('accuracy', 0)
                            eval_result['cross_client_loss'] = cross_result.get('loss', 0)
                    
                    eval_results[client_id] = eval_result
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 评估失败: {str(e)}")
                    
                # 恢复客户端设备
                client.device = original_device
                
                # 释放内存
                torch.cuda.empty_cache()
            
            # 合并结果
            for client_id in client_ids:
                if client_id in phase1_results and client_id in phase2_results and client_id in eval_results:
                    # 合并阶段一和阶段二结果
                    train_result = {
                        'phase1_loss': phase1_results[client_id].get('loss', 0),
                        'phase2_loss': phase2_results[client_id].get('loss', 0),
                        'global_loss': phase2_results[client_id].get('global_loss', 0),
                        'feature_loss': phase2_results[client_id].get('feature_loss', 0),
                        'phase1_accuracy': phase1_results[client_id].get('accuracy', 0),
                        'phase2_accuracy': phase2_results[client_id].get('accuracy', 0),
                        'training_time': phase1_results[client_id].get('training_time', 0) + 
                                    phase2_results[client_id].get('training_time', 0)
                    }
                    
                    cluster_results[client_id] = train_result
                    cluster_eval_results[client_id] = eval_results[client_id]
                    cluster_time_stats[client_id] = {
                        "training_time": train_result['training_time'],
                        "total_time": train_result['training_time'] + 0.1  # 加0.1秒作为其他开销
                    }
            
            # 将结果添加到队列
            results_queue.put({
                'cluster_id': cluster_id,
                'server_model': server_model.cpu().state_dict(),
                'global_classifier': global_classifier.cpu().state_dict() if global_classifier else None,
                'train_results': cluster_results,
                'eval_results': cluster_eval_results,
                'shared_states': shared_states,  # 添加共享层状态
                'time_stats': cluster_time_stats,
                'model_load_time': model_load_time
            })
            
            self.logger.info(f"聚类 {cluster_id} 训练完成")
            
        except Exception as e:
            import traceback
            error_msg = f"聚类 {cluster_id} 训练失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            # 报告错误
            results_queue.put({
                'cluster_id': cluster_id,
                'error': error_msg
            })

    # 添加跨客户端评估方法
    def _evaluate_cross_client(self, client_id, client_model, server_model, global_classifier):
        """评估客户端模型在其他客户端数据上的表现"""
        # 确保使用同一个设备 - 使用固定设备cuda:0或者默认设备
        # 获取当前可用的第一个设备，避免设备混合使用
        if torch.cuda.is_available():
            device = torch.device('cuda:0')  # 固定使用第一个GPU
        else:
            device = torch.device('cpu')
        
        # 确保所有模型都在同一设备上
        client_model = client_model.to(device)
        server_model = server_model.to(device)
        global_classifier = global_classifier.to(device)
        
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        global_classifier.eval()
        
        all_results = []
        
        # 获取当前客户端
        current_client = self.client_manager.get_client(client_id)
        if current_client is None:
            return None
        
        # 获取其他客户端(最多取3个)
        other_clients = []
        for other_id, client in self.client_manager.clients.items():
            if other_id != client_id:
                other_clients.append(client)
                if len(other_clients) >= 3:
                    break
        
        if not other_clients:
            return None
        
        # 评估每个其他客户端的数据
        for other_client in other_clients:
            test_data = other_client.test_data
            
            # 准备统计
            correct = 0
            total = 0
            loss_sum = 0.0
            batch_count = 0
            
            with torch.no_grad():
                for data, target in test_data:
                    # 确保数据在正确的设备上
                    data, target = data.to(device), target.to(device)
                    
                    # 前向传播
                    try:
                        # 使用新的模型接口，处理三个返回值
                        local_logits, shared_features, _ = client_model(data)
                        
                        # 确保特征在正确设备上
                        shared_features = shared_features.to(device)
                        
                        # 服务器处理
                        server_features = server_model(shared_features)
                        
                        # 确保特征在正确设备上
                        server_features = server_features.to(device)
                        
                        # 全局分类
                        global_logits = global_classifier(server_features)
                        
                        # 计算损失
                        loss = F.cross_entropy(global_logits, target)
                        loss_sum += loss.item()
                        batch_count += 1
                        
                        # 计算准确率
                        _, predicted = global_logits.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                    except Exception as e:
                        self.logger.error(f"跨客户端评估错误: {str(e)}")
                        # 打印更多调试信息
                        if torch.cuda.is_available():
                            self.logger.error(f"当前设备: {device}, 模型设备: client={next(client_model.parameters()).device}, server={next(server_model.parameters()).device}, classifier={next(global_classifier.parameters()).device}")
                        continue
                    
                    # 限制评估批次数量，节省时间
                    if batch_count >= 5:
                        break
            
            # 计算该客户端数据的结果
            if total > 0:
                accuracy = 100.0 * correct / total
                avg_loss = loss_sum / max(1, batch_count)
                
                all_results.append({
                    'other_client_id': other_client.client_id,
                    'accuracy': accuracy,
                    'loss': avg_loss
                })
        
        # 计算平均结果
        if all_results:
            avg_accuracy = sum(r['accuracy'] for r in all_results) / len(all_results)
            avg_loss = sum(r['loss'] for r in all_results) / len(all_results)
            
            return {
                'accuracy': avg_accuracy,
                'loss': avg_loss,
                'details': all_results
            }
        
        return None
    
    def execute_parallel_training(self, round_idx=0):
        """执行并行训练"""
        start_time = time.time()
        
        # 没有聚类映射时返回空结果
        if not self.cluster_map:
            self.logger.warning("没有设置聚类映射，无法执行训练")
            return {}, {}, {}, {}, {}, 0
        
        # 创建结果队列
        results_queue = queue.Queue()
        
        # 创建线程
        threads = []
        for cluster_id, client_ids in self.cluster_map.items():
            thread = threading.Thread(
                target=self._train_cluster,
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
        global_classifier_states = {}  # 全局分类器状态
        shared_states = {}  # 新增：用于收集共享状态
        time_stats = {}  # 新增时间统计收集
        
        while not results_queue.empty():
            result = results_queue.get()
            
            # 检查是否有错误
            if 'error' in result:
                self.logger.error(f"聚类 {result['cluster_id']} 返回错误: {result['error']}")
                continue
            
            # 保存结果
            cluster_id = result['cluster_id']
            server_models[cluster_id] = result['server_model']
            if 'global_classifier' in result and result['global_classifier']:
                global_classifier_states[cluster_id] = result['global_classifier']

            # 新增：收集共享层状态
            if 'shared_states' in result:
                for client_id, shared_state in result['shared_states'].items():
                    shared_states[client_id] = shared_state
            
            # 合并训练结果
            for client_id, client_result in result['train_results'].items():
                train_results[client_id] = client_result
            
            # 合并评估结果
            for client_id, eval_result in result['eval_results'].items():
                eval_results[client_id] = eval_result
                
            # 合并时间统计
            if 'time_stats' in result:
                for client_id, client_time in result['time_stats'].items():
                    time_stats[client_id] = client_time
                
                # 添加模型加载时间
                if 'model_load_time' in result:
                    for client_id in result['time_stats'].keys():
                        time_stats[client_id]['model_load_time'] = result['model_load_time']
        
        training_time = time.time() - start_time
        self.logger.info(f"并行训练完成，耗时: {training_time:.2f}秒")
        
        return train_results, eval_results, server_models, global_classifier_states, shared_states, time_stats, training_time
        
    def _aggregate_classifiers(self, classifier_states, weights):
        """聚合多个全局分类器状态"""
        if not classifier_states:
            return {}
        
        # 获取第一个状态的键
        keys = next(iter(classifier_states.values())).keys()
        
        # 初始化结果
        result = {}
        
        # 对每个参数进行加权平均
        for key in keys:
            # 初始化累加器
            weighted_sum = None
            total_weight = 0.0
            
            # 加权累加
            for cluster_id, state in classifier_states.items():
                if key in state:
                    weight = weights.get(cluster_id, 0.0)
                    total_weight += weight
                    
                    # 累加
                    if weighted_sum is None:
                        weighted_sum = weight * state[key].clone().to(self.default_device)
                    else:
                        weighted_sum += weight * state[key].clone().to(self.default_device)
            
            # 计算平均值
            if weighted_sum is not None and total_weight > 0:
                result[key] = weighted_sum / total_weight
        
        return result

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
        
        # # 调整alpha - 个性化与全局平衡
        # if acc_gap > 5.0:
        #     # 本地模型明显更好，增加个性化权重
        #     self.alpha = min(0.9, self.alpha + 0.1)
        # elif acc_gap < -5.0:
        #     # 全局模型明显更好，增加全局权重
        #     self.alpha = max(0.1, self.alpha - 0.1)
        # elif global_trend < -1.0 and local_trend > 0:
        #     # 全局性能下降但本地性能上升，适当增加个性化权重
        #     self.alpha = min(0.6, self.alpha + 0.03)
        # elif global_trend > 0.5 and local_trend < 0:
        #     # 全局性能上升但本地性能下降，适当增加全局权重
        #     self.alpha = max(0.2, self.alpha - 0.05)

        # # 调整lambda_feature - 特征对齐
        # if imbalance_trend > 0.5 or current_global_acc < 40:
        #     # 不平衡度增加或全局性能差，更强力增强特征对齐
        #     self.lambda_feature = min(0.5, self.lambda_feature + 0.1)
        # elif global_trend < -1.0:
        #     # 全局性能下降，增强特征对齐
        #     self.lambda_feature = min(0.5, self.lambda_feature + 0.05)
        # elif global_trend > 1.0 and imbalance_trend < 0:
        #     # 全局性能上升且不平衡度下降，适当减弱特征对齐
        #     self.lambda_feature = max(0.05, self.lambda_feature - 0.03)

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
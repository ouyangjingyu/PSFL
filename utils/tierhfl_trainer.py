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
    def __init__(self, client_manager, server_model, global_classifier=None, device="cuda", max_workers=None):
        """初始化训练器"""
        self.client_manager = client_manager
        self.server_model = server_model
        self.global_classifier = global_classifier
        self.default_device = device
        
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
        """训练单个聚类的工作函数"""
        try:
            # 获取当前聚类的设备
            device = self.device_map.get(cluster_id, self.default_device)
            self.logger.info(f"聚类 {cluster_id} 开始训练，设备: {device}")
            
            # 计时 - 增加模型加载时间记录
            model_load_start = time.time()
            
            # 将服务器模型和全局分类器移到正确设备
            server_model = copy.deepcopy(self.server_model).to(device)
            global_classifier = copy.deepcopy(self.global_classifier).to(device) if self.global_classifier else None
            
            model_load_time = time.time() - model_load_start
            # 新增：针对全局分类器的预热训练
            if round_idx < 5 and global_classifier is not None:
                self.logger.info(f"聚类 {cluster_id} 开始全局分类器预热训练")
                
                # 为全局分类器创建专用优化器
                classifier_optimizer = torch.optim.Adam(
                    global_classifier.parameters(),
                    lr=0.001,
                    weight_decay=0.0001  # 增加正则化
                )
                
                # 从客户端收集训练批次
                warmup_batches = []
                for client_id in client_ids:
                    client = self.client_manager.get_client(client_id)
                    if client is None or client_id not in self.client_models:
                        continue
                    
                    client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                    
                    # 收集少量批次用于预热训练
                    batch_count = 0
                    for data, target in client.train_data:
                        if batch_count >= 5:  # 每个客户端最多5个批次
                            break
                        warmup_batches.append((client_id, data, target))
                        batch_count += 1
                
                # 预热训练全局分类器
                if warmup_batches:
                    for _ in range(3):  # 多次遍历预热数据
                        random.shuffle(warmup_batches)
                        
                        for client_id, data, target in warmup_batches:
                            data, target = data.to(device), target.to(device)
                            client = self.client_manager.get_client(client_id)
                            client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                            
                            with torch.no_grad():  # 不计算客户端模型梯度
                                _, features = client_model(data)
                                server_features = server_model(features, tier=client.tier)
                            
                            # 只训练全局分类器
                            classifier_optimizer.zero_grad()
                            global_logits = global_classifier(server_features)
                            loss = F.cross_entropy(global_logits, target)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(global_classifier.parameters(), max_norm=1.0)
                            classifier_optimizer.step()
                            
            # 保存训练结果
            cluster_results = {}
            cluster_eval_results = {}
            cluster_time_stats = {}  # 新增时间统计
            
            if round_idx == 0:
                self.logger.info(f"聚类 {cluster_id} 开始特征维度检查")
                # 创建一个样本输入
                sample_input = torch.randn(1, 3, 32, 32).to(device)
                
                for client_id in client_ids:
                    client = self.client_manager.get_client(client_id)
                    if client is None or client_id not in self.client_models:
                        continue
                        
                    tier = client.tier
                    try:
                        client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                        self.logger.info(f"检查客户端 {client_id} (Tier {tier}) 的特征维度")
                        
                        # 执行前向传播
                        local_logits, features = client_model(sample_input)
                        self.logger.info(f"客户端 {client_id} 特征输出维度: {features.shape}, 本地分类器输出维度: {local_logits.shape}")
                        
                        # 检查服务器模型处理
                        server_features = server_model(features, tier=tier)
                        self.logger.info(f"服务器处理后特征维度: {server_features.shape}")
                        
                        # 检查全局分类器
                        if global_classifier:
                            global_logits = global_classifier(server_features)
                            self.logger.info(f"全局分类器输出维度: {global_logits.shape}")
                        
                        # 客户端模型移回CPU
                        client_model = client_model.cpu()
                    except Exception as e:
                        self.logger.error(f"客户端 {client_id} 维度检查失败: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                self.logger.info(f"聚类 {cluster_id} 完成特征维度检查")

            # 检查聚类中是否包含客户端6
            has_client6 = 6 in client_ids
            if has_client6:
                print(f"\n[CLUSTER TRAINER] 聚类 {cluster_id} 包含客户端6")
                print(f"[CLUSTER TRAINER] 聚类 {cluster_id} 所有客户端: {client_ids}")
                
                # 设置调试标志
                if hasattr(server_model, '_debug_client_id'):
                    server_model._debug_client_id = 6
                if global_classifier and hasattr(global_classifier, '_debug_client_id'):
                    global_classifier._debug_client_id = 6

            

            # 训练每个客户端
            for client_id in client_ids:
                client_start_time = time.time()  # 客户端总时间开始
                
                # 获取客户端和模型
                client = self.client_manager.get_client(client_id)
                if client is None or client_id not in self.client_models:
                    self.logger.warning(f"客户端 {client_id} 不存在或没有模型，跳过")
                    continue
                
                # 特殊监控客户端6
                is_client6 = client_id == 6
                if is_client6:
                    print(f"\n[TRAINER] 开始处理客户端6，所属聚类: {cluster_id}, 轮次: {round_idx}")
                    print(f"[TRAINER] 客户端6 Tier: {client.tier}, 学习率: {client.lr}")
                    
                    # 检查模型参数统计
                    client_model = self.client_models[client_id]
                    n_params = sum(p.numel() for p in client_model.parameters())
                    print(f"[TRAINER] 客户端6模型参数数量: {n_params}")
                    
                    # 为客户端模型添加调试标志
                    if hasattr(client_model, 'client_id'):
                        client_model.client_id = 6
                    
                    # 为损失函数添加调试标志
                    if hasattr(client.feature_alignment_loss, '_debug_client_id'):
                        client.feature_alignment_loss._debug_client_id = 6

                # 计时 - 模型复制
                copy_start_time = time.time()
                try:
                    client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 模型复制到设备 {device} 失败: {str(e)}")
                    continue
                copy_time = time.time() - copy_start_time
                
                # 执行训练 - 传入全局分类器
                try:
                    # 修改client.train方法以支持全局分类器
                    train_result = client.train(client_model, server_model, global_classifier, round_idx)
                    cluster_results[client_id] = train_result
                    
                    # 更新客户端模型
                    model_transfer_start = time.time()
                    self.client_models[client_id] = client_model.cpu()
                    model_transfer_time = time.time() - model_transfer_start
                    
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 训练失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
                
                # 执行评估 - 针对客户端6添加更详细日志
                if is_client6:
                    print(f"[TRAINER] 开始评估客户端6...")
                    
                eval_start_time = time.time()

                try:
                    eval_result = client.evaluate(client_model, server_model, global_classifier)
                    cluster_eval_results[client_id] = eval_result
                    
                    if is_client6:
                        print(f"[TRAINER] 客户端6评估结果:")
                        print(f"  - 本地准确率: {eval_result.get('local_accuracy', 0):.2f}%")
                        print(f"  - 全局准确率: {eval_result.get('global_accuracy', 0):.2f}%")
                        print(f"  - 测试损失: {eval_result.get('test_loss', 0):.4f}")
                        
                        # 检查每个类别的准确率
                        if 'global_per_class_acc' in eval_result:
                            print(f"  - 全局每类准确率: {[f'{acc:.1f}%' for acc in eval_result['global_per_class_acc']]}")
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 评估失败: {str(e)}")
                    if is_client6:
                        print(f"[TRAINER] 客户端6评估出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    continue

                eval_time = time.time() - eval_start_time
                
                # 记录时间开销
                client_total_time = time.time() - client_start_time
                cluster_time_stats[client_id] = {
                    "copy_time": copy_time,
                    "training_time": train_result.get('training_time', 0),
                    "model_transfer_time": model_transfer_time,
                    "evaluation_time": eval_time,
                    "total_time": client_total_time
                }
                
            # 将聚类结果添加到队列
            results_queue.put({
                'cluster_id': cluster_id,
                'server_model': server_model.cpu().state_dict(),
                'global_classifier': global_classifier.cpu().state_dict() if global_classifier else None,
                'train_results': cluster_results,
                'eval_results': cluster_eval_results,
                'time_stats': cluster_time_stats,  # 新增时间统计
                'model_load_time': model_load_time  # 记录模型加载时间
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
        
        return train_results, eval_results, server_models, global_classifier_states, time_stats, training_time
        
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
        """优化的参数调整策略，强化个性化能力"""
        if len(self.history['local_accuracy']) < 3:
            return {'alpha': self.alpha, 'lambda_feature': self.lambda_feature}
        
        # 获取性能指标
        current_local_acc = self.history['local_accuracy'][-1]
        current_global_acc = self.history['global_accuracy'][-1]
        
        # 计算本地和全局准确率差距
        acc_gap = current_local_acc - current_global_acc
        
        # 训练轮次
        round_idx = len(self.history['local_accuracy'])
        early_stage = round_idx < 20
        late_stage = round_idx >= 60
        
        # 优化alpha，优先提升本地模型性能
        if current_local_acc < 75:  # 本地准确率较低时
            self.alpha = min(0.8, self.alpha + 0.05)  # 快速增加本地权重
        elif acc_gap < -5:  # 全局模型明显优于本地模型
            self.alpha = min(0.7, self.alpha + 0.03)  # 增加本地权重
        elif current_local_acc >= 85:  # 本地模型已经很好
            self.alpha = max(0.6, min(0.75, self.alpha))  # 稳定在较高区间
        else:  # 默认情况
            self.alpha = min(0.75, self.alpha + 0.02)  # 缓慢增加本地权重
        
        # 特征对齐调整 - 后期减弱约束增强个性化
        if late_stage:
            self.lambda_feature = max(0.05, self.lambda_feature - 0.02)
        
        return {
            'alpha': self.alpha,
            'lambda_feature': self.lambda_feature
        } 

# 数据分布聚类器 - 基于数据特性进行聚类
class DataDistributionClusterer:
    def __init__(self, num_clusters=3):
        """初始化聚类器"""
        self.num_clusters = num_clusters
    
    def cluster_clients(self, client_models, client_ids, eval_dataset=None, device='cuda'):
        """基于客户端评估性能和模型参数聚类"""
        # 防止客户端数小于聚类数
        if len(client_ids) <= self.num_clusters:
            clusters = {i: [client_id] for i, client_id in enumerate(client_ids[:self.num_clusters])}
            self.clustering_history.append({
                'timestamp': time.time(),
                'clusters': copy.deepcopy(clusters),
                'num_clients': len(client_ids)
            })
            return clusters
        
        # 收集客户端模型参数
        client_features = {}
        for client_id in client_ids:
            if client_id in client_models:
                model = client_models[client_id]
                params_vec = []
                # 只提取共享参数
                for name, param in model.named_parameters():
                    if not any(x in name for x in ['fc', 'linear', 'classifier']):
                        params_vec.append(param.detach().cpu().view(-1))
                
                if params_vec:
                    # 连接所有参数
                    client_features[client_id] = torch.cat(params_vec).numpy()
        
        # 如果没有足够的特征数据，使用简单分配
        if len(client_features) < self.num_clusters:
            clusters = {}
            for i in range(self.num_clusters):
                clusters[i] = []
            
            for i, client_id in enumerate(client_ids):
                cluster_idx = i % self.num_clusters
                clusters[cluster_idx].append(client_id)
                
            self.clustering_history.append({
                'timestamp': time.time(),
                'clusters': copy.deepcopy(clusters),
                'num_clients': len(client_ids)
            })
            return clusters
        
        # 使用K-means算法聚类
        try:
            from sklearn.cluster import KMeans
            
            # 准备特征矩阵
            feature_matrix = np.array(list(client_features.values()))
            
            # 标准化特征
            feature_mean = np.mean(feature_matrix, axis=0)
            feature_std = np.std(feature_matrix, axis=0) + 1e-5
            feature_matrix = (feature_matrix - feature_mean) / feature_std
            
            # 执行K-means聚类
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(feature_matrix)
            
            # 获取聚类标签
            cluster_labels = kmeans.labels_
            
            # 构建聚类映射
            clusters = {i: [] for i in range(self.num_clusters)}
            
            # 分配客户端到聚类
            for i, client_id in enumerate(client_features.keys()):
                cluster_idx = cluster_labels[i]
                clusters[cluster_idx].append(client_id)
            
            # 分配没有特征的客户端
            for client_id in client_ids:
                if client_id not in client_features:
                    # 找到最小的聚类
                    min_cluster = min(clusters.items(), key=lambda x: len(x[1]))[0]
                    clusters[min_cluster].append(client_id)
        except:
            # 如果K-means失败，回退到简单分配
            clusters = {}
            for i in range(self.num_clusters):
                clusters[i] = []
            
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
        
class ModelFeatureClusterer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clustering_history = []
    
    def cluster_clients(self, client_models, client_ids, eval_dataset=None, device='cuda'):
        """基于模型特征的聚类方法"""
        clusters = {i: [] for i in range(self.num_clusters)}
        
        print("\n[CLUSTERING] 开始客户端聚类...")
    
        # 检查客户端6是否在列表中
        has_client6 = 6 in client_ids
        if has_client6:
            print("[CLUSTERING] 客户端6将参与聚类")
        
        # 提取模型特征
        client_features = {}
        feature_dims = []  # 记录所有特征向量的维度
        
        # 第一步：提取特征并记录维度
        for client_id in client_ids:
            if client_id in client_models:
                model = client_models[client_id]
                features = []
                
                # 只提取卷积层和归一化层统计特征
                for name, param in model.named_parameters():
                    if ('conv' in name or 'norm' in name) and 'weight' in name:
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

                    # 特别检查客户端6的特征
                    if client_id == 6:
                        print(f"[CLUSTERING] 客户端6特征维度: {len(features_array)}")
                        print(f"[CLUSTERING] 客户端6特征统计: min={features_array.min():.4f}, max={features_array.max():.4f}, mean={features_array.mean():.4f}, std={features_array.std():.4f}")
                        # 检查是否有异常值
                        nan_count = np.isnan(features_array).sum()
                        inf_count = np.isinf(features_array).sum()
                        print(f"[CLUSTERING] 客户端6特征异常值: NaN={nan_count}, Inf={inf_count}")
        
        # 检查所有特征向量的维度是否一致
        if feature_dims and len(set(feature_dims)) > 1:
            # 如果维度不一致，找出最常见的维度
            from collections import Counter
            dim_counter = Counter(feature_dims)
            common_dim = dim_counter.most_common(1)[0][0]
            
            print(f"发现不同维度的特征向量: {dict(dim_counter)}，使用最常见维度: {common_dim}")
            
            # 填充或裁剪特征向量到相同维度
            for client_id in list(client_features.keys()):
                feat = client_features[client_id]
                if len(feat) != common_dim:
                    if len(feat) < common_dim:
                        # 如果特征太短，使用填充（用0填充）
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
                
                # 打印形状信息以调试
                print(f"特征矩阵形状: {features_matrix.shape}")
                
                # 确保特征矩阵是浮点型
                features_matrix = features_matrix.astype(np.float32)
                
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
                
                # 处理没有特征的客户端
                for client_id in client_ids:
                    if client_id not in client_features:
                        # 分配到最小聚类
                        min_cluster = min(clusters.items(), key=lambda x: len(x[1]))[0]
                        clusters[min_cluster].append(client_id)
                        
            except Exception as e:
                print(f"K-means聚类失败: {str(e)}，使用备选方案")
                import traceback
                traceback.print_exc()
                # 回退到备选方案 - 基于tier分组
                tier_groups = {}
                for client_id in client_ids:
                    client = self.client_manager.get_client(client_id) if hasattr(self, 'client_manager') else None
                    tier = getattr(client, 'tier', client_id % 7 + 1)  # 如果无法获取tier，使用client_id模拟
                    
                    if tier not in tier_groups:
                        tier_groups[tier] = []
                    tier_groups[tier].append(client_id)
                
                # 将tier组分配到聚类中
                cluster_idx = 0
                for tier, clients in tier_groups.items():
                    for client_id in clients:
                        clusters[cluster_idx % self.num_clusters].append(client_id)
                        cluster_idx += 1
        else:
            # 备选方案：均匀分配
            for i, client_id in enumerate(client_ids):
                cluster_idx = i % self.num_clusters
                clusters[cluster_idx].append(client_id)
        

        # 执行聚类后检查客户端6的分配
        for cluster_id, clients in clusters.items():
            if 6 in clients:
                print(f"[CLUSTERING] 客户端6被分配到聚类 {cluster_id}")
                print(f"[CLUSTERING] 聚类 {cluster_id} 中的所有客户端: {clients}")
                
                # 计算该聚类与其他聚类的差异
                if len(client_features) > 0 and 6 in client_features:
                    client6_feature = client_features[6]
                    for other_id, other_feature in client_features.items():
                        if other_id == 6 or other_id not in client_ids:
                            continue
                        
                        # 计算特征向量余弦相似度
                        try:
                            # 确保长度一致
                            min_len = min(len(client6_feature), len(other_feature))
                            sim = np.dot(client6_feature[:min_len], other_feature[:min_len]) / (
                                np.linalg.norm(client6_feature[:min_len]) * np.linalg.norm(other_feature[:min_len])
                            )
                            print(f"[CLUSTERING] 客户端6与客户端{other_id}的特征相似度: {sim:.4f}")
                        except Exception as e:
                            print(f"[CLUSTERING] 计算相似度出错: {str(e)}")

        # 记录聚类结果
        self.clustering_history.append({
            'timestamp': time.time(),
            'clusters': copy.deepcopy(clusters),
            'num_clients': len(client_ids)
        })
            
        return clusters
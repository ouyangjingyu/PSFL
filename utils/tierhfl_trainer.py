import torch
import threading
import queue
import time
import copy
import numpy as np
from collections import defaultdict
import logging

# 簇感知并行训练器 - 支持聚类并行训练
class ClusterAwareParallelTrainer:
    def __init__(self, client_manager, server_model, device="cuda", max_workers=None):
        """初始化训练器"""
        self.client_manager = client_manager
        self.server_model = server_model
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
            
            # 将服务器模型移到正确设备
            server_model = copy.deepcopy(self.server_model).to(device)
            model_load_time = time.time() - model_load_start
            
            # 保存训练结果
            cluster_results = {}
            cluster_eval_results = {}
            cluster_time_stats = {}  # 新增时间统计
            
            # 训练每个客户端
            for client_id in client_ids:
                client_start_time = time.time()  # 客户端总时间开始
                
                # 获取客户端和模型
                client = self.client_manager.get_client(client_id)
                if client is None or client_id not in self.client_models:
                    self.logger.warning(f"客户端 {client_id} 不存在或没有模型，跳过")
                    continue
                
                # 计时 - 模型复制
                copy_start_time = time.time()
                try:
                    client_model = copy.deepcopy(self.client_models[client_id]).to(device)
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 模型复制到设备 {device} 失败: {str(e)}")
                    continue
                copy_time = time.time() - copy_start_time
                
                # 执行训练
                try:
                    train_result = client.train(client_model, server_model, round_idx)
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
                
                # 执行评估
                eval_start_time = time.time()
                try:
                    eval_result = client.evaluate(client_model, server_model)
                    cluster_eval_results[client_id] = eval_result
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 评估失败: {str(e)}")
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
            return {}, {}, {}, {}, 0  # 修改返回值，增加时间统计
        
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
        
        return train_results, eval_results, server_models, time_stats, training_time  # 增加时间统计返回
        
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

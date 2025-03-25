import torch
import time
import threading
import queue
import copy
from memory_utils import free_memory, safe_model_copy, safe_to_device

def train_cluster_improved(self, cluster_id, client_ids, train_fn, eval_fn, **kwargs):
    """
    改进的聚类训练函数，优化内存管理
    
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
        # 使用安全复制方法
        original_classifier = kwargs['global_classifier']
        cluster_classifier = safe_model_copy(original_classifier, device)
        # 替换原始分类器
        kwargs['global_classifier'] = cluster_classifier
    
    # 在该聚类中串行训练每个客户端
    for i, client_id in enumerate(client_ids):
        # 在每个客户端训练前清理内存
        free_memory()
        
        # 如果该客户端模型不存在，跳过
        if client_id not in self.client_models:
            print(f"客户端 {client_id} 模型不存在，跳过")
            continue
        
        # 使用安全复制方法
        client_model = safe_model_copy(self.client_models[client_id], device)
        server_model = safe_model_copy(self.server_models[client_id], device)
        
        # 如果存在共享分类器，使用共享分类器替换服务器模型的分类器
        # 注意：在新架构中，服务器可能没有分类器，此处代码保留以兼容旧代码
        if self.shared_classifier is not None:
            if hasattr(server_model, 'classifier'):
                server_classifier = safe_model_copy(self.shared_classifier, device)
                server_model.classifier = server_classifier
            elif hasattr(server_model, 'fc'):
                server_fc = safe_model_copy(self.shared_classifier, device)
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
            # 将模型移回CPU以节省GPU内存
            if isinstance(client_model, torch.nn.Module):
                client_model = client_model.cpu()
            if isinstance(server_model, torch.nn.Module):
                server_model = server_model.cpu()
                
            cluster_result['client_models'][client_id] = client_model
            cluster_result['server_models'][client_id] = server_model
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
                    eval_kwargs['global_classifier'] = safe_to_device(eval_kwargs['global_classifier'], device)
                
                # 评估
                eval_result = eval_fn(
                    client_id=client_id,
                    client_model=client_model,
                    server_model=server_model,
                    device=device,
                    **eval_kwargs
                )
                cluster_result['evaluation_metrics'][client_id] = eval_result
            
            # 记得在每个客户端训练后释放内存
            free_memory()
        
        except Exception as e:
            print(f"训练客户端 {client_id} 时出错: {str(e)}")
            self.client_training_stats[client_id].update({
                'cluster_id': cluster_id,
                'error': str(e),
                'success': False
            })
            
            # 即使出错也继续训练下一个客户端
            free_memory()
            continue
    
    return cluster_result

def worker_improved(self, cluster_id, client_ids, train_fn, eval_fn, kwargs, result_queue):
    """
    改进的工作线程函数，用于并行训练不同聚类
    """
    try:
        # 创建线程专用的kwargs副本，避免跨线程干扰
        thread_kwargs = kwargs.copy()
        print(f"聚类 {cluster_id} 收到的kwargs键: {list(kwargs.keys())}")
        
        # 更多调试
        if 'global_classifier' in thread_kwargs:
            print(f"聚类 {cluster_id} global_classifier类型: {type(thread_kwargs['global_classifier'])}")

        # 使用改进的训练聚类函数
        result = train_cluster_improved(self, cluster_id, client_ids, train_fn, eval_fn, **thread_kwargs)
        result_queue.put((cluster_id, result))
        
        # 训练完成后释放内存
        free_memory()
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"聚类 {cluster_id} 训练时出错: {str(e)}\n详细错误:\n{error_trace}")
        result_queue.put((cluster_id, {'error': str(e), 'traceback': error_trace}))
        
        # 出错也释放内存
        free_memory()

def setup_training_improved(self, cluster_map, max_workers=None):
    """
    改进的训练环境设置函数
    
    Args:
        cluster_map: 聚类映射
        max_workers: 最大并行工作线程数
    """
    # 初始化资源感知调度器
    self.scheduler = ResourceAwareScheduler(self.client_resources, cluster_map)
    
    # 获取优化的训练调度
    schedule = self.scheduler.schedule_training(cluster_map, self.client_models)
    device_map = schedule['device_map']
    
    # 限制最大并行度，防止内存溢出
    # GPU显存有限，如果有多个GPU，考虑每个GPU最多分配一个聚类
    if max_workers is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if max_workers > gpu_count:
            print(f"警告: 并行度 ({max_workers}) 大于GPU数量 ({gpu_count})，可能导致内存问题")
            print(f"自动调整为每个GPU一个聚类（{gpu_count}）")
            max_workers = gpu_count
    
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
    
    # 注入改进的训练方法
    # 注意: 这是猴子补丁，在生产环境中可能需要更优雅的方式
    self.trainer.train_cluster = train_cluster_improved.__get__(self.trainer, type(self.trainer))
    self.trainer._worker = worker_improved.__get__(self.trainer, type(self.trainer))
    
    self.logger.info(f"训练设置完成，聚类数量: {len(optimized_cluster_map)}")
    for cluster_id, clients in optimized_cluster_map.items():
        self.logger.info(f"聚类 {cluster_id}: {len(clients)} 客户端，设备: {device_map[cluster_id]}")

def execute_training_improved(self, train_fn, eval_fn=None, **kwargs):
    """
    改进的训练执行函数
    
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
    
    # 释放内存，确保训练开始前系统资源可用
    free_memory()
    
    # 添加全局分类器到kwargs - 确保使用kwargs['global_classifier']
    if self.shared_classifier is not None:
        # 使用共享分类器的副本而不是原始对象
        kwargs['global_classifier'] = safe_model_copy(self.shared_classifier, 'cpu')
    
    # 执行并行训练
    start_time = time.time()
    self.logger.info("开始并行训练...")
    
    # 使用原始训练方法，因为我们已经注入了改进的训练函数
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
        
        # 收集损失和准确率
        losses = []
        accs = []
        for client_metrics in metrics.values():
            if isinstance(client_metrics, dict):
                if 'loss' in client_metrics:
                    losses.append(client_metrics['loss'])
                if 'accuracy' in client_metrics:
                    accs.append(client_metrics['accuracy'])
        
        avg_loss = np.mean(losses) if losses else 0
        avg_acc = np.mean(accs) if accs else 0
        
        self.logger.info(f"聚类 {cluster_id}: {success_count} 客户端完成训练, 平均损失: {avg_loss:.4f}, 平均准确率: {avg_acc:.2f}%")
    
    total_time = time.time() - start_time
    
    # 最后释放内存
    free_memory()
    
    # 返回训练结果
    return cluster_results, client_stats, total_time
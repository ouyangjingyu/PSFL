import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np

# 导入训练策略
from utils.training_strategy import OptimizedTrainingStrategy, create_training_strategy_for_client

def train_client_with_global_classifier(client_id, client_model, server_model, device, 
                                       client_manager, round_idx, global_classifier=None, 
                                       local_epochs=None, split_rounds=1, **kwargs):
    """使用全局分类器的客户端训练函数 - 用于并行训练框架，适应GroupNorm模型"""
    try:
        # 获取客户端
        client = client_manager.get_client(client_id)
        if client is None:
            return {'error': f"客户端 {client_id} 不存在"}
        
        # 设置设备
        client.device = device
        print(f"客户端 {client_id} 使用设备: {device}")
        
        # 确保全局分类器在正确设备上
        if global_classifier is not None:
            global_classifier = global_classifier.to(device)
            for param in global_classifier.parameters():
                param.data = param.data.to(device)
        
        # 创建针对GroupNorm优化的训练策略
        training_strategy = create_training_strategy_for_client(
            client.tier,
            is_low_resource=(client.resources.get('compute_power', 1.0) < 0.5)
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 第1步：本地训练 - 确保深度复制模型
        client_model_copy = copy.deepcopy(client_model)
        
        # 创建组件优化器
        optimizers = training_strategy.create_component_optimizers(client_model_copy)
        
        # 创建调度器字典
        # schedulers = training_strategy.create_schedulers_dict(optimizers)
        
        # 执行本地训练
        client_state, local_stats = client.local_train_with_optimizers(
            client_model_copy, 
            optimizers=optimizers,
            epochs=local_epochs
        )
        local_train_time = local_stats['time']
        
        # 第2步：拆分学习训练 - 使用已经本地训练过的模型继续拆分学习训练
        # 对于拆分学习，创建同样的组件优化器，但加入全局分类器
        sl_optimizers = training_strategy.create_component_optimizers(
            client_model_copy, 
            global_classifier=global_classifier
        )
        
        client_state_sl, server_state, sl_stats = client.train_split_learning_with_optimizers(
            client_model_copy, 
            copy.deepcopy(server_model),
            optimizers=sl_optimizers,
            global_classifier=global_classifier,
            rounds=split_rounds
        )
        sl_train_time = sl_stats['time']
        
        # 合并结果并计算总训练时间
        total_time = local_train_time + sl_train_time
        
        # 构建结果字典
        merged_stats = {
            'local_train': local_stats,
            'split_learning': sl_stats,
            
            # 全局分类器结果
            'loss': sl_stats['loss'],
            'accuracy': sl_stats['accuracy'],
            
            # 本地分类器的拆分学习结果
            'local_sl_loss': sl_stats.get('local_loss', 0),
            'local_sl_accuracy': sl_stats.get('local_accuracy', 0),
            
            # 全局分类器的拆分学习结果
            'global_sl_loss': sl_stats.get('global_loss', 0), 
            'global_sl_accuracy': sl_stats.get('global_accuracy', 0),
            
            # 本地训练结果
            'local_train_loss': local_stats.get('loss', 0),
            'local_train_accuracy': local_stats.get('accuracy', 0),
            
            'time': total_time,
            'local_train_time': local_train_time,
            'sl_train_time': sl_train_time,
            'data_size': sl_stats.get('data_size', 0),
            'communication_mb': sl_stats.get('data_transmitted_mb', 0),
            'lr': local_stats.get('lr_final', client.learning_rate),
            
            # 保存训练后的模型状态
            'client_model_state': client_state_sl,
            'server_model_state': server_state
        }
        
        # 返回训练结果
        return merged_stats
        
    except Exception as e:
        import traceback
        error_msg = f"客户端 {client_id} 训练失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}


class EnhancedClient:
    """增强的客户端类，支持异构设备环境下的训练与优化"""
    
    def __init__(self, idx, device, lr=0.001, local_epochs=5, dataset_train=None, 
                 dataset_test=None, tier=1, tier_scheduler=None, resources=None):
        """
        初始化客户端
        
        Args:
            idx: 客户端索引
            device: 计算设备
            lr: 学习率
            local_epochs: 本地训练轮数
            dataset_train: 训练数据集
            dataset_test: 测试数据集
            tier: 客户端tier级别
            tier_scheduler: tier调度器
            resources: 资源信息
        """
        self.idx = idx
        self.device = device
        self.learning_rate = lr
        self.local_epochs = local_epochs
        self.train_data = dataset_train
        self.test_data = dataset_test
        self.tier = tier
        self.tier_scheduler = tier_scheduler
        self.resources = resources or {}
        
        # 创建优化的训练策略
        self.training_strategy = create_training_strategy_for_client(
            tier,
            is_low_resource=(self.resources.get('compute_power', 1.0) < 0.5)
        )
        
        # 训练相关参数
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0

        # 训练统计信息
        self.training_stats = {
            'local_train_loss': [],
            'local_train_acc': [],
            'sl_train_loss': [],
            'sl_train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'training_time': []
        }
        
        # 模型参数大小
        self.model_size_mb = 0
    
    def update_tier(self, new_tier):
        """
        更新客户端tier级别
        
        Args:
            new_tier: 新的tier级别
        """
        self.tier = new_tier
        # 更新训练策略
        self.training_strategy = create_training_strategy_for_client(
            new_tier,
            is_low_resource=(self.resources.get('compute_power', 1.0) < 0.5)
        )
    
    def calculate_model_size(self, model):
        """
        计算模型参数大小（MB）
        
        Args:
            model: 要计算大小的模型
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        self.model_size_mb = total_size / (1024 ** 2)
        
        return self.model_size_mb
    
    def pre_train(self, model, pretrain_epochs=5):
        """
        预训练模型，为聚类准备特征
        
        Args:
            model: 客户端模型
            pretrain_epochs: 预训练轮数
            
        Returns:
            model_state_dict: 模型参数
            time_pretrained: 预训练耗时
        """
        model.train()
        time_start = time.time()
        
        # 使用优化的训练策略
        optimizer = self.training_strategy.create_optimizer(model, optimizer_type="Adam")
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        for epoch in range(pretrain_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_data):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(images)
                
                # 处理可能的元组输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 计算损失并反向传播
                labels = labels.to(torch.long)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_size = labels.size(0)
                
                # 更新统计信息
                epoch_loss += loss.item() * batch_size
                epoch_correct += batch_correct
                epoch_samples += batch_size
            
            # 计算每个epoch的平均值
            if epoch_samples > 0:
                epoch_loss /= epoch_samples
                epoch_acc = 100.0 * epoch_correct / epoch_samples
                
                # 更新统计信息
                total_loss += epoch_loss
                total_acc += epoch_acc
                total_samples += 1
        
        # 计算平均损失和准确率
        avg_loss = total_loss / max(1, total_samples)
        avg_acc = total_acc / max(1, total_samples)
        
        # 计算预训练耗时
        time_pretrained = time.time() - time_start
        
        # 计算模型大小
        self.calculate_model_size(model)
        
        print(f"Client {self.idx} - Pre-training completed: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, Time={time_pretrained:.2f}s, Size={self.model_size_mb:.2f}MB")
        
        return model.state_dict(), time_pretrained
    
    def local_train_with_optimizers(self, model, optimizers=None, epochs=None):
        """
        使用组件优化器进行本地训练
        
        Args:
            model: 客户端模型
            optimizers: 组件优化器字典
            epochs: 训练轮数
            
        Returns:
            model_state_dict: 模型参数
            stats: 训练统计信息
        """
        # 设置训练模式
        model.train()
        model = model.to(self.device)
        
        # 使用自定义epochs或默认值
        epochs = epochs or self.local_epochs
        
        # 如果没有提供优化器，使用训练策略创建
        if optimizers is None:
            optimizers = self.training_strategy.create_component_optimizers(model)
        
        # 获取特征提取器和分类器优化器
        feature_optimizer = optimizers.get('feature_extractor')
        classifier_optimizer = optimizers.get('classifier')
        
        # 如果未找到组件优化器，创建单一优化器
        if feature_optimizer is None and classifier_optimizer is None:
            optimizer = self.training_strategy.create_optimizer(model)
            optimizers = {'model': optimizer}
            feature_optimizer = optimizer
            classifier_optimizer = optimizer
        
        # 创建数据加载器
        train_loader = self.train_data
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 记录统计信息
        train_loss = 0
        correct = 0
        total = 0
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 清零梯度
                if feature_optimizer:
                    feature_optimizer.zero_grad()
                if classifier_optimizer:
                    classifier_optimizer.zero_grad()
                
                # 前向传播 - 获取输出和特征
                outputs, _ = model(data)
                
                # 计算损失
                loss = criterion(outputs, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数 - 分别用不同优化器
                if feature_optimizer:
                    feature_optimizer.step()
                if classifier_optimizer:
                    classifier_optimizer.step()
                
                # 更新统计信息
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
            
            # 计算本轮epoch的平均损失和准确率
            epoch_loss /= len(train_loader)
            epoch_acc = 100.0 * epoch_correct / epoch_total
            
            # 打印训练进度
            print(f"Client {self.idx} - Local Epoch {epoch+1}/{epochs}: "
                f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, "
                f"LR={self.learning_rate:.6f}")
            
            # 累计统计信息
            train_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # 计算总训练时间
        train_time = time.time() - start_time
        
        # 计算平均损失和准确率
        train_loss /= epochs
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        # 获取当前学习率
        current_lr = None
        if feature_optimizer:
            for param_group in feature_optimizer.param_groups:
                current_lr = param_group['lr']
                break
        
        # 返回训练后的模型状态和统计信息
        return model.state_dict(), {
            'loss': train_loss,
            'accuracy': accuracy,
            'time': train_time,
            'lr_final': current_lr or self.learning_rate
        }
    
    def train_split_learning_with_optimizers(self, client_model, server_model, optimizers=None, global_classifier=None, rounds=1):
        """
        使用组件优化器进行拆分学习训练
        
        Args:
            client_model: 客户端模型
            server_model: 服务器模型
            optimizers: 组件优化器字典
            global_classifier: 全局分类器
            rounds: 训练轮数
            
        Returns:
            client_state: 客户端模型参数
            server_state: 服务器模型参数
            stats: 训练统计信息
        """
        # 设置训练模式
        client_model.train()
        server_model.train()
        if global_classifier is not None:
            global_classifier.train()
        
        # 移动模型到设备
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        if global_classifier is not None:
            global_classifier = global_classifier.to(self.device)
        
        # 如果没有提供优化器，使用训练策略创建
        if optimizers is None:
            optimizers = self.training_strategy.create_component_optimizers(client_model, global_classifier)
        
        # 获取优化器
        client_optimizer = optimizers.get('feature_extractor')
        classifier_optimizer = optimizers.get('classifier')
        global_optimizer = optimizers.get('global_classifier')
        
        # 为服务器模型创建优化器
        server_optimizer = self.training_strategy.create_optimizer(
            server_model,
            optimizer_type="Adam"
        )
        
        # 创建数据加载器
        train_loader = self.train_data
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 记录统计信息
        stats = {
            'loss': 0, 'accuracy': 0,
            'local_loss': 0, 'local_accuracy': 0,
            'global_loss': 0, 'global_accuracy': 0,
            'time': 0, 'data_transmitted_mb': 0
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 拆分学习训练循环
        for round_idx in range(rounds):
            round_loss = 0
            round_correct = 0
            round_total = 0
            
            # 本地分类器统计
            local_loss = 0
            local_correct = 0
            
            # 全局分类器统计
            global_loss = 0
            global_correct = 0
            
            # 数据传输量
            data_transmitted = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 1. 客户端前向传播
                client_out, client_features = client_model(data)
                
                # 计算本地分类器的损失和准确率(如果有)
                if client_out is not None:
                    local_batch_loss = criterion(client_out, target)
                    local_loss += local_batch_loss.item()
                    _, local_predicted = torch.max(client_out.data, 1)
                    local_correct += (local_predicted == target).sum().item()
                
                # 2. 发送客户端特征到服务器
                # 模拟数据传输
                feature_size_mb = client_features.element_size() * client_features.nelement() / (1024 * 1024)
                data_transmitted += feature_size_mb
                
                # 3. 服务器前向传播
                server_features = server_model(client_features)
                
                # 4. 应用全局分类器(如果有)
                if global_classifier is not None:
                    global_out = global_classifier(server_features)
                    global_batch_loss = criterion(global_out, target)
                    global_loss += global_batch_loss.item()
                    _, global_predicted = torch.max(global_out.data, 1)
                    global_correct += (global_predicted == target).sum().item()
                    
                    # 使用全局分类器的输出作为主要损失
                    loss = global_batch_loss
                else:
                    # 如果没有全局分类器，使用本地分类器的损失
                    loss = local_batch_loss if 'local_batch_loss' in locals() else 0
                
                # 清零所有优化器的梯度
                if client_optimizer:
                    client_optimizer.zero_grad()
                if server_optimizer:
                    server_optimizer.zero_grad()
                if classifier_optimizer:
                    classifier_optimizer.zero_grad()
                if global_optimizer:
                    global_optimizer.zero_grad()
                
                # 5. 反向传播 - 计算梯度
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=1.0)
                if global_classifier is not None:
                    torch.nn.utils.clip_grad_norm_(global_classifier.parameters(), max_norm=1.0)
                
                # 6. 使用不同的优化器更新各个组件
                if client_optimizer:
                    client_optimizer.step()
                if server_optimizer:
                    server_optimizer.step()
                if classifier_optimizer:
                    classifier_optimizer.step()
                if global_optimizer:
                    global_optimizer.step()
                
                # 更新统计信息
                round_loss += loss.item()
                
                # 使用最终预测(全局或本地)更新准确率统计
                if global_classifier is not None:
                    pred = global_predicted
                else:
                    pred = local_predicted if 'local_predicted' in locals() else None
                
                if pred is not None:
                    round_total += target.size(0)
                    round_correct += (pred == target).sum().item()
            
            # 计算本轮平均损失和准确率
            num_batches = len(train_loader)
            round_loss /= num_batches
            round_acc = 100.0 * round_correct / round_total if round_total > 0 else 0
            
            # 计算本地分类器指标
            local_loss /= num_batches
            local_acc = 100.0 * local_correct / round_total if round_total > 0 else 0
            
            # 计算全局分类器指标
            if global_classifier is not None:
                global_loss /= num_batches
                global_acc = 100.0 * global_correct / round_total if round_total > 0 else 0
            
            # 累计统计信息
            stats['loss'] += round_loss / rounds
            stats['accuracy'] += round_acc / rounds
            stats['local_loss'] += local_loss / rounds
            stats['local_accuracy'] += local_acc / rounds
            if global_classifier is not None:
                stats['global_loss'] += global_loss / rounds
                stats['global_accuracy'] += global_acc / rounds
        
        # 计算总训练时间和传输数据总量
        stats['time'] = time.time() - start_time
        stats['data_transmitted_mb'] = data_transmitted
        stats['data_size'] = len(self.train_data.dataset) if hasattr(self.train_data, 'dataset') else 0
        
        # 返回训练后的模型状态和统计信息
        return client_model.state_dict(), server_model.state_dict(), stats

class ClientManager:
    """管理多个客户端的类"""
    
    def __init__(self):
        """初始化客户端管理器"""
        self.clients = {}
        self.client_clusters = {}
        self.default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def add_client(self, client_id, tier, train_dataset, test_dataset, 
              device=None, lr=0.001, local_epochs=5, resources=None):
        """
        添加客户端
        
        Args:
            client_id: 客户端ID
            tier: 客户端tier级别
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            device: 计算设备
            lr: 学习率
            local_epochs: 本地训练轮数
            resources: 资源信息
            
        Returns:
            创建的客户端
        """
        device = device or self.default_device
        
        # 创建客户端前确保设备存在并可用
        if 'cuda' in device and not torch.cuda.is_available():
            print(f"CUDA不可用，将客户端 {client_id} 设置为CPU")
            device = 'cpu'
            
        # 创建客户端
        client = EnhancedClient(
            idx=client_id,
            device=device,
            lr=lr,
            local_epochs=local_epochs,
            dataset_train=train_dataset,
            dataset_test=test_dataset,
            tier=tier,
            resources=resources
        )
        
        # 保存客户端
        self.clients[client_id] = client
        
        return client
    
    def get_client(self, client_id):
        """
        获取客户端
        
        Args:
            client_id: 客户端ID
            
        Returns:
            客户端对象
        """
        return self.clients.get(client_id)
    
    def update_client_tier(self, client_id, new_tier):
        """
        更新客户端tier级别
        
        Args:
            client_id: 客户端ID
            new_tier: 新的tier级别
            
        Returns:
            是否成功更新
        """
        if client_id in self.clients:
            self.clients[client_id].update_tier(new_tier)
            return True
        return False
    
    def set_client_clusters(self, clusters):
        """
        设置客户端聚类
        
        Args:
            clusters: 聚类映射
        """
        self.client_clusters = clusters
    
    def get_client_cluster(self, client_id):
        """
        获取客户端所属聚类
        
        Args:
            client_id: 客户端ID
            
        Returns:
            聚类ID
        """
        for cluster_id, client_ids in self.client_clusters.items():
            if client_id in client_ids:
                return cluster_id
        return None
    
    def pre_train_clients(self, client_ids, models, epochs=5):
        """
        预训练客户端模型
        
        Args:
            client_ids: 客户端ID列表
            models: 客户端模型字典
            epochs: 预训练轮数
            
        Returns:
            预训练结果
        """
        results = {}
        
        for client_id in client_ids:
            if client_id not in self.clients or client_id not in models:
                continue
                
            client = self.clients[client_id]
            model = models[client_id].to(client.device)
            
            # 执行预训练
            model_state, time_used = client.pre_train(model, pretrain_epochs=epochs)
            
            # 保存结果
            results[client_id] = {
                'model_state': model_state,
                'time': time_used,
                'model_size_mb': client.model_size_mb
            }
        
        return results
    
    def train_clients_local(self, client_ids, models, epochs=None):
        """
        本地训练客户端模型
        
        Args:
            client_ids: 客户端ID列表
            models: 客户端模型字典
            epochs: 训练轮数
            
        Returns:
            训练结果
        """
        results = {}
        
        for client_id in client_ids:
            if client_id not in self.clients or client_id not in models:
                continue
                
            client = self.clients[client_id]
            model = models[client_id].to(client.device)
            
            # 执行本地训练
            model_state, stats = client.local_train_with_optimizers(model, epochs=epochs)
            
            # 保存结果
            results[client_id] = {
                'model_state': model_state,
                'stats': stats
            }
        
        return results
    
    def train_clients_split_learning(self, client_ids, client_models, server_models, 
                               global_classifier=None, server_optimizers=None, rounds=1):
        """
        拆分学习训练客户端模型
        
        Args:
            client_ids: 客户端ID列表
            client_models: 客户端模型字典
            server_models: 服务器模型字典
            global_classifier: 全局分类器（可选）
            server_optimizers: 服务器优化器字典
            rounds: 训练轮数
            
        Returns:
            训练结果
        """
        results = {}
        
        for client_id in client_ids:
            if client_id not in self.clients or client_id not in client_models or client_id not in server_models:
                continue
                
            client = self.clients[client_id]
            client_model = client_models[client_id].to(client.device)
            server_model = server_models[client_id].to(client.device)
            
            # 确保全局分类器在正确的设备上
            if global_classifier is not None:
                global_classifier_device = global_classifier.to(client.device)
            else:
                global_classifier_device = None
            
            # 获取服务器优化器
            server_optimizer = None
            if server_optimizers and client_id in server_optimizers:
                server_optimizer = server_optimizers[client_id]
            
            # 执行拆分学习训练，使用优化后的方法
            client_state, server_state, stats = client.train_split_learning_with_optimizers(
                client_model, server_model, 
                global_classifier=global_classifier_device,
                rounds=rounds
            )
            
            # 保存结果
            results[client_id] = {
                'client_state': client_state,
                'server_state': server_state,
                'stats': stats
            }
        
        return results
    
    def evaluate_clients(self, client_ids, client_models, server_models, global_classifier=None):
        """
        评估客户端模型
        
        Args:
            client_ids: 客户端ID列表
            client_models: 客户端模型字典
            server_models: 服务器模型字典
            global_classifier: 全局分类器（可选）
            
        Returns:
            评估结果
        """
        results = {}
        
        for client_id in client_ids:
            if client_id not in self.clients or client_id not in client_models or client_id not in server_models:
                continue
                
            client = self.clients[client_id]
            client_model = client_models[client_id].to(client.device)
            server_model = server_models[client_id].to(client.device)
            
            # 确保全局分类器在正确设备上
            global_classifier_device = None
            if global_classifier is not None:
                global_classifier_device = global_classifier.to(client.device)
            
            # 执行评估，包括全局分类器参数
            eval_stats = client.evaluate(client_model, server_model, global_classifier_device)
            
            # 保存结果
            results[client_id] = eval_stats
        
        return results

def evaluate_client_function(client_id, client_model, server_model, device, client_manager, round_idx, global_classifier=None, **kwargs):
    """客户端评估函数（用于并行训练框架）- 记录本地和全局分类器结果"""
    try:
        client = client_manager.get_client(client_id)
        if client is None:
            return {'error': f"客户端 {client_id} 不存在"}
        
        # 确保client.device与传入的device一致
        client.device = device
        
        # 确保global_classifier完全在正确设备上
        if global_classifier is not None:
            global_classifier = global_classifier.to(device)
            # 确保所有参数和缓冲区都在设备上
            for param in global_classifier.parameters():
                param.data = param.data.to(device)
            for buffer in global_classifier.buffers():
                buffer.data = buffer.data.to(device)
        
        # 执行评估 - 传递全局分类器
        eval_start_time = time.time()
        eval_stats = client.evaluate(client_model, server_model, global_classifier=global_classifier)
        eval_time = time.time() - eval_start_time
        
        # 添加评估时间
        eval_stats['time'] = eval_time
        
        # 记录到wandb - 同时记录本地和全局分类器结果
        eval_metrics = {
            # 本地分类器结果
            f"client_{client_id}/round_{round_idx}/local_test_loss": eval_stats['local_loss'],
            f"client_{client_id}/round_{round_idx}/local_test_accuracy": eval_stats['local_accuracy'],
            
            # 全局分类器结果
            f"client_{client_id}/round_{round_idx}/global_test_loss": eval_stats['global_loss'],
            f"client_{client_id}/round_{round_idx}/global_test_accuracy": eval_stats['global_accuracy'],
            
            # 兼容旧代码
            f"client_{client_id}/round_{round_idx}/test_loss": eval_stats['loss'],
            f"client_{client_id}/round_{round_idx}/test_accuracy": eval_stats['accuracy'],
            
            f"client_{client_id}/round_{round_idx}/test_time": eval_time
        }
        
        # 记录本地分类器每个类别的准确率
        for i, acc in enumerate(eval_stats.get('local_per_class_accuracy', [])):
            eval_metrics[f"client_{client_id}/round_{round_idx}/local_class_{i}_accuracy"] = acc
        
        # 记录全局分类器每个类别的准确率
        for i, acc in enumerate(eval_stats.get('global_per_class_accuracy', [])):
            eval_metrics[f"client_{client_id}/round_{round_idx}/global_class_{i}_accuracy"] = acc
            # 兼容旧代码
            eval_metrics[f"client_{client_id}/round_{round_idx}/class_{i}_accuracy"] = acc
        
        # 在这里我们可以调用wandb.log，但主函数中已经进行了wandb记录
        # wandb.log(eval_metrics)
        
        return eval_stats
        
    except Exception as e:
        import traceback
        error_msg = f"客户端 {client_id} 评估失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}
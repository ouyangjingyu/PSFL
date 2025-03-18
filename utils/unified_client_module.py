import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np


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
        self.ldr_train = dataset_train
        self.ldr_test = dataset_test
        self.tier = tier
        self.tier_scheduler = tier_scheduler
        self.resources = resources or {}
        
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
    
    def init_optimizer(self, model, optimizer_name="Adam", weight_decay=5e-4):
        """
        初始化优化器
        
        Args:
            model: 客户端模型
            optimizer_name: 优化器名称
            weight_decay: 权重衰减参数
        """
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=weight_decay, 
                amsgrad=True
            )
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=self.learning_rate, 
                momentum=0.9,
                nesterov=True,
                weight_decay=weight_decay
            )
        else:
            # 默认使用Adam
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=weight_decay
            )
        
        # 初始化学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=20,
            gamma=0.9
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
        
        # 初始化优化器
        if self.optimizer is None:
            self.init_optimizer(model)
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        for epoch in range(pretrain_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = model(images)
                
                # 处理可能的元组输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 计算损失并反向传播
                labels = labels.to(torch.long)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
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
    
    def local_train(self, model, local_epochs=None):
        """
        本地训练模型
        
        Args:
            model: 客户端模型
            local_epochs: 本地训练轮数（可选）
            
        Returns:
            model_state_dict: 模型参数
            training_stats: 训练统计信息
        """
        model.train()
        time_start = time.time()
        
        # 使用传入的轮数或默认值
        epochs = local_epochs if local_epochs is not None else self.local_epochs
        
        # 初始化优化器
        if self.optimizer is None:
            self.init_optimizer(model)
        
        stats = {
            'loss': [],
            'accuracy': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = model(images)
                
                # 处理可能的元组输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 计算损失并反向传播
                labels = labels.to(torch.long)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
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
                stats['loss'].append(epoch_loss)
                stats['accuracy'].append(epoch_acc)
                stats['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                # 更新学习率
                self.lr_scheduler.step()
                
                print(f"Client {self.idx} - Local Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, LR={self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 更新当前epoch计数
            self.current_epoch += 1
        
        # 计算训练耗时
        training_time = time.time() - time_start
        
        # 计算平均损失和准确率
        avg_loss = np.mean(stats['loss']) if stats['loss'] else 0
        avg_acc = np.mean(stats['accuracy']) if stats['accuracy'] else 0
        
        # 更新训练统计信息
        self.training_stats['local_train_loss'].append(avg_loss)
        self.training_stats['local_train_acc'].append(avg_acc)
        self.training_stats['training_time'].append(training_time)
        
        # 计算模型大小
        self.calculate_model_size(model)
        
        # 创建返回结果
        result = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'time': training_time,
            'epochs': epochs,
            'lr_final': self.optimizer.param_groups[0]['lr'],
            'model_size_mb': self.model_size_mb,
            'data_size': len(self.ldr_train.dataset) if hasattr(self.ldr_train, 'dataset') else 0
        }
        
        return model.state_dict(), result
    
    def train_split_learning(self, client_model, server_model, server_optimizer=None, rounds=1):
        """
        执行拆分学习训练
        
        Args:
            client_model: 客户端模型
            server_model: 服务器模型
            server_optimizer: 服务器优化器
            rounds: 训练轮数
            
        Returns:
            client_model_state_dict: 客户端模型参数
            server_model_state_dict: 服务器模型参数
            stats: 训练统计信息
        """
        # 设置模型为训练模式
        client_model.train()
        server_model.train()
        
        # 初始化客户端优化器
        if self.optimizer is None:
            self.init_optimizer(client_model)
        
        # 初始化服务器优化器（如果未提供）
        if server_optimizer is None:
            server_optimizer = torch.optim.Adam(
                server_model.parameters(),
                lr=self.learning_rate,
                weight_decay=5e-4
            )
        
        # 记录数据传输大小
        intermediate_data_size = 0
        
        # 训练统计信息
        stats = {
            'loss': [],
            'accuracy': [],
            'time': 0,
            'data_transmitted_mb': 0
        }
        
        # 开始计时
        time_start = time.time()
        
        for round_idx in range(rounds):
            round_loss = 0.0
            round_correct = 0
            round_samples = 0
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 客户端前向传播
                client_optimizer = self.optimizer
                client_optimizer.zero_grad()
                features, activations = client_model(images)
                
                # 记录中间特征大小
                features_size_bytes = features.nelement() * features.element_size()
                intermediate_data_size += features_size_bytes + labels.nelement() * labels.element_size()
                
                # 确保特征需要梯度
                features_clone = features.clone().detach().requires_grad_(True)
                
                # 服务器前向传播
                server_optimizer.zero_grad()
                outputs = server_model(features_clone)
                
                # 计算损失
                labels = labels.to(torch.long)
                loss = self.criterion(outputs, labels)
                
                # 服务器反向传播
                loss.backward()
                
                # 获取特征梯度
                features_grad = features_clone.grad.clone().detach()
                
                # 更新服务器模型
                server_optimizer.step()
                
                # 客户端反向传播
                features.backward(features_grad)
                client_optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_size = labels.size(0)
                
                # 更新统计信息
                round_loss += loss.item() * batch_size
                round_correct += batch_correct
                round_samples += batch_size
            
            # 计算每个轮次的平均值
            if round_samples > 0:
                round_loss /= round_samples
                round_acc = 100.0 * round_correct / round_samples
                
                # 更新统计信息
                stats['loss'].append(round_loss)
                stats['accuracy'].append(round_acc)
                
                print(f"Client {self.idx} - SplitLearning Round {round_idx+1}/{rounds}: Loss={round_loss:.4f}, Acc={round_acc:.2f}%")
        
        # 计算训练耗时
        training_time = time.time() - time_start
        
        # 计算中间数据大小（MB）
        intermediate_data_size_mb = intermediate_data_size / (1024 ** 2)
        
        # 更新训练统计信息
        avg_loss = np.mean(stats['loss']) if stats['loss'] else 0
        avg_acc = np.mean(stats['accuracy']) if stats['accuracy'] else 0
        self.training_stats['sl_train_loss'].append(avg_loss)
        self.training_stats['sl_train_acc'].append(avg_acc)
        
        # 更新结果统计信息
        stats['time'] = training_time
        stats['data_transmitted_mb'] = intermediate_data_size_mb
        stats['data_size'] = len(self.ldr_train.dataset) if hasattr(self.ldr_train, 'dataset') else 0
        
        return client_model.state_dict(), server_model.state_dict(), stats
    
    def evaluate(self, client_model, server_model):
        """
        评估模型性能
        
        Args:
            client_model: 客户端模型
            server_model: 服务器模型
            
        Returns:
            eval_stats: 评估统计信息
        """
        # 设置模型为评估模式
        client_model.eval()
        server_model.eval()
        
        test_loss = 0.0
        test_correct = 0
        test_samples = 0
        
        class_correct = [0] * 10  # 假设10个类别
        class_total = [0] * 10
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 客户端前向传播
                features, _ = client_model(images)
                
                # 服务器前向传播
                outputs = server_model(features)
                
                # 计算损失
                labels = labels.to(torch.long)
                loss = self.criterion(outputs, labels)
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_size = labels.size(0)
                
                # 更新统计信息
                test_loss += loss.item() * batch_size
                test_correct += batch_correct
                test_samples += batch_size
                
                # 更新每个类别的准确率
                for i in range(batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label < len(class_total):
                        class_total[label] += 1
                        if pred == label:
                            class_correct[label] += 1
        
        # 计算平均损失和准确率
        if test_samples > 0:
            test_loss /= test_samples
            test_acc = 100.0 * test_correct / test_samples
        else:
            test_loss = 0
            test_acc = 0
        
        # 计算每个类别的准确率
        per_class_accuracy = []
        for i in range(len(class_total)):
            if class_total[i] > 0:
                per_class_accuracy.append(100.0 * class_correct[i] / class_total[i])
            else:
                per_class_accuracy.append(0)
        
        # 更新统计信息
        self.training_stats['test_loss'].append(test_loss)
        self.training_stats['test_acc'].append(test_acc)
        
        # 创建评估结果
        eval_stats = {
            'loss': test_loss,
            'accuracy': test_acc,
            'per_class_accuracy': per_class_accuracy,
            'samples': test_samples
        }
        
        print(f"Client {self.idx} - Test: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        return eval_stats


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
            model_state, stats = client.local_train(model, local_epochs=epochs)
            
            # 保存结果
            results[client_id] = {
                'model_state': model_state,
                'stats': stats
            }
        
        return results
    
    def train_clients_split_learning(self, client_ids, client_models, server_models, server_optimizers=None, rounds=1):
        """
        拆分学习训练客户端模型
        
        Args:
            client_ids: 客户端ID列表
            client_models: 客户端模型字典
            server_models: 服务器模型字典
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
            
            # 获取服务器优化器
            server_optimizer = None
            if server_optimizers and client_id in server_optimizers:
                server_optimizer = server_optimizers[client_id]
            
            # 执行拆分学习训练
            client_state, server_state, stats = client.train_split_learning(
                client_model, server_model, server_optimizer, rounds
            )
            
            # 保存结果
            results[client_id] = {
                'client_state': client_state,
                'server_state': server_state,
                'stats': stats
            }
        
        return results
    
    def evaluate_clients(self, client_ids, client_models, server_models):
        """
        评估客户端模型
        
        Args:
            client_ids: 客户端ID列表
            client_models: 客户端模型字典
            server_models: 服务器模型字典
            
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
            
            # 执行评估
            eval_stats = client.evaluate(client_model, server_model)
            
            # 保存结果
            results[client_id] = eval_stats
        
        return results
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
    
        # 修改EnhancedClient.local_train方法，确保设备正确
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
        # 确保模型在正确的设备上
        model = model.to(self.device)
        
        # 更关键的是，确保模型的所有参数都在正确的设备上
        for param in model.parameters():
            param.data = param.data.to(self.device)
        
        # 然后设置为训练模式
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
                # 确保数据也在正确的设备上
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
    
    def train_split_learning(self, client_model, server_model, global_classifier=None, server_optimizer=None, rounds=1):
        """
        执行拆分学习训练，同时记录本地分类器和全局分类器的训练指标
        """
        # 确保所有模型都在正确的设备上
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        if global_classifier is not None:
            global_classifier = global_classifier.to(self.device)
        
        # 关键：确保所有模型参数都在正确设备上
        for param in client_model.parameters():
            param.data = param.data.to(self.device)
        for param in server_model.parameters():
            param.data = param.data.to(self.device)
        if global_classifier is not None:
            for param in global_classifier.parameters():
                param.data = param.data.to(self.device)
        
        # 设置模型为训练模式
        client_model.train()
        server_model.train()
        if global_classifier is not None:
            global_classifier.train()
        
        # 初始化客户端优化器
        if self.optimizer is None:
            self.init_optimizer(client_model)
        client_optimizer = self.optimizer
        
        # 初始化服务器优化器（如果未提供）
        if server_optimizer is None:
            server_optimizer = torch.optim.Adam(
                server_model.parameters(),
                lr=self.learning_rate,
                weight_decay=5e-4
            )
        
        # 初始化全局分类器优化器
        if global_classifier is not None:
            global_classifier_optimizer = torch.optim.Adam(
                global_classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=5e-4
            )
        
        # 记录数据传输大小
        intermediate_data_size = 0
        
        # 训练统计信息 - 分别记录本地和全局分类器的指标
        stats = {
            # 全局分类器指标
            'global_loss': [],
            'global_accuracy': [],
            
            # 本地分类器指标
            'local_loss': [],
            'local_accuracy': [],
            
            # 其他统计信息
            'time': 0,
            'data_transmitted_mb': 0
        }
        
        # 开始计时
        time_start = time.time()
        
        for round_idx in range(rounds):
            # 本地分类器指标
            round_local_loss = 0.0
            round_local_correct = 0
            # 全局分类器指标
            round_global_loss = 0.0
            round_global_correct = 0
            # 样本数
            round_samples = 0
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 1. 客户端前向传播
                client_optimizer.zero_grad()
                client_outputs = client_model(images)
                
                # 客户端模型应该返回(logits, features)
                if isinstance(client_outputs, tuple):
                    client_logits, client_features = client_outputs
                    
                    # 计算本地分类器损失和准确率
                    local_loss = self.criterion(client_logits, labels)
                    
                    _, local_predicted = torch.max(client_logits.data, 1)
                    local_batch_correct = (local_predicted == labels).sum().item()
                    
                    # 更新本地分类器统计
                    round_local_loss += local_loss.item() * labels.size(0)
                    round_local_correct += local_batch_correct
                else:
                    # 如果没有返回元组，则假设只返回了特征
                    client_features = client_outputs
                    client_logits = None
                    local_loss = 0
                
                # 确保client_features需要梯度
                if not client_features.requires_grad:
                    client_features = client_features.clone().detach().requires_grad_(True)
                
                # 记录中间特征大小
                features_size_bytes = client_features.nelement() * client_features.element_size()
                intermediate_data_size += features_size_bytes + labels.nelement() * labels.element_size()
                
                # 2. 服务器前向传播
                server_optimizer.zero_grad()
                server_features = server_model(client_features)
                
                # 3. 全局分类器前向传播
                if global_classifier is not None:
                    global_classifier_optimizer.zero_grad()
                    global_outputs = global_classifier(server_features)
                    
                    # 计算全局损失
                    global_loss = self.criterion(global_outputs, labels)
                    
                    # 计算全局准确率
                    _, global_predicted = torch.max(global_outputs.data, 1)
                    global_batch_correct = (global_predicted == labels).sum().item()
                    
                    # 更新全局分类器统计
                    round_global_loss += global_loss.item() * labels.size(0)
                    round_global_correct += global_batch_correct
                    
                    # 4. 反向传播 - 使用全局损失
                    global_loss.backward()
                else:
                    # 如果没有全局分类器，则使用服务器特征作为输出
                    global_outputs = server_features
                    global_loss = self.criterion(global_outputs, labels)
                    global_loss.backward()
                
                # 5. 更新全局分类器
                if global_classifier is not None:
                    global_classifier_optimizer.step()
                
                # 6. 获取服务器特征的梯度
                server_features_grad = None
                if server_features.grad is not None:
                    server_features_grad = server_features.grad.clone()
                
                # 7. 更新服务器模型
                server_optimizer.step()
                
                # 8. 获取客户端特征的梯度
                client_features_grad = None
                if client_features.grad is not None:
                    client_features_grad = client_features.grad.clone()
                
                # 9. 客户端反向传播和更新
                if client_features_grad is not None:
                    if not client_features.requires_grad:
                        client_features.backward(client_features_grad)
                    client_optimizer.step()
                
                # 更新样本数
                round_samples += labels.size(0)
            
            # 计算每个轮次的平均值
            if round_samples > 0:
                # 本地分类器指标
                round_local_loss /= round_samples
                round_local_acc = 100.0 * round_local_correct / round_samples
                stats['local_loss'].append(round_local_loss)
                stats['local_accuracy'].append(round_local_acc)
                
                # 全局分类器指标
                round_global_loss /= round_samples
                round_global_acc = 100.0 * round_global_correct / round_samples
                stats['global_loss'].append(round_global_loss)
                stats['global_accuracy'].append(round_global_acc)
                
                print(f"客户端 {self.idx} - 拆分学习轮次 {round_idx+1}/{rounds}:")
                print(f"  本地分类器: 损失={round_local_loss:.4f}, 准确率={round_local_acc:.2f}%")
                print(f"  全局分类器: 损失={round_global_loss:.4f}, 准确率={round_global_acc:.2f}%")
        
        # 计算训练耗时
        training_time = time.time() - time_start
        
        # 计算中间数据大小（MB）
        intermediate_data_size_mb = intermediate_data_size / (1024 ** 2)
        
        # 更新训练统计信息
        avg_local_loss = np.mean(stats['local_loss']) if stats['local_loss'] else 0
        avg_local_acc = np.mean(stats['local_accuracy']) if stats['local_accuracy'] else 0
        avg_global_loss = np.mean(stats['global_loss']) if stats['global_loss'] else 0
        avg_global_acc = np.mean(stats['global_accuracy']) if stats['global_accuracy'] else 0
        
        self.training_stats['sl_train_loss'].append(avg_global_loss)
        self.training_stats['sl_train_acc'].append(avg_global_acc)
        
        # 更新结果统计信息
        stats['time'] = training_time
        stats['data_transmitted_mb'] = intermediate_data_size_mb
        stats['data_size'] = len(self.ldr_train.dataset) if hasattr(self.ldr_train, 'dataset') else 0
        
        # 为了兼容旧代码，保留原有的loss和accuracy字段，使用全局分类器的结果
        stats['loss'] = avg_global_loss
        stats['accuracy'] = avg_global_acc
        
        return client_model.state_dict(), server_model.state_dict(), stats
        
    def evaluate(self, client_model, server_model, global_classifier=None):
        """
        评估模型性能 - 同时评估客户端本地分类器和端到端的全局分类器
        """
        # 确保所有模型都在正确的设备上
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        if global_classifier is not None:
            global_classifier = global_classifier.to(self.device)
        
        # 关键：确保所有模型参数都在正确设备上
        for param in client_model.parameters():
            param.data = param.data.to(self.device)
        for param in server_model.parameters():
            param.data = param.data.to(self.device)
        if global_classifier is not None:
            for param in global_classifier.parameters():
                param.data = param.data.to(self.device)
        
        # 设置模型为评估模式
        client_model.eval()
        server_model.eval()
        if global_classifier is not None:
            global_classifier.eval()
        
        # 客户端本地分类器指标
        local_test_loss = 0.0
        local_test_correct = 0
        local_class_correct = [0] * 10  # 假设10个类别
        local_class_total = [0] * 10
        
        # 全局分类器指标
        global_test_loss = 0.0
        global_test_correct = 0
        global_class_correct = [0] * 10
        global_class_total = [0] * 10
        
        test_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                # 确保数据在正确的设备上
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 1. 客户端前向传播
                client_outputs = client_model(images)
                
                # 客户端模型应该返回(logits, features)
                if isinstance(client_outputs, tuple):
                    client_logits, client_features = client_outputs
                    
                    # 评估客户端本地分类器性能
                    local_loss = self.criterion(client_logits, labels)
                    local_test_loss += local_loss.item() * labels.size(0)
                    
                    _, local_predicted = torch.max(client_logits.data, 1)
                    local_batch_correct = (local_predicted == labels).sum().item()
                    local_test_correct += local_batch_correct
                    
                    # 更新本地分类器每个类别的准确率
                    for i in range(labels.size(0)):
                        label = labels[i]
                        pred = local_predicted[i]
                        if label < len(local_class_total):
                            local_class_total[label] += 1
                            if pred == label:
                                local_class_correct[label] += 1
                else:
                    # 如果客户端没有返回元组，则无法评估本地分类器
                    client_features = client_outputs
                    local_test_loss = 0
                    local_test_correct = 0
                
                # 2. 服务器前向传播
                server_features = server_model(client_features)
                
                # 3. 全局分类器前向传播
                if global_classifier is not None:
                    global_outputs = global_classifier(server_features)
                    
                    # 评估全局分类器性能
                    global_loss = self.criterion(global_outputs, labels)
                    global_test_loss += global_loss.item() * labels.size(0)
                    
                    _, global_predicted = torch.max(global_outputs.data, 1)
                    global_batch_correct = (global_predicted == labels).sum().item()
                    global_test_correct += global_batch_correct
                    
                    # 更新全局分类器每个类别的准确率
                    for i in range(labels.size(0)):
                        label = labels[i]
                        pred = global_predicted[i]
                        if label < len(global_class_total):
                            global_class_total[label] += 1
                            if pred == label:
                                global_class_correct[label] += 1
                
                # 更新总样本数
                test_samples += labels.size(0)
        
        # 计算本地分类器平均损失和准确率
        if test_samples > 0:
            local_test_loss /= test_samples
            local_test_acc = 100.0 * local_test_correct / test_samples
            
            # 计算全局分类器平均损失和准确率
            global_test_loss /= test_samples
            global_test_acc = 100.0 * global_test_correct / test_samples
        else:
            local_test_loss = 0
            local_test_acc = 0
            global_test_loss = 0
            global_test_acc = 0
        
        # 计算每个类别的准确率
        local_per_class_accuracy = []
        for i in range(len(local_class_total)):
            if local_class_total[i] > 0:
                local_per_class_accuracy.append(100.0 * local_class_correct[i] / local_class_total[i])
            else:
                local_per_class_accuracy.append(0)
        
        global_per_class_accuracy = []
        for i in range(len(global_class_total)):
            if global_class_total[i] > 0:
                global_per_class_accuracy.append(100.0 * global_class_correct[i] / global_class_total[i])
            else:
                global_per_class_accuracy.append(0)
        
        # 更新统计信息
        self.training_stats['test_loss'].append(global_test_loss)  # 使用全局分类器损失
        self.training_stats['test_acc'].append(global_test_acc)    # 使用全局分类器准确率
        
        # 创建评估结果 - 包含本地和全局分类器结果
        eval_stats = {
            # 本地分类器结果
            'local_loss': local_test_loss,
            'local_accuracy': local_test_acc,
            'local_per_class_accuracy': local_per_class_accuracy,
            
            # 全局分类器结果
            'global_loss': global_test_loss,
            'global_accuracy': global_test_acc,
            'global_per_class_accuracy': global_per_class_accuracy,
            
            # 兼容旧代码，使用全局分类器结果
            'loss': global_test_loss,
            'accuracy': global_test_acc,
            'per_class_accuracy': global_per_class_accuracy,
            
            'samples': test_samples
        }
        
        print(f"客户端 {self.idx} - 测试:")
        print(f"  本地分类器: 损失={local_test_loss:.4f}, 准确率={local_test_acc:.2f}%")
        print(f"  全局分类器: 损失={global_test_loss:.4f}, 准确率={global_test_acc:.2f}%")
        
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
            model_state, stats = client.local_train(model, local_epochs=epochs)
            
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
            
            # 执行拆分学习训练，传递全局分类器
            client_state, server_state, stats = client.train_split_learning(
                client_model, server_model, 
                global_classifier=global_classifier_device,
                server_optimizer=server_optimizer, 
                rounds=rounds
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

def train_client_with_global_classifier(client_id, client_model, server_model, device, 
                         client_manager, round_idx, coordinator, local_epochs=None, split_rounds=1, global_classifier=None, **kwargs):
    """
    使用全局分类器服务的客户端训练函数
    """
    try:
        client = client_manager.get_client(client_id)
        if client is None:
            return {'error': f"客户端 {client_id} 不存在"}
        
        # 确保客户端设备与传入的设备一致
        client.device = device
        
        # 记录开始时间
        start_time = time.time()
        
        # 第1步：本地训练
        client_model_copy = copy.deepcopy(client_model)
        client_state, local_stats = client.local_train(client_model_copy, local_epochs)
        local_train_time = local_stats['time']
        
        # 获取客户端所属聚类
        cluster_id = client_manager.get_client_cluster(client_id)
        
        # 第2步：拆分学习但使用全局分类器服务
        # 修改EnhancedClient.train_split_learning的逻辑，以支持全局分类器服务
        
        # 初始化客户端优化器
        if not hasattr(client, 'optimizer') or client.optimizer is None:
            client.init_optimizer(client_model_copy)
        client_optimizer = client.optimizer
        
        # 初始化服务器优化器
        server_model_copy = copy.deepcopy(server_model).to(device)
        server_optimizer = torch.optim.Adam(
            server_model_copy.parameters(),
            lr=client.learning_rate,
            weight_decay=5e-4
        )
        
        # 设置模型为训练模式
        client_model_copy.train()
        server_model_copy.train()
        
        # 记录数据传输大小
        intermediate_data_size = 0
        
        # 训练统计信息
        stats = {
            'global_loss': [],
            'global_accuracy': [],
            'local_loss': [],
            'local_accuracy': [],
            'time': 0,
            'data_transmitted_mb': 0
        }
        
        # 开始计时
        sl_start_time = time.time()
        
        for round_idx in range(split_rounds):
            # 本地分类器指标
            round_local_loss = 0.0
            round_local_correct = 0
            # 全局分类器指标
            round_global_loss = 0.0
            round_global_correct = 0
            # 样本数
            round_samples = 0
            
            for batch_idx, (images, labels) in enumerate(client.ldr_train):
                images, labels = images.to(device), labels.to(device)
                
                # 1. 客户端前向传播
                client_optimizer.zero_grad()
                client_outputs = client_model_copy(images)
                
                # 客户端模型应该返回(logits, features)
                if isinstance(client_outputs, tuple):
                    client_logits, client_features = client_outputs
                    
                    # 计算本地分类器损失和准确率
                    local_loss = client.criterion(client_logits, labels)
                    
                    _, local_predicted = torch.max(client_logits.data, 1)
                    local_batch_correct = (local_predicted == labels).sum().item()
                    
                    # 更新本地分类器统计
                    round_local_loss += local_loss.item() * labels.size(0)
                    round_local_correct += local_batch_correct
                else:
                    # 如果没有返回元组，则假设只返回了特征
                    client_features = client_outputs
                    client_logits = None
                    local_loss = 0
                
                # 确保client_features需要梯度
                if not client_features.requires_grad:
                    client_features = client_features.clone().detach().requires_grad_(True)
                
                # 记录中间特征大小
                features_size_bytes = client_features.nelement() * client_features.element_size()
                intermediate_data_size += features_size_bytes + labels.nelement() * labels.element_size()
                
                # 2. 服务器前向传播
                server_optimizer.zero_grad()
                server_features = server_model_copy(client_features)
                
                # 3. 使用全局分类器服务处理特征
                global_result = coordinator.process_features_sync(
                    cluster_id, server_features, labels
                )
                
                global_loss = global_result['loss']
                global_metrics = global_result['metrics']
                feature_gradients = global_result['gradients']
                
                # 更新全局分类器统计
                global_batch_correct = int(global_metrics['accuracy'] * labels.size(0) / 100)
                round_global_loss += global_loss * labels.size(0)
                round_global_correct += global_batch_correct
                
                # 4. 使用全局分类器返回的梯度更新服务器模型
                if feature_gradients is not None:
                    server_features.backward(feature_gradients)
                server_optimizer.step()
                
                # 5. 更新客户端模型
                if client_features.grad is not None:
                    client_optimizer.step()
                
                # 更新样本数
                round_samples += labels.size(0)
            
            # 计算每个轮次的平均值
            if round_samples > 0:
                # 本地分类器指标
                round_local_loss /= round_samples
                round_local_acc = 100.0 * round_local_correct / round_samples
                stats['local_loss'].append(round_local_loss)
                stats['local_accuracy'].append(round_local_acc)
                
                # 全局分类器指标
                round_global_loss /= round_samples
                round_global_acc = 100.0 * round_global_correct / round_samples
                stats['global_loss'].append(round_global_loss)
                stats['global_accuracy'].append(round_global_acc)
        
        # 计算训练耗时
        sl_train_time = time.time() - sl_start_time
        
        # 计算中间数据大小（MB）
        intermediate_data_size_mb = intermediate_data_size / (1024 ** 2)
        
        # 更新训练统计信息
        avg_local_loss = np.mean(stats['local_loss']) if stats['local_loss'] else 0
        avg_local_acc = np.mean(stats['local_accuracy']) if stats['local_accuracy'] else 0
        avg_global_loss = np.mean(stats['global_loss']) if stats['global_loss'] else 0
        avg_global_acc = np.mean(stats['global_accuracy']) if stats['global_accuracy'] else 0
        
        # 更新结果统计信息
        stats['time'] = sl_train_time
        stats['total_time'] = local_train_time + sl_train_time
        stats['data_transmitted_mb'] = intermediate_data_size_mb
        stats['data_size'] = len(client.ldr_train.dataset) if hasattr(client.ldr_train, 'dataset') else 0
        
        # 为了兼容旧代码，保留原有的loss和accuracy字段
        stats['loss'] = avg_global_loss
        stats['accuracy'] = avg_global_acc
        
        # 返回结果
        return {
            'client_model_state': client_model_copy.state_dict(),
            'server_model_state': server_model_copy.state_dict(),
            'local_train': local_stats,
            'split_learning': stats,
            'loss': avg_global_loss,
            'accuracy': avg_global_acc,
            'local_sl_loss': avg_local_loss,
            'local_sl_accuracy': avg_local_acc,
            'global_sl_loss': avg_global_loss,
            'global_sl_accuracy': avg_global_acc,
            'local_train_loss': local_stats.get('loss', 0),
            'local_train_accuracy': local_stats.get('accuracy', 0),
            'time': local_train_time + sl_train_time,
            'local_train_time': local_train_time,
            'sl_train_time': sl_train_time,
            'data_size': stats['data_size'],
            'communication_mb': intermediate_data_size_mb,
            'lr': local_stats.get('lr_final', client.learning_rate)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"客户端 {client_id} 训练失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'error': error_msg}
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import copy
from collections import defaultdict

class TierHFLClient:
    """TierHFL客户端类"""
    def __init__(self, client_id, tier, train_data, test_data, model, 
                 device='cuda', lr=0.001, local_epochs=1):
        self.client_id = client_id
        self.tier = tier
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        
        # 创建优化器 -在train-epoch中创建
        
        # 学习率调度器
        self.lr_scheduler_configs = {
            'shared': {'type': 'cosine', 'T_max': 100, 'eta_min': lr*0.1},
            'global': {'type': 'step', 'step_size': 20, 'gamma': 0.5},
            'local': {'type': 'plateau', 'factor': 0.7, 'patience': 5, 'min_lr': lr*0.05}
        }
        
        # 训练统计
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'local_loss': [],
            'global_loss': [],
            'feature_loss': [],
        }
        
        # 本地数据分布信息
        self.data_distribution = self._analyze_data_distribution()
    
    def _analyze_data_distribution(self):
        """分析本地数据分布"""
        distribution = defaultdict(int)
        total = 0
        
        # 统计各类别数量
        for _, labels in self.train_data:
            for label in labels.numpy():
                distribution[int(label)] += 1
                total += 1
        
        # 计算百分比
        percentage = {k: v/total*100 for k, v in distribution.items()}
        
        return {
            'counts': dict(distribution),
            'percentage': percentage,
            'total': total
        }
    
    def get_distribution_similarity(self, other_distribution):
        """计算与其他分布的相似度"""
        # 获取所有可能的类别
        all_classes = set(self.data_distribution['percentage'].keys()).union(
            other_distribution['percentage'].keys())
        
        # 计算JS散度
        kl_divergence = 0
        for c in all_classes:
            p1 = self.data_distribution['percentage'].get(c, 0) / 100
            p2 = other_distribution['percentage'].get(c, 0) / 100
            
            # 防止除零
            p1 = max(p1, 1e-10)
            p2 = max(p2, 1e-10)
            
            # 计算中间分布
            m = (p1 + p2) / 2
            
            # 计算KL散度
            kl_divergence += 0.5 * (p1 * np.log(p1/m) + p2 * np.log(p2/m))
        
        # 相似度 = 1 - JS散度
        similarity = 1 - min(1, np.sqrt(kl_divergence))
        
        return similarity
    
    def train_epoch(self, server_model, global_classifier, loss_fn, gradient_guide, round_idx, total_rounds=100):
        """修复梯度计算冲突的训练方法"""
        self.model.train()
        server_model.train()
        global_classifier.train()
        
        # 使用固定值代替动态查询，避免潜在的内存访问问题
        clients_in_group = 3  # 默认值
        
        # 创建优化器
        local_optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if 'local_path' in n or 'local_classifier' in n],
            lr=self.lr * 1.2
        )
        shared_optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if 'conv1' in n or 'bn1' in n or 'layer1' in n or 'layer2' in n or 'global_adapter' in n],
            lr=self.lr
        )
        server_optimizer = torch.optim.Adam(
            server_model.parameters(),
            lr=self.lr * 0.8
        )
        classifier_optimizer = torch.optim.Adam(
            global_classifier.parameters(),
            lr=self.lr
        )
        
        # 训练统计初始化
        epoch_loss = 0
        epoch_local_loss = 0
        epoch_global_loss = 0
        epoch_feature_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 梯度更新计数器
        update_counter = 0
        
        # 训练循环
        for data, target in self.train_data:
            data, target = data.to(self.device), target.to(self.device)
            
            # ===== 步骤1: 只训练个性化路径 =====
            # 确保只有本地路径的参数需要梯度
            for name, param in self.model.named_parameters():
                if 'local_path' in name or 'local_classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            local_optimizer.zero_grad()
            
            # 前向传播 - 本地路径
            local_logits, _, _ = self.model(data)
            local_loss = F.cross_entropy(local_logits, target)
            
            # 反向传播
            local_loss.backward()
            local_optimizer.step()
            
            # ===== 步骤2: 训练共享层和服务器模型 =====
            # 重置参数的requires_grad状态
            for name, param in self.model.named_parameters():
                if 'local_path' in name or 'local_classifier' in name:
                    param.requires_grad = False
                elif any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2', 'global_adapter']):
                    param.requires_grad = True
            
            # 清除梯度
            shared_optimizer.zero_grad()
            server_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            # 前向传播 - 共享路径和服务器
            _, global_features, local_features = self.model(data)
            server_features = server_model(global_features)
            global_logits = global_classifier(server_features)
            
            # 计算损失
            global_loss = F.cross_entropy(global_logits, target)
            feature_loss = loss_fn._feature_alignment(local_features, server_features)
            
            # 动态权重
            progress = float(round_idx) / float(total_rounds)
            alpha = max(0.3, min(0.7, 0.3 + 0.4 * progress))
            beta = 1.0 - alpha
            
            # 全局损失
            shared_loss = alpha * global_loss + beta * feature_loss
            
            # 反向传播
            shared_loss.backward()
            
            # 更新参数
            shared_optimizer.step()
            
            # 控制服务器更新频率
            update_counter += 1
            update_interval = max(1, int(clients_in_group * 0.5))
            
            if update_counter >= update_interval:
                server_optimizer.step()
                classifier_optimizer.step()
                update_counter = 0
            
            # 重置所有参数的requires_grad状态为True（为下一轮做准备）
            for param in self.model.parameters():
                param.requires_grad = True
            
            # 计算准确率
            with torch.no_grad():
                _, predicted = local_logits.max(1)
                correct = predicted.eq(target).sum().item()
            
            # 统计累计
            total_loss = local_loss.item() + shared_loss.item()
            epoch_loss += total_loss
            epoch_local_loss += local_loss.item()
            epoch_global_loss += global_loss.item()
            epoch_feature_loss += feature_loss.item()
            epoch_acc += correct / target.size(0)
            num_batches += 1
        
        # 计算平均值
        epoch_loss /= num_batches
        epoch_local_loss /= num_batches
        epoch_global_loss /= num_batches
        epoch_feature_loss /= num_batches
        epoch_acc /= num_batches
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        # 更新统计信息
        self.stats['train_loss'].append(epoch_loss)
        self.stats['train_acc'].append(epoch_acc)
        self.stats['local_loss'].append(epoch_local_loss)
        self.stats['global_loss'].append(epoch_global_loss)
        self.stats['feature_loss'].append(epoch_feature_loss)
        
        return {
            'loss': epoch_loss,
            'local_loss': epoch_local_loss,
            'global_loss': epoch_global_loss,
            'feature_loss': epoch_feature_loss,
            'accuracy': epoch_acc * 100,
            'training_time': training_time
        }

    def train(self, server_model, global_classifier, loss_fn, gradient_guide, round_idx, total_rounds=100):
        """训练多个epoch，包含对服务器模型的引用"""
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        server_model = server_model.to(self.device)
        global_classifier = global_classifier.to(self.device)
        
        # 训练多个epoch
        results = []
        for epoch in range(self.local_epochs):
            epoch_result = self.train_epoch(
                server_model, global_classifier, loss_fn, 
                gradient_guide, round_idx, total_rounds)
            results.append(epoch_result)
            
            # 输出训练信息
            print(f"Client {self.client_id} (Tier {self.tier}) - "
                f"Epoch {epoch+1}/{self.local_epochs}, "
                f"Loss: {epoch_result['loss']:.4f}, "
                f"Acc: {epoch_result['accuracy']:.2f}%")
        
        # 计算平均结果
        avg_result = {
            'loss': np.mean([r['loss'] for r in results]),
            'local_loss': np.mean([r['local_loss'] for r in results]),
            'global_loss': np.mean([r['global_loss'] for r in results]),
            'feature_loss': np.mean([r['feature_loss'] for r in results]),
            'accuracy': np.mean([r['accuracy'] for r in results]),
            'training_time': sum([r['training_time'] for r in results])
        }
        
        # 返回训练结果和模型状态
        return avg_result, self.model.state_dict()
    
    def evaluate(self, server_model, global_classifier):
        """评估客户端模型
        
        测试模型在本地测试集上的性能
        
        Args:
            server_model: 服务器模型
            global_classifier: 全局分类器
            
        Returns:
            评估结果统计
        """
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        server_model = server_model.to(self.device)
        global_classifier = global_classifier.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        server_model.eval()
        global_classifier.eval()
        
        # 统计
        test_loss = 0
        local_correct = 0
        global_correct = 0
        total = 0
        
        # 分类别统计
        num_classes = 10  # 默认10类
        class_correct_local = [0] * num_classes
        class_correct_global = [0] * num_classes
        class_total = [0] * num_classes
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_data:
                # 移动数据到设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 客户端前向传播
                local_logits, global_features, _ = self.model(data)
                
                # 服务器处理
                server_features = server_model(global_features)
                global_logits = global_classifier(server_features)
                
                # 计算损失
                loss = criterion(global_logits, target)
                test_loss += loss.item()
                
                # 计算本地和全局准确率
                _, local_preds = local_logits.max(1)
                _, global_preds = global_logits.max(1)
                
                # 更新统计
                total += target.size(0)
                local_correct += local_preds.eq(target).sum().item()
                global_correct += global_preds.eq(target).sum().item()
                
                # 更新分类别统计
                for i in range(target.size(0)):
                    label = target[i].item()
                    if label < num_classes:
                        class_total[label] += 1
                        if local_preds[i] == label:
                            class_correct_local[label] += 1
                        if global_preds[i] == label:
                            class_correct_global[label] += 1
        
        # 计算总体准确率
        local_acc = 100.0 * local_correct / total
        global_acc = 100.0 * global_correct / total
        avg_loss = test_loss / len(self.test_data)
        
        # 计算每个类别的准确率
        local_class_acc = [100.0 * correct / max(1, total) 
                          for correct, total in zip(class_correct_local, class_total)]
        global_class_acc = [100.0 * correct / max(1, total) 
                           for correct, total in zip(class_correct_global, class_total)]
        
        # 更新统计信息
        self.stats['test_loss'].append(avg_loss)
        self.stats['test_acc'].append(global_acc)
        
        return {
            'test_loss': avg_loss,
            'local_accuracy': local_acc,
            'global_accuracy': global_acc,
            'local_per_class_acc': local_class_acc,
            'global_per_class_acc': global_class_acc,
            'total_samples': total
        }
        
class TierHFLClientManager:
    """TierHFL客户端管理器"""
    def __init__(self):
        self.clients = {}
        self.default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def add_client(self, client_id, tier, train_data, test_data, model, 
                  device=None, lr=0.001, local_epochs=1):
        """添加客户端
        
        Args:
            client_id: 客户端ID
            tier: 资源级别
            train_data: 训练数据
            test_data: 测试数据
            model: 客户端模型
            device: 设备
            lr: 学习率
            local_epochs: 本地训练轮次
            
        Returns:
            创建的客户端
        """
        device = device or self.default_device
        
        client = TierHFLClient(
            client_id=client_id,
            tier=tier,
            train_data=train_data,
            test_data=test_data,
            model=model,
            device=device,
            lr=lr,
            local_epochs=local_epochs
        )
        
        self.clients[client_id] = client
        return client
    
    def get_client(self, client_id):
        """获取客户端"""
        return self.clients.get(client_id)
    
    def get_all_clients(self):
        """获取所有客户端"""
        return self.clients
    
    def get_client_distribution(self, client_id):
        """获取客户端数据分布"""
        client = self.get_client(client_id)
        if client:
            return client.data_distribution
        return None
    
    def calculate_distribution_similarity_matrix(self):
        """计算所有客户端之间的分布相似度矩阵"""
        client_ids = list(self.clients.keys())
        n_clients = len(client_ids)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        for i, id1 in enumerate(client_ids):
            for j, id2 in enumerate(client_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自身相似度为1
                else:
                    client1 = self.clients[id1]
                    client2 = self.clients[id2]
                    similarity_matrix[i, j] = client1.get_distribution_similarity(
                        client2.data_distribution)
        
        return similarity_matrix, client_ids
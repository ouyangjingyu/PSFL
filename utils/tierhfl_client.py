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
                 device='cuda', lr=0.001, local_epochs=1, feature_extract_batch=16):
        self.client_id = client_id
        self.tier = tier
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.feature_extract_batch = feature_extract_batch  # 添加特征提取批次大小
        
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
    
    # 添加阶段一训练方法 - 仅训练个性化路径
    def train_phase1(self, round_idx, total_rounds=100):
        """阶段一：训练个性化路径
        
        Args:
            round_idx: 当前轮次
            total_rounds: 总轮次
            
        Returns:
            训练结果
        """
        self.model.train()
        
        # 冻结共享层参数
        for name, param in self.model.named_parameters():
            if 'local_path' in name or 'local_classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 创建优化器 - 只优化个性化路径参数
        local_optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if 'local_path' in n or 'local_classifier' in n],
            lr=self.lr * 1.2
        )
        
        # 训练统计初始化
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练循环
        for data, target in self.train_data:
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除梯度
            local_optimizer.zero_grad()
            
            # 前向传播 - 只需要本地路径
            local_logits, _, _ = self.model(data)
            
            # 计算损失
            loss = F.cross_entropy(local_logits, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            local_optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                _, predicted = local_logits.max(1)
                correct = predicted.eq(target).sum().item()
            
            # 统计累计
            epoch_loss += loss.item()
            epoch_acc += correct / target.size(0)
            num_batches += 1
        
        # 计算平均值
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        # 更新统计信息
        self.stats['train_loss'].append(epoch_loss)
        self.stats['train_acc'].append(epoch_acc)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc * 100,
            'training_time': training_time
        }
    
    # 添加阶段二训练方法 - 仅训练共享层
    def train_phase2(self, server_model, global_classifier, loss_fn, round_idx, total_rounds=100):
        """阶段二：训练共享层和服务器模型
        
        Args:
            server_model: 服务器模型
            global_classifier: 全局分类器
            loss_fn: 损失函数
            round_idx: 当前轮次
            total_rounds: 总轮次
            
        Returns:
            训练结果, 共享层状态
        """
        self.model.train()
        server_model.train()
        global_classifier.train()
        
        # 冻结个性化路径参数，启用共享层参数
        for name, param in self.model.named_parameters():
            if 'local_path' in name or 'local_classifier' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 创建优化器 - 只优化共享层参数
        shared_optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if any(
                layer in n for layer in ['conv1', 'bn1', 'layer1', 'layer2_shared'])],
            lr=self.lr
        )
        
        # 服务器优化器 - 在服务器端创建和更新
        
        # 训练统计初始化
        epoch_loss = 0
        epoch_global_loss = 0
        epoch_feature_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 收集特征和标签，用于经验回放
        collected_features = []
        collected_labels = []
        
        # 训练循环
        for data, target in self.train_data:
            data, target = data.to(self.device), target.to(self.device)
            
            # 清除梯度
            shared_optimizer.zero_grad()
            
            # 前向传播 - 共享路径到服务器
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
            
            # 收集特征和标签
            with torch.no_grad():
                collected_features.append(server_features.detach())
                collected_labels.append(target.detach())
            
            # 计算准确率
            with torch.no_grad():
                _, predicted = global_logits.max(1)
                correct = predicted.eq(target).sum().item()
            
            # 统计累计
            epoch_loss += shared_loss.item()
            epoch_global_loss += global_loss.item()
            epoch_feature_loss += feature_loss.item()
            epoch_acc += correct / target.size(0)
            num_batches += 1
        
        # 计算平均值
        epoch_loss /= num_batches
        epoch_global_loss /= num_batches
        epoch_feature_loss /= num_batches
        epoch_acc /= num_batches
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        # 更新统计信息
        self.stats['global_loss'].append(epoch_global_loss)
        self.stats['feature_loss'].append(epoch_feature_loss)
        
        # 合并所有收集的特征和标签
        if collected_features:
            all_features = torch.cat(collected_features, dim=0)
            all_labels = torch.cat(collected_labels, dim=0)
            # 更新全局分类器的经验回放缓冲区
            global_classifier.update_buffer(all_features, all_labels, self.device)
        
        # 为经验回放添加训练步骤
        self._train_with_replay(server_model, global_classifier)
        
        # 获取共享层状态
        shared_state = {}
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2_shared']):
                shared_state[name] = param.data.clone()
        
        return {
            'loss': epoch_loss,
            'global_loss': epoch_global_loss,
            'feature_loss': epoch_feature_loss,
            'accuracy': epoch_acc * 100,
            'training_time': training_time
        }, shared_state
    
    # 添加经验回放训练方法
    def _train_with_replay(self, server_model, global_classifier, batch_size=64):
        """使用经验回放训练全局分类器
        
        Args:
            server_model: 服务器模型
            global_classifier: 全局分类器
            batch_size: 批次大小
        """
        # 从缓冲区采样数据
        replay_features, replay_labels = global_classifier.sample_from_buffer(batch_size, self.device)
        
        # 如果没有回放数据，跳过
        if replay_features is None:
            return
        
        # 设置训练模式
        server_model.train()
        global_classifier.train()
        
        # 创建优化器 - 只优化分类器
        classifier_optimizer = torch.optim.Adam(global_classifier.parameters(), lr=self.lr * 0.5)
        
        # 清除梯度
        classifier_optimizer.zero_grad()
        
        # 前向传播
        logits = global_classifier(replay_features)
        
        # 计算损失
        loss = F.cross_entropy(logits, replay_labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        classifier_optimizer.step()
    
    # 修改extract_features_and_labels方法，分批提取特征
    def extract_features_and_labels(self):
        """从训练数据中提取特征和标签，确保维度一致"""
        self.model.eval()
        
        features_list = []
        labels_list = []
        
        batch_size = getattr(self, 'feature_extract_batch', 16)
        
        with torch.no_grad():
            for data, target in self.train_data:
                try:
                    # 获取当前批次
                    batch_data = data.to(self.device)
                    batch_target = target.to(self.device)
                    
                    # 提取全局特征
                    _, global_features, _ = self.model(batch_data)
                    
                    # 不要改变特征的维度结构，保持原始输出结构
                    # 这很重要，因为服务器模型期望特定维度的输入
                    
                    # 收集特征和标签
                    features_list.append(global_features.cpu())
                    labels_list.append(batch_target.cpu())
                    
                except Exception as e:
                    print(f"特征提取错误: {str(e)}")
                    continue
        
        # 如果有收集到的特征和标签，合并它们
        if features_list and labels_list:
            all_features = torch.cat(features_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            return all_features, all_labels
        
        return None, None
    
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
                  device=None, lr=0.001, local_epochs=1, feature_extract_batch=16):
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
            feature_extract_batch: 特征提取批次大小
            
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
            local_epochs=local_epochs,
            feature_extract_batch=feature_extract_batch
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
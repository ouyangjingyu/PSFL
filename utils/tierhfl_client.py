import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np

# 混合损失 - 平衡个性化和全局性能
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # 平衡因子
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, local_logits, global_logits, target):
        local_loss = self.criterion(local_logits, target)
        global_loss = self.criterion(global_logits, target)
        return self.alpha * local_loss + (1 - self.alpha) * global_loss, local_loss, global_loss
    
    def update_alpha(self, alpha):
        """更新平衡因子"""
        self.alpha = alpha

# 修复后的特征对齐损失 - 处理不同维度的特征
# 在utils/tierhfl_client.py中约第49行
class FeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
        
    def forward(self, client_features, server_features):
        """计算特征对齐损失，使用自适应池化统一尺寸"""
        # 确保特征是4D或2D张量
        if len(client_features.shape) == 4:  # 4D: [B, C, H, W]
            batch_size = client_features.size(0)
            # 应用全局平均池化得到每个通道的表示
            client_pooled = F.adaptive_avg_pool2d(client_features, (1, 1))
            client_features = client_pooled.view(batch_size, -1)
            
        if len(server_features.shape) == 4:  # 4D: [B, C, H, W]
            batch_size = server_features.size(0)
            # 应用全局平均池化得到每个通道的表示
            server_pooled = F.adaptive_avg_pool2d(server_features, (1, 1))
            server_features = server_pooled.view(batch_size, -1)
        
        # 统一特征维度
        if client_features.size(1) != server_features.size(1):
            target_dim = min(client_features.size(1), server_features.size(1))
            
            # 使用线性投影到相同维度
            if client_features.size(1) > target_dim:
                client_features = client_features[:, :target_dim]
            
            if server_features.size(1) > target_dim:
                server_features = server_features[:, :target_dim]
        
        # 计算特征之间的余弦相似度
        client_norm = F.normalize(client_features, dim=1)
        server_norm = F.normalize(server_features, dim=1)
        
        # 余弦相似度损失
        cosine_sim = torch.mean(torch.sum(client_norm * server_norm, dim=1))
        cosine_loss = 1.0 - cosine_sim
        
        # 均方误差损失
        mse_loss = F.mse_loss(client_features, server_features)
        
        # 结合多种损失
        return 0.7 * cosine_loss + 0.3 * mse_loss

# 增强型客户端 - 支持分层模型和双重学习目标
class TierHFLClient:
    def __init__(self, client_id, tier, train_data, test_data, device='cuda', 
                 lr=0.001, local_epochs=1):
        self.client_id = client_id
        self.tier = tier
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        
        # 训练参数
        self.alpha = 0.5  # 本地与全局损失平衡因子
        self.lambda_feature = 0.1  # 特征对齐损失权重
        
        # 创建损失函数
        self.hybrid_loss = HybridLoss(self.alpha)
        self.feature_alignment_loss = FeatureAlignmentLoss()
        
        # 训练统计信息
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'local_loss': [],
            'global_loss': [],
            'feature_loss': []
        }

    def update_learning_rate(self, round_idx, lr_factor=0.85, decay_rounds=10):
        """根据轮次更新学习率"""
        if round_idx > 0 and round_idx % decay_rounds == 0:
            self.lr *= lr_factor
            return True
        return False
        
    def train(self, client_model, server_model, round_idx=0):
        """客户端训练过程"""
        # 确保模型在正确的设备上
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        
        # 设置为训练模式
        client_model.train()
        server_model.train()
        
        # 创建优化器 - 区分个性化参数和共享参数
        if hasattr(client_model, 'get_shared_params') and hasattr(client_model, 'get_personalized_params'):
            shared_params = list(client_model.get_shared_params().values())
            personalized_params = list(client_model.get_personalized_params().values())
            
            # 使用不同学习率和权重衰减
            # optimizer = torch.optim.Adam([
            #     {'params': shared_params, 'lr': self.lr, 'weight_decay': 1e-4},  # 添加权重衰减
            #     {'params': personalized_params, 'lr': self.lr * 1.5, 'weight_decay': 5e-4}  # 个性化参数使用更高权重衰减
            # ])
            # 使用不同学习率 - 反转权重，提升共享参数学习率
            optimizer = torch.optim.Adam([
                {'params': shared_params, 'lr': self.lr * 1.2},  # 提高共享参数学习率
                {'params': personalized_params, 'lr': self.lr}   # 降低个性化参数学习率
            ])
        else:
            # 后备方案，使用所有参数
            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.lr)
        
        # 开始计时
        start_time = time.time()
        
        # 记录详细训练指标
        batch_times = []
        epoch_losses = []
        epoch_local_losses = []
        epoch_global_losses = []
        epoch_feature_losses = []
        
        # 收集训练统计信息
        epoch_stats = {
            'total_loss': 0.0,
            'local_loss': 0.0,
            'global_loss': 0.0,
            'feature_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        # 添加早停相关变量
        early_stop_patience = 3
        best_loss = float('inf')
        patience_counter = 0
        # 训练循环
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, (data, target) in enumerate(self.train_data):
                batch_start = time.time()
                
                # 将数据移到设备上
                data, target = data.to(self.device), target.to(self.device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 客户端前向传播
                local_logits, features = client_model(data)
                
                # 服务器前向传播
                try:
                    server_output = server_model(features, tier=self.tier)
                    if isinstance(server_output, tuple) and len(server_output) == 2:
                        global_logits, server_features = server_output
                    else:
                        # 如果服务器模型只返回一个输出，假设它是global_logits
                        global_logits = server_output
                        server_features = features  # 直接使用客户端特征
                except Exception as e:
                    print(f"服务器前向传播失败: {str(e)}")
                    # 使用local_logits作为备用
                    global_logits = local_logits
                    server_features = features
                
                # 计算混合损失
                loss, local_loss, global_loss = self.hybrid_loss(local_logits, global_logits, target)
                
                # 尝试计算特征对齐损失
                try:
                    feature_loss = self.feature_alignment_loss(features, server_features)
                    # 添加特征对齐损失
                    total_loss = loss + self.lambda_feature * feature_loss
                except Exception as e:
                    print(f"特征对齐损失计算失败: {str(e)}")
                    # 使用基本损失作为备用
                    total_loss = loss
                    feature_loss = torch.tensor(0.0, device=self.device)
                
                epoch_loss += total_loss.item()
                num_batches += 1

                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                # 更新统计信息
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['local_loss'] += local_loss.item()
                epoch_stats['global_loss'] += global_loss.item()
                epoch_stats['feature_loss'] += feature_loss.item()
                epoch_stats['batch_count'] += 1
                
                # 计算全局分类准确率
                _, predicted = global_logits.max(1)
                epoch_stats['total'] += target.size(0)
                epoch_stats['correct'] += predicted.eq(target).sum().item()
                
                # 记录批次处理时间
                batch_times.append(time.time() - batch_start)
            
            # 记录每个epoch的损失
            epoch_losses.append(epoch_stats['total_loss'] / epoch_stats['batch_count'])
            epoch_local_losses.append(epoch_stats['local_loss'] / epoch_stats['batch_count'])
            epoch_global_losses.append(epoch_stats['global_loss'] / epoch_stats['batch_count'])
            epoch_feature_losses.append(epoch_stats['feature_loss'] / epoch_stats['batch_count'])

            # 检查早停条件
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                if avg_epoch_loss < best_loss * 0.995:  # 至少需要0.5%的改进
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stop_patience:
                    print(f"客户端 {self.client_id} 早停于第 {epoch+1}/{self.local_epochs} 轮")
                    break
        
        # 计算平均损失和准确率
        num_batches = epoch_stats['batch_count']
        avg_loss = epoch_stats['total_loss'] / max(1, num_batches)
        avg_local_loss = epoch_stats['local_loss'] / max(1, num_batches)
        avg_global_loss = epoch_stats['global_loss'] / max(1, num_batches)
        avg_feature_loss = epoch_stats['feature_loss'] / max(1, num_batches)
        accuracy = 100.0 * epoch_stats['correct'] / max(1, epoch_stats['total'])
        
        # 记录训练时间
        training_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / max(1, len(batch_times))
        
        # 保存统计信息
        self.stats['train_loss'].append(avg_loss)
        self.stats['train_acc'].append(accuracy)
        self.stats['local_loss'].append(avg_local_loss)
        self.stats['global_loss'].append(avg_global_loss)
        self.stats['feature_loss'].append(avg_feature_loss)
        
        # 返回更详细的结果
        return {
            'model_state': client_model.state_dict(),
            'avg_loss': avg_loss,
            'avg_local_loss': avg_local_loss,
            'avg_global_loss': avg_global_loss,
            'avg_feature_loss': avg_feature_loss,
            'epoch_losses': epoch_losses,
            'accuracy': accuracy,
            'training_time': training_time,
            'avg_batch_time': avg_batch_time,
            'total_batches': num_batches
        }
        
    def evaluate(self, client_model, server_model):
        """客户端评估过程"""
        # 确保模型在正确的设备上
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        
        # 统计信息
        local_correct = 0
        global_correct = 0
        total = 0
        test_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        # 每个类别的准确率统计
        num_classes = 10  # 默认为10，可以根据需要调整
        if hasattr(server_model, 'num_classes'):
            num_classes = server_model.num_classes
            
        local_class_correct = [0] * num_classes
        local_class_total = [0] * num_classes
        global_class_correct = [0] * num_classes
        global_class_total = [0] * num_classes
        
        with torch.no_grad():
            for data, target in self.test_data:
                # 将数据移到设备上
                data, target = data.to(self.device), target.to(self.device)
                
                # 客户端前向传播
                local_logits, features = client_model(data)
                
                # 服务器前向传播
                try:
                    server_output = server_model(features, tier=self.tier)
                    if isinstance(server_output, tuple) and len(server_output) == 2:
                        global_logits, _ = server_output
                    else:
                        # 如果服务器模型只返回一个输出，假设它是global_logits
                        global_logits = server_output
                except Exception as e:
                    print(f"评估时服务器前向传播失败: {str(e)}")
                    # 使用local_logits作为备用
                    global_logits = local_logits
                
                # 计算损失
                loss = criterion(global_logits, target)
                test_loss += loss.item()
                
                # 计算本地准确率
                _, local_pred = local_logits.max(1)
                local_correct += local_pred.eq(target).sum().item()
                
                # 计算全局准确率
                _, global_pred = global_logits.max(1)
                global_correct += global_pred.eq(target).sum().item()
                
                # 更新总样本数
                total += target.size(0)
                
                # 更新每个类别的准确率统计
                for i in range(len(target)):
                    label = target[i].item()
                    if label < num_classes:
                        local_class_total[label] += 1
                        global_class_total[label] += 1
                        if local_pred[i] == label:
                            local_class_correct[label] += 1
                        if global_pred[i] == label:
                            global_class_correct[label] += 1
        
        # 计算平均损失和准确率
        test_loader_len = len(self.test_data)
        avg_loss = test_loss / max(1, test_loader_len)
        local_accuracy = 100.0 * local_correct / max(1, total)
        global_accuracy = 100.0 * global_correct / max(1, total)
        
        # 计算每个类别的准确率
        local_per_class_acc = [100.0 * correct / max(1, total) for correct, total in zip(local_class_correct, local_class_total)]
        global_per_class_acc = [100.0 * correct / max(1, total) for correct, total in zip(global_class_correct, global_class_total)]
        
        # 计算类别不平衡度
        if min(global_per_class_acc) > 0:
            global_imbalance = max(global_per_class_acc) / min(global_per_class_acc)
        else:
            global_imbalance = float('inf')
        
        return {
            'test_loss': avg_loss,
            'local_accuracy': local_accuracy,
            'global_accuracy': global_accuracy,
            'local_per_class_acc': local_per_class_acc,
            'global_per_class_acc': global_per_class_acc,
            'global_imbalance': global_imbalance
        }
    
    def update_alpha(self, alpha):
        """更新本地和全局损失的平衡因子"""
        self.alpha = alpha
        self.hybrid_loss.update_alpha(alpha)
    
    def update_lambda_feature(self, lambda_feature):
        """更新特征对齐损失权重"""
        self.lambda_feature = lambda_feature

# 客户端管理器
class TierHFLClientManager:
    def __init__(self):
        self.clients = {}
        self.default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def add_client(self, client_id, tier, train_data, test_data, 
                   device=None, lr=0.001, local_epochs=1):
        """添加客户端"""
        device = device or self.default_device
        
        client = TierHFLClient(
            client_id=client_id,
            tier=tier,
            train_data=train_data,
            test_data=test_data,
            device=device,
            lr=lr,
            local_epochs=local_epochs
        )
        
        self.clients[client_id] = client
        return client
    
    def get_client(self, client_id):
        """获取客户端"""
        return self.clients.get(client_id)
    
    def update_client_tier(self, client_id, new_tier):
        """更新客户端的tier级别"""
        if client_id in self.clients:
            self.clients[client_id].tier = new_tier
            return True
        return False
    
    def update_client_alpha(self, client_id, alpha):
        """更新客户端的alpha值"""
        if client_id in self.clients:
            self.clients[client_id].update_alpha(alpha)
            return True
        return False
    
    def update_client_feature_lambda(self, client_id, lambda_feature):
        """更新客户端的特征对齐损失权重"""
        if client_id in self.clients:
            self.clients[client_id].update_lambda_feature(lambda_feature)
            return True
        return False
    
    def update_all_clients_alpha(self, alpha):
        """更新所有客户端的alpha值"""
        for client in self.clients.values():
            client.update_alpha(alpha)
    
    def update_all_clients_feature_lambda(self, lambda_feature):
        """更新所有客户端的特征对齐损失权重"""
        for client in self.clients.values():
            client.update_lambda_feature(lambda_feature)

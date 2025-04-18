import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np
import math

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

# 特征对齐损失 - 处理不同维度的特征
class EnhancedFeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super(EnhancedFeatureAlignmentLoss, self).__init__()
        
    def forward(self, client_features, server_features, round_idx=0):
        """改进的特征对齐损失计算"""
        # 添加调试模式
        debug_mode = hasattr(self, '_debug_client_id') and self._debug_client_id == 6
        
        if debug_mode:
            print(f"\n[Feature Loss DEBUG] 客户端特征形状: {client_features.shape}")
            print(f"[Feature Loss DEBUG] 服务器特征形状: {server_features.shape}")
        
        # 确保特征是4D或2D张量
        if len(client_features.shape) == 4:  # 4D: [B, C, H, W]
            batch_size = client_features.size(0)
            client_pooled = F.adaptive_avg_pool2d(client_features, (1, 1))
            client_features = client_pooled.view(batch_size, -1)
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 池化后客户端特征形状: {client_features.shape}")
        
        if len(server_features.shape) == 4:  # 4D: [B, C, H, W]
            batch_size = server_features.size(0)
            server_pooled = F.adaptive_avg_pool2d(server_features, (1, 1))
            server_features = server_pooled.view(batch_size, -1)
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 池化后服务器特征形状: {server_features.shape}")
        
        # 统一特征维度
        if client_features.size(1) != server_features.size(1):
            if debug_mode:
                print(f"[Feature Loss DEBUG] 特征维度不匹配! 客户端: {client_features.size(1)}, 服务器: {server_features.size(1)}")
            
            target_dim = min(client_features.size(1), server_features.size(1))
            
            if client_features.size(1) > target_dim:
                client_features = client_features[:, :target_dim]
            
            if server_features.size(1) > target_dim:
                server_features = server_features[:, :target_dim]
                
            if debug_mode:
                print(f"[Feature Loss DEBUG] 调整后维度: {target_dim}")
        
        # 标准化特征向量并检测异常值
        try:
            client_norm = F.normalize(client_features, dim=1)
            server_norm = F.normalize(server_features, dim=1)
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 客户端归一化后是否有NaN: {torch.isnan(client_norm).any().item()}")
                print(f"[Feature Loss DEBUG] 服务器归一化后是否有NaN: {torch.isnan(server_norm).any().item()}")
            
            # 余弦相似度
            cosine_sim = torch.mean(torch.sum(client_norm * server_norm, dim=1))
            cosine_loss = 1.0 - cosine_sim
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 余弦相似度: {cosine_sim.item():.4f}")
                print(f"[Feature Loss DEBUG] 特征对齐损失: {cosine_loss.item():.4f}")
        except Exception as e:
            if debug_mode:
                print(f"[Feature Loss DEBUG] 计算特征对齐损失出错: {str(e)}")
            # 出错时返回一个默认损失值
            return torch.tensor(1.0, device=client_features.device)
        
        # 随训练轮次渐进增强特征对齐强度
        alignment_weight = min(0.8, 0.2 + round_idx/100)
        
        return cosine_loss * alignment_weight

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
        self.feature_alignment_loss = EnhancedFeatureAlignmentLoss()
        
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
        
    def train(self, client_model, server_model, global_classifier, round_idx=0):
        """客户端训练过程-- 拆分学习模式"""
        # 确保模型在正确的设备上
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        global_classifier = global_classifier.to(self.device)
        
        # 设置为训练模式
        client_model.train()
        server_model.train()
        global_classifier.train()
        

        # 优化器设置部分
        if hasattr(client_model, 'get_shared_params') and hasattr(client_model, 'get_personalized_params'):
            shared_params = list(client_model.get_shared_params().values())
            personalized_params = list(client_model.get_personalized_params().values())
            
            # 大幅提高分类器学习率强化个性化
            classifier_lr_multiplier = 4.0
            if self.tier <= 3:  # 高性能设备可用更激进学习率
                classifier_lr_multiplier = 5.0
            
            classifier_lr = self.lr * classifier_lr_multiplier
            
            # 优化特征提取部分
            optimizer_shared = torch.optim.Adam(
                shared_params, 
                lr=self.lr,
                weight_decay=1e-4
            )
            
            # 优化个性化分类器部分
            optimizer_personal = torch.optim.AdamW(  # 从SGD改为AdamW
                personalized_params, 
                lr=classifier_lr,
                weight_decay=2e-4  # 更强的正则化
            )
            
            if self.client_id % 10 == 0:  # 避免过多日志，只显示部分客户端信息
                print(f"客户端 {self.client_id} - 特征学习率: {self.lr:.6f}, 分类器学习率: {classifier_lr:.6f}")
        else:
            # 后备方案
            optimizer_shared = torch.optim.Adam(client_model.parameters(), lr=self.lr)
            optimizer_personal = None
        
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
        # 初始化早停相关变量
        early_stop_patience = max(5, 10 - round_idx // 10)  # 随着训练进行减少耐心
        best_loss = float('inf')
        patience_counter = 0
        min_epochs = 2  # 最少训练轮次
        initial_loss = float('inf')
        # 训练循环
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, (data, target) in enumerate(self.train_data):
                batch_start = time.time()
                
                # 将数据移到设备上
                data, target = data.to(self.device), target.to(self.device)
                
                # 清除梯度
                optimizer_shared.zero_grad()
                if optimizer_personal:
                    optimizer_personal.zero_grad()
                
                # 客户端前向传播
                try:
                    local_logits, features = client_model(data)
                except Exception as e:
                    print(f"客户端模型前向传播失败: {str(e)}")
                    continue
                
                # 服务器前向传播 - 只得到特征
                try:
                    server_features = server_model(features, tier=self.tier)
                    
                    # 全局分类器前向传播
                    global_logits = global_classifier(server_features)
                except Exception as e:
                    print(f"服务器特征提取或全局分类器前向传播失败: {str(e)}, 特征形状: {features.shape}")
                    # 使用local_logits作为备用
                    global_logits = local_logits
                    server_features = features.view(features.size(0), -1)  # 确保是2D张量
                
                # 计算混合损失
                loss, local_loss, global_loss = self.hybrid_loss(local_logits, global_logits, target)
                
                # 计算特征对齐损失
                try:
                    # 本地特征与服务器特征对齐
                    # 需要确保特征维度匹配，可能需要调整
                    feature_loss = self.feature_alignment_loss(features, server_features, round_idx)
                    total_loss = loss + self.lambda_feature * feature_loss
                except Exception as e:
                    print(f"特征对齐损失计算失败: {str(e)}")
                    total_loss = loss
                    feature_loss = torch.tensor(0.0, device=self.device)
                
                epoch_loss += total_loss.item()
                num_batches += 1

                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer_shared.step()
                if optimizer_personal:
                    optimizer_personal.step()
                
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
            
            # 计算每轮平均损失
            if num_batches > 0:
                epoch_loss = epoch_loss / num_batches
                print(f"客户端 {self.client_id} - 轮次 {epoch+1}/{self.local_epochs}, 损失: {epoch_loss:.4f}")
    
            # 记录每个epoch的损失
            epoch_losses.append(epoch_stats['total_loss'] / epoch_stats['batch_count'])
            epoch_local_losses.append(epoch_stats['local_loss'] / epoch_stats['batch_count'])
            epoch_global_losses.append(epoch_stats['global_loss'] / epoch_stats['batch_count'])
            epoch_feature_losses.append(epoch_stats['feature_loss'] / epoch_stats['batch_count'])

            # 检查早停条件
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                
                # 记录初始损失
                if epoch == 0:
                    initial_loss = avg_epoch_loss
                
                # 记录最佳损失
                is_best = avg_epoch_loss < best_loss * 0.995  # 至少需要0.5%的改进
                if is_best:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # 决定是否早停
                if patience_counter >= early_stop_patience and epoch >= min_epochs:
                    relative_improvement = (initial_loss - best_loss) / initial_loss
                    
                    # 根据相对改进决定是否继续训练
                    if relative_improvement > 0.05:  # 至少5%的改进
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
        
    def evaluate(self, client_model, server_model, global_classifier):
        """客户端评估过程"""
        # 确保模型在正确的设备上
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)
        global_classifier = global_classifier.to(self.device)
        # 添加调试标志
        is_client6 = (self.client_id == 6)
        # 设置为评估模式
        client_model.eval()
        server_model.eval()
        global_classifier.eval()
        
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
        
        # 添加收集全局分类器预测的功能
        global_predictions = []
        with torch.no_grad():
            for data, target in self.test_data:
                batch_count = 0  # 添加批次计数
                # 将数据移到设备上
                data, target = data.to(self.device), target.to(self.device)
                
                # 客户端前向传播
                local_logits, features = client_model(data)

                # 只对前几个批次进行详细调试，避免信息过多
                do_debug = is_client6 and batch_count < 3
                
                if do_debug:
                    print(f"\n[Client6 Debug] 批次 {batch_count} 客户端特征:")
                    print(f"- 形状: {features.shape}")
                    print(f"- 范围: {features.min().item():.4f} 到 {features.max().item():.4f}")
                    
                    # 检查特征是否有异常值
                    has_nan = torch.isnan(features).any().item()
                    has_inf = torch.isinf(features).any().item()
                    print(f"- 特征有NaN: {has_nan}, 有Inf: {has_inf}")
                
                # 服务器前向传播
                try:
                    server_features = server_model(features, tier=self.tier)

                    if do_debug:
                        print(f"[Client6 Debug] 服务器处理后特征:")
                        print(f"- 形状: {server_features.shape}")
                        print(f"- 范围: {server_features.min().item():.4f} 到 {server_features.max().item():.4f}")
                        print(f"- 均值: {server_features.mean().item():.4f}")
                        print(f"- 标准差: {server_features.std().item():.4f}")
                        
                        # 检查特征是否有异常值
                        has_nan = torch.isnan(server_features).any().item()
                        has_inf = torch.isinf(server_features).any().item()
                        print(f"- 特征有NaN: {has_nan}, 有Inf: {has_inf}")



                    global_logits = global_classifier(server_features)

                    if do_debug:
                        print(f"[Client6 Debug] 全局分类器输出:")
                        print(f"- 形状: {global_logits.shape}")
                        print(f"- 范围: {global_logits.min().item():.4f} 到 {global_logits.max().item():.4f}")
                        
                        # 检查是否所有样本预测相同类别
                        pred_classes = torch.argmax(global_logits, dim=1)
                        unique_preds = torch.unique(pred_classes)
                        print(f"- 预测的类别: {pred_classes[:5].cpu().numpy()}")
                        print(f"- 不同类别数量: {len(unique_preds)}")
                        print(f"- 预测概率分布: {torch.softmax(global_logits[0], dim=0).cpu().numpy().round(3)}")
                        
                        # 如果都预测同一个类别，这是问题的关键
                        if len(unique_preds) == 1:
                            print(f"!!! 警告: 全局分类器总是预测同一类别: {unique_preds.item()} !!!")
                except Exception as e:
                    if is_client6:
                        print(f"[Client6 Error] 服务器处理或全局分类时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    # 使用local_logits作为备用
                    global_logits = local_logits
                
                batch_count += 1
                
                # 计算损失
                loss = criterion(global_logits, target)
                test_loss += loss.item()
                
                # 计算本地准确率
                _, local_pred = local_logits.max(1)
                local_correct += local_pred.eq(target).sum().item()
                
                # 计算全局准确率
                _, global_pred = global_logits.max(1)
                global_correct += global_pred.eq(target).sum().item()

                # 收集预测结果
                global_predictions.extend(global_pred.cpu().numpy().tolist())
                
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
            'global_imbalance': global_imbalance,
            'global_predictions': global_predictions  # 添加预测结果收集
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
        
    # 在客户端管理器中添加监控方法
    def add_client(self, client_id, tier, train_data, test_data, device=None, lr=0.001, local_epochs=1):
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
        
        # 针对客户端6添加特殊监控
        if client_id == 6:
            print(f"\n[CLIENT MANAGER] 注册客户端6 - Tier: {tier}")
            print(f"[CLIENT MANAGER] 客户端6训练集样本数: {len(train_data.dataset)}")
            print(f"[CLIENT MANAGER] 客户端6测试集样本数: {len(test_data.dataset)}")
            
            # 检查数据集分布
            try:
                # 获取前5个样本的标签
                print("[CLIENT MANAGER] 分析客户端6数据集...")
                sample_labels = []
                for i, (_, labels) in enumerate(train_data):
                    sample_labels.extend(labels.tolist())
                    if i >= 2:  # 只检查前几个批次
                        break
                
                # 统计标签分布
                label_counts = {}
                for label in sample_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                    
                print(f"[CLIENT MANAGER] 客户端6训练集标签分布(部分): {label_counts}")
            except Exception as e:
                print(f"[CLIENT MANAGER] 分析客户端6数据时出错: {str(e)}")
        
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

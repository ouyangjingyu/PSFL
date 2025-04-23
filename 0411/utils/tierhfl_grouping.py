import numpy as np
import torch
from collections import defaultdict
import time
import copy

class TierHFLGroupingStrategy:
    """TierHFL客户端分组策略"""
    def __init__(self, num_groups=3):
        self.num_groups = num_groups
        self.grouping_history = []
        self.current_grouping = {}
        self.similarity_matrices = {
            'data': None,  # 数据分布相似度
            'feature': None,  # 特征相似度
            'tier': None  # Tier相似度
        }
        # 相似度权重（初始值）
        self.weights = {
            'data': 0.6,
            'feature': 0.3,
            'tier': 0.1
        }
    
    def update_weights(self, round_idx):
        """根据训练轮次更新权重"""
        if round_idx < 20:
            # 初期阶段，强调数据分布相似度
            self.weights = {
                'data': 0.6,
                'feature': 0.3,
                'tier': 0.1
            }
        elif round_idx < 50:
            # 中期阶段，加强特征相似度权重
            self.weights = {
                'data': 0.4,
                'feature': 0.5,
                'tier': 0.1
            }
        else:
            # 后期阶段，进一步加强特征相似度
            self.weights = {
                'data': 0.3,
                'feature': 0.6,
                'tier': 0.1
            }
    
    def calculate_data_similarity(self, client_manager):
        """计算数据分布相似度矩阵"""
        similarity_matrix, client_ids = client_manager.calculate_distribution_similarity_matrix()
        self.similarity_matrices['data'] = {
            'matrix': similarity_matrix,
            'client_ids': client_ids
        }
        return similarity_matrix, client_ids
    
    def calculate_feature_similarity(self, client_models, device='cuda'):
        """计算模型特征相似度矩阵"""
        client_ids = list(client_models.keys())
        n_clients = len(client_ids)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        # 提取每个客户端模型的特征
        model_features = {}
        for client_id, model in client_models.items():
            # 提取共享基础层的特征
            shared_params = []
            for name, param in model.named_parameters():
                if 'shared_base' in name:
                    shared_params.append(param.detach().cpu().view(-1))
            
            if shared_params:
                # 连接所有参数
                model_features[client_id] = torch.cat(shared_params).numpy()
        
        # 计算相似度
        for i, id1 in enumerate(client_ids):
            for j, id2 in enumerate(client_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自身相似度为1
                else:
                    if id1 in model_features and id2 in model_features:
                        f1 = model_features[id1]
                        f2 = model_features[id2]
                        
                        # 归一化
                        f1_norm = np.linalg.norm(f1)
                        f2_norm = np.linalg.norm(f2)
                        
                        if f1_norm > 0 and f2_norm > 0:
                            # 余弦相似度
                            similarity_matrix[i, j] = np.dot(f1, f2) / (f1_norm * f2_norm)
                        else:
                            similarity_matrix[i, j] = 0.0
                    else:
                        similarity_matrix[i, j] = 0.0
        
        self.similarity_matrices['feature'] = {
            'matrix': similarity_matrix,
            'client_ids': client_ids
        }
        
        return similarity_matrix, client_ids
    
    def calculate_tier_similarity(self, client_manager):
        """计算Tier相似度矩阵"""
        clients = client_manager.get_all_clients()
        client_ids = list(clients.keys())
        n_clients = len(client_ids)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        # 获取客户端tier值
        tiers = {}
        for client_id, client in clients.items():
            tiers[client_id] = client.tier
        
        # 计算tier差距
        min_tier = min(tiers.values())
        max_tier = max(tiers.values())
        tier_range = max(1, max_tier - min_tier)
        
        for i, id1 in enumerate(client_ids):
            for j, id2 in enumerate(client_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自身相似度为1
                else:
                    # tier差距越小，相似度越高
                    tier_diff = abs(tiers[id1] - tiers[id2])
                    similarity_matrix[i, j] = 1.0 - (tier_diff / tier_range)
        
        self.similarity_matrices['tier'] = {
            'matrix': similarity_matrix,
            'client_ids': client_ids
        }
        
        return similarity_matrix, client_ids
    
    def calculate_combined_similarity(self):
        """计算综合相似度矩阵"""
        # 确保所有相似度矩阵都已计算
        if any(v is None for v in self.similarity_matrices.values()):
            raise ValueError("请先计算所有相似度矩阵")
        
        # 获取客户端ID
        client_ids = self.similarity_matrices['data']['client_ids']
        n_clients = len(client_ids)
        
        # 初始化综合相似度矩阵
        combined_matrix = np.zeros((n_clients, n_clients))
        
        # 综合所有相似度矩阵
        for sim_type, weight in self.weights.items():
            sim_data = self.similarity_matrices[sim_type]
            combined_matrix += weight * sim_data['matrix']
        
        return combined_matrix, client_ids
    
    def assign_groups(self, similarity_matrix, client_ids):
        """基于相似度矩阵分配客户端到组"""
        n_clients = len(client_ids)
        
        # 确保组数不超过客户端数
        num_groups = min(self.num_groups, n_clients)
        
        # 初始选择最远的num_groups个客户端作为中心
        centers = []
        remaining = list(range(n_clients))
        
        # 选择第一个中心（随机选择）
        first_center = np.random.choice(remaining)
        centers.append(first_center)
        remaining.remove(first_center)
        
        # 选择其他中心（最远点策略）
        for _ in range(1, num_groups):
            if not remaining:
                break
                
            # 计算每个剩余点到最近中心的距离
            distances = []
            for idx in remaining:
                min_dist = min(1 - similarity_matrix[idx, center] for center in centers)
                distances.append((idx, min_dist))
            
            # 选择距离最大的点作为新中心
            next_center = max(distances, key=lambda x: x[1])[0]
            centers.append(next_center)
            remaining.remove(next_center)
        
        # 分配其他客户端到最相似的中心
        grouping = {i: [] for i in range(num_groups)}
        for i, center_idx in enumerate(centers):
            grouping[i].append(client_ids[center_idx])
        
        for idx in remaining:
            # 计算到每个中心的相似度
            similarities = [(i, similarity_matrix[idx, center]) 
                           for i, center in enumerate(centers)]
            
            # 分配到最相似的中心
            best_group, _ = max(similarities, key=lambda x: x[1])
            grouping[best_group].append(client_ids[idx])
        
        return grouping
    
    def group_clients(self, client_manager, client_models, round_idx):
        """对客户端进行分组
        
        Args:
            client_manager: 客户端管理器
            client_models: 客户端模型字典
            round_idx: 当前训练轮次
            
        Returns:
            客户端分组
        """
        # 更新权重
        self.update_weights(round_idx)
        
        # 计算各种相似度矩阵
        self.calculate_data_similarity(client_manager)
        self.calculate_feature_similarity(client_models)
        self.calculate_tier_similarity(client_manager)
        
        # 计算综合相似度
        combined_matrix, client_ids = self.calculate_combined_similarity()
        
        # 分配客户端到组
        grouping = self.assign_groups(combined_matrix, client_ids)
        
        # 存储分组历史
        self.grouping_history.append({
            'round': round_idx,
            'grouping': copy.deepcopy(grouping),
            'weights': copy.deepcopy(self.weights),
            'timestamp': time.time()
        })
        
        # 更新当前分组
        self.current_grouping = grouping
        
        return grouping
    
    def get_current_grouping(self):
        """获取当前分组"""
        return self.current_grouping
    
    def get_grouping_history(self):
        """获取分组历史"""
        return self.grouping_history
    
    def get_client_group(self, client_id):
        """获取客户端所在的组"""
        for group_id, clients in self.current_grouping.items():
            if client_id in clients:
                return group_id
        return None
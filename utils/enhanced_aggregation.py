import torch
import copy
import numpy as np
from collections import OrderedDict, defaultdict
import torch.nn as nn
import logging

# 导入设备管理器
from utils.device_manager import DeviceManager

class EnhancedAggregator:
    """增强的模型聚合器，支持特征提取层聚合和设备管理"""
    
    def __init__(self, device_manager=None):
        """
        初始化聚合器
        
        Args:
            device_manager: 设备管理器实例，如果为None则自动创建
        """
        self.device_manager = device_manager or DeviceManager()
        self.logger = logging.getLogger("EnhancedAggregator")
        self.logger.setLevel(logging.INFO)
        
        # 初始化聚合设备
        self.aggregation_device = self.device_manager.select_aggregation_device()
        
        self.logger.info(f"增强聚合器初始化完成，聚合设备: {self.aggregation_device}")
    
    def aggregate_parameters(self, client_params, client_weights, only_feature_extraction=True):
        """
        聚合客户端参数
        
        Args:
            client_params: 客户端参数字典，键为客户端ID，值为参数字典
            client_weights: 客户端权重字典，键为客户端ID，值为权重
            only_feature_extraction: 是否只聚合特征提取层
            
        Returns:
            聚合后的参数字典
        """
        if not client_params:
            self.logger.warning("没有客户端参数可聚合")
            return {}
        
        # 准备参数用于聚合，确保设备一致性
        agg_params = self.device_manager.get_client_models_params_for_aggregation(
            client_params, 
            extract_feature_only=only_feature_extraction
        )
        
        # 检查是否有可聚合的参数
        if not agg_params:
            self.logger.warning("没有可聚合的参数")
            return {}
        
        # 获取所有参数的键集合
        all_keys = set()
        for params in agg_params.values():
            all_keys.update(params.keys())
        
        # 创建聚合结果字典
        result = {}
        
        # 逐参数聚合
        for key in all_keys:
            # 收集所有拥有该参数的客户端参数和权重
            key_params = []
            key_weights = []
            
            for client_id, params in agg_params.items():
                if key in params:
                    key_params.append(params[key])
                    key_weights.append(client_weights.get(client_id, 1.0))
            
            # 如果没有客户端拥有该参数，跳过
            if not key_params:
                continue
            
            # 归一化权重
            weight_sum = sum(key_weights)
            if weight_sum > 0:
                normalized_weights = [w / weight_sum for w in key_weights]
            else:
                normalized_weights = [1.0 / len(key_weights)] * len(key_weights)
            
            # 特殊处理GroupNorm参数
            if 'gn' in key:
                if 'weight' in key or 'bias' in key:
                    # 对于GroupNorm的权重和偏置，使用加权平均
                    weighted_param = torch.zeros_like(key_params[0], dtype=torch.float32, device=self.aggregation_device)
                    for param, weight in zip(key_params, normalized_weights):
                        weighted_param += param.to(dtype=torch.float32, device=self.aggregation_device) * weight
                    result[key] = weighted_param
            else:
                # 其他参数使用加权平均
                weighted_param = torch.zeros_like(key_params[0], dtype=torch.float32, device=self.aggregation_device)
                for param, weight in zip(key_params, normalized_weights):
                    weighted_param += param.to(dtype=torch.float32, device=self.aggregation_device) * weight
                result[key] = weighted_param
        
        self.logger.info(f"聚合完成，共聚合 {len(result)} 个参数")
        return result
    
    def cluster_aware_aggregation(self, client_params, client_weights, cluster_map, client_tiers=None):
        """
        集群感知的聚合，先在聚类内聚合，再聚合聚类结果
        
        Args:
            client_params: 客户端参数字典
            client_weights: 客户端权重字典
            cluster_map: 聚类映射，键为聚类ID，值为客户端ID列表
            client_tiers: 客户端tier信息，键为客户端ID，值为tier级别
            
        Returns:
            聚类聚合结果字典，键为聚类ID，值为聚合后的参数
        """
        # 聚类聚合结果
        cluster_models = {}
        
        # 对每个聚类进行内部聚合
        for cluster_id, client_ids in cluster_map.items():
            # 选择该聚类的客户端
            cluster_client_params = {cid: client_params[cid] for cid in client_ids if cid in client_params}
            cluster_client_weights = {cid: client_weights.get(cid, 1.0) for cid in client_ids}
            
            # 如果该聚类没有客户端，跳过
            if not cluster_client_params:
                continue
            
            # 执行聚合
            cluster_agg_result = self.aggregate_parameters(
                cluster_client_params, 
                cluster_client_weights, 
                only_feature_extraction=True
            )
            
            # 保存聚合结果
            cluster_models[cluster_id] = cluster_agg_result
        
        self.logger.info(f"聚类聚合完成，共聚合 {len(cluster_models)} 个聚类")
        return cluster_models
    
    def global_aggregation(self, cluster_models, cluster_weights):
        """
        全局聚合，聚合所有聚类结果
        
        Args:
            cluster_models: 聚类模型字典，键为聚类ID，值为参数字典
            cluster_weights: 聚类权重字典，键为聚类ID，值为权重
            
        Returns:
            全局聚合结果
        """
        # 如果没有聚类模型，返回空字典
        if not cluster_models:
            self.logger.warning("没有聚类模型可聚合")
            return {}
        
        # 执行全局聚合
        global_model = self.aggregate_parameters(
            cluster_models, 
            cluster_weights, 
            only_feature_extraction=True
        )
        
        self.logger.info(f"全局聚合完成，共聚合 {len(global_model)} 个参数")
        return global_model
    
    def enhanced_hierarchical_aggregation(self, client_params, client_weights, cluster_map, 
                                         client_tiers=None, global_model_template=None, 
                                         num_classes=10, device=None):
        """
        增强的层次聚合方法，解决设备不一致和类型转换问题
        
        Args:
            client_params: 客户端参数字典
            client_weights: 客户端权重字典
            cluster_map: 聚类映射
            client_tiers: 客户端tier信息（可选）
            global_model_template: 全局模型模板（可选）
            num_classes: 类别数量
            device: 指定设备（可选）
            
        Returns:
            global_model: 全局聚合结果
            cluster_models: 聚类聚合结果
            aggregation_log: 聚合日志
        """
        # 设置聚合设备
        if device:
            self.aggregation_device = device
            self.device_manager.aggregation_device = device
        else:
            self.aggregation_device = self.device_manager.select_aggregation_device()
        
        self.logger.info(f"层次聚合使用设备: {self.aggregation_device}")
        
        # 初始化聚合日志
        aggregation_log = {
            'device': str(self.aggregation_device),
            'cluster_stats': {},
            'client_stats': {}
        }
        
        # 记录聚类统计信息
        for cluster_id, client_ids in cluster_map.items():
            aggregation_log['cluster_stats'][f'cluster_{cluster_id}'] = {
                'size': len(client_ids),
                'clients': client_ids
            }
        
        # 1. 执行聚类内聚合
        cluster_models = self.cluster_aware_aggregation(
            client_params, 
            client_weights, 
            cluster_map, 
            client_tiers
        )
        
        # 2. 计算聚类权重，基于聚类大小
        cluster_weights = {}
        total_clients = sum(len(clients) for clients in cluster_map.values())
        
        if total_clients > 0:
            for cluster_id, clients in cluster_map.items():
                cluster_weights[cluster_id] = len(clients) / total_clients
        else:
            # 避免除零错误
            n_clusters = max(1, len(cluster_map))
            for cluster_id in cluster_map.keys():
                cluster_weights[cluster_id] = 1.0 / n_clusters
        
        # 3. 执行全局聚合
        global_model = self.global_aggregation(
            cluster_models, 
            cluster_weights
        )
        
        # 4. 如果提供了全局模型模板，确保参数完整性
        if global_model_template is not None:
            # 复制模板中存在但聚合结果中不存在的参数
            template_state_dict = global_model_template.state_dict() if hasattr(global_model_template, 'state_dict') else global_model_template
            
            for key, param in template_state_dict.items():
                if key not in global_model:
                    # 只复制非特征提取层参数
                    if any(substr in key for substr in ['classifier', 'projection', 'fc']):
                        global_model[key] = param.clone().to(self.aggregation_device)
            
            # 记录使用了多少模板键
            template_keys_used = len(template_state_dict.keys()) - len(global_model.keys())
            aggregation_log['template_keys_used'] = template_keys_used
        
        # 5. 验证和修复参数
        for key, param in global_model.items():
            # 检查NaN值
            if torch.isnan(param).any():
                global_model[key] = torch.nan_to_num(param, nan=0.0)
                aggregation_log['repair_nan'] = aggregation_log.get('repair_nan', 0) + torch.isnan(param).sum().item()
            
            # 检查Inf值
            if torch.isinf(param).any():
                global_model[key] = torch.nan_to_num(param, posinf=1e6, neginf=-1e6)
                aggregation_log['repair_inf'] = aggregation_log.get('repair_inf', 0) + torch.isinf(param).sum().item()
        
        # 清理缓存，释放内存
        self.device_manager.clear_cuda_cache()
        
        return global_model, cluster_models, aggregation_log
    
    def update_client_models(self, client_models_dict, global_model=None, cluster_models=None, cluster_map=None):
        """
        使用聚合结果更新客户端模型
        
        Args:
            client_models_dict: 客户端模型字典
            global_model: 全局聚合模型（可选）
            cluster_models: 聚类聚合模型（可选）
            cluster_map: 聚类映射（可选，用于确定客户端所属聚类）
            
        Returns:
            更新后的客户端模型字典
        """
        # 如果没有聚类模型，使用全局模型更新所有客户端
        if not cluster_models and global_model:
            return self.device_manager.restore_aggregated_model_params(
                global_model, 
                client_models_dict
            )
        
        # 如果没有聚类映射，无法使用聚类模型更新
        if not cluster_map:
            return client_models_dict
        
        # 更新每个客户端模型
        updated_models = {}
        
        for client_id, model in client_models_dict.items():
            # 确定客户端所属聚类
            client_cluster_id = None
            for cluster_id, clients in cluster_map.items():
                if client_id in clients:
                    client_cluster_id = cluster_id
                    break
            
            # 如果找到所属聚类且聚类模型存在，使用聚类模型更新
            if client_cluster_id is not None and client_cluster_id in cluster_models:
                # 创建模型副本
                updated_model = copy.deepcopy(model)
                
                # 获取当前状态字典
                if hasattr(updated_model, 'state_dict'):
                    current_state_dict = updated_model.state_dict()
                    
                    # 获取聚类模型
                    cluster_model = cluster_models[client_cluster_id]
                    
                    # 将聚类模型参数移至客户端设备
                    device_params = self.device_manager.to_original_device(
                        cluster_model, 
                        client_id, 
                        restore_type=True
                    )
                    
                    # 只更新特征提取层参数
                    for name, param in device_params.items():
                        if name in current_state_dict:
                            current_state_dict[name] = param
                    
                    # 加载更新后的状态字典
                    updated_model.load_state_dict(current_state_dict)
                    
                    # 保存更新后的模型
                    updated_models[client_id] = updated_model
                else:
                    # 对于非标准模型，跳过
                    updated_models[client_id] = model
            else:
                # 如果没有找到所属聚类或聚类模型不存在，保持原模型
                updated_models[client_id] = model
        
        return updated_models

# 兼容旧代码的函数接口
def enhanced_hierarchical_aggregation_no_projection(client_models_params, client_weights, client_clusters, 
                                                 client_tiers=None, global_model_template=None, 
                                                 num_classes=10, device=None):
    """
    增强的层次聚合函数接口，兼容旧代码
    
    Args:
        client_models_params: 客户端模型参数字典
        client_weights: 客户端权重字典
        client_clusters: 客户端聚类映射
        client_tiers: 客户端tier信息（可选）
        global_model_template: 全局模型模板（可选）
        num_classes: 类别数量
        device: 聚合设备（可选）
        
    Returns:
        global_model: 全局聚合结果
        cluster_models: 聚类聚合结果
        aggregation_log: 聚合日志
    """
    # 创建聚合器
    aggregator = EnhancedAggregator(DeviceManager())
    
    # 执行层次聚合
    return aggregator.enhanced_hierarchical_aggregation(
        client_models_params, 
        client_weights, 
        client_clusters, 
        client_tiers, 
        global_model_template, 
        num_classes, 
        device
    )
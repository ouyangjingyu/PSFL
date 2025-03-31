import torch
import logging
from collections import defaultdict

class DeviceManager:
    """设备管理器，负责管理不同设备上的模型参数，确保聚合过程中的设备一致性"""
    
    def __init__(self, default_device=None):
        """
        初始化设备管理器
        
        Args:
            default_device: 默认设备，如果为None则自动选择可用的最佳设备
        """
        self.default_device = default_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_map = {}  # 跟踪客户端/参数所在的设备
        self.logger = logging.getLogger("DeviceManager")
        self.logger.setLevel(logging.INFO)
        
        # 为聚合选择的设备
        self.aggregation_device = self.default_device
        
        # 跟踪设备使用
        self.device_usage = defaultdict(int)
        
        # 如果有多个GPU，跟踪GPU内存使用情况
        self.gpu_memory_usage = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.gpu_memory_usage[f'cuda:{i}'] = 0
        
        self.logger.info(f"设备管理器初始化完成，默认设备: {self.default_device}")
        
    def register_client_device(self, client_id, device):
        """
        注册客户端使用的设备
        
        Args:
            client_id: 客户端ID
            device: 设备名称或对象
        """
        device_str = str(device)
        self.device_map[client_id] = device_str
        self.device_usage[device_str] += 1
        self.logger.debug(f"客户端 {client_id} 注册到设备 {device_str}")
        
    def get_client_device(self, client_id):
        """
        获取客户端使用的设备
        
        Args:
            client_id: 客户端ID
            
        Returns:
            设备名称
        """
        return self.device_map.get(client_id, self.default_device)
    
    def select_aggregation_device(self):
        """
        为聚合过程选择最佳设备
        
        Returns:
            选定的设备名称
        """
        # 如果没有CUDA，使用CPU
        if not torch.cuda.is_available():
            self.aggregation_device = 'cpu'
            return 'cpu'
        
        # 选择内存最充足的GPU
        best_device = 'cuda:0'
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device = device
        
        self.aggregation_device = best_device
        self.logger.info(f"为聚合过程选择设备: {best_device}")
        return best_device
    
    def to_aggregation_device(self, data, data_type=None):
        """
        将数据移至聚合设备
        
        Args:
            data: 要移动的数据（可以是张量、字典、列表等）
            data_type: 数据原始类型，如果为None则保持原类型
            
        Returns:
            移动到聚合设备后的数据
        """
        # 确保聚合设备已选择
        if not self.aggregation_device:
            self.select_aggregation_device()
            
        # 处理张量
        if isinstance(data, torch.Tensor):
            # 记录原始数据类型
            original_dtype = data.dtype if data_type is None else data_type
            
            # 移至聚合设备
            result = data.to(device=self.aggregation_device, dtype=torch.float32)
            
            # 记录类型转换信息，用于稍后恢复
            result.original_dtype = original_dtype
            return result
            
        # 处理字典
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self.to_aggregation_device(value, data_type)
            return result
            
        # 处理列表
        elif isinstance(data, list):
            return [self.to_aggregation_device(item, data_type) for item in data]
            
        # 其他类型不处理
        else:
            return data
    
    def to_original_device(self, data, client_id=None, restore_type=True):
        """
        将数据从聚合设备移回原始设备
        
        Args:
            data: 要移动的数据
            client_id: 客户端ID，用于确定目标设备，如果为None则使用默认设备
            restore_type: 是否恢复原始数据类型
            
        Returns:
            移动回原始设备的数据
        """
        # 确定目标设备
        target_device = self.device_map.get(client_id, self.default_device) if client_id is not None else self.default_device
        
        # 处理张量
        if isinstance(data, torch.Tensor):
            # 确定目标数据类型
            target_dtype = getattr(data, 'original_dtype', data.dtype) if restore_type else data.dtype
            
            # 移至目标设备并恢复数据类型
            return data.to(device=target_device, dtype=target_dtype)
            
        # 处理字典
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self.to_original_device(value, client_id, restore_type)
            return result
            
        # 处理列表
        elif isinstance(data, list):
            return [self.to_original_device(item, client_id, restore_type) for item in list]
            
        # 其他类型不处理
        else:
            return data
    
    def synchronize_model_params(self, model_params, target_device=None):
        """
        确保模型参数在同一设备上
        
        Args:
            model_params: 模型参数字典
            target_device: 目标设备，如果为None则使用聚合设备
            
        Returns:
            同步后的模型参数
        """
        # 确定目标设备
        target_device = target_device or self.aggregation_device
        
        # 创建新的参数字典，确保所有参数在同一设备上
        synced_params = {}
        
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor):
                # 记录原始数据类型
                original_dtype = param.dtype
                
                # 移至目标设备
                synced_param = param.to(device=target_device, dtype=torch.float32)
                
                # 添加原始数据类型属性
                synced_param.original_dtype = original_dtype
                synced_params[name] = synced_param
            else:
                synced_params[name] = param
        
        return synced_params
    
    def extract_feature_extraction_params(self, model_params):
        """
        提取特征提取层的参数
        
        Args:
            model_params: 模型参数字典
            
        Returns:
            只包含特征提取层参数的字典
        """
        # 特征提取层名称标识
        feature_extraction_keys = ['conv', 'gn', 'layer', 'downsample']
        non_feature_extraction_keys = ['classifier', 'projection', 'fc']
        
        # 提取特征提取层参数
        feature_params = {}
        
        for name, param in model_params.items():
            # 检查是否属于特征提取层
            is_feature_layer = any(key in name for key in feature_extraction_keys)
            is_non_feature_layer = any(key in name for key in non_feature_extraction_keys)
            
            if is_feature_layer and not is_non_feature_layer:
                feature_params[name] = param
        
        return feature_params
    
    def get_client_models_params_for_aggregation(self, client_models_dict, extract_feature_only=True):
        """
        准备客户端模型参数用于聚合，确保设备一致性
        
        Args:
            client_models_dict: 客户端模型字典，键为客户端ID，值为模型
            extract_feature_only: 是否只提取特征提取层参数
            
        Returns:
            准备好用于聚合的模型参数字典
        """
        # 选择聚合设备
        self.select_aggregation_device()
        
        # 准备聚合参数
        aggregation_params = {}
        
        for client_id, model in client_models_dict.items():
            # 获取模型参数
            if hasattr(model, 'state_dict'):
                model_params = model.state_dict()
            else:
                model_params = model
            
            # 提取特征提取层参数（如果需要）
            if extract_feature_only:
                if hasattr(model, 'get_feature_extraction_params'):
                    # 使用模型自带的方法
                    feature_params = model.get_feature_extraction_params()
                else:
                    # 使用设备管理器的方法
                    feature_params = self.extract_feature_extraction_params(model_params)
                
                # 同步到聚合设备
                synced_params = self.synchronize_model_params(feature_params)
            else:
                # 同步所有参数到聚合设备
                synced_params = self.synchronize_model_params(model_params)
            
            # 保存到聚合参数字典
            aggregation_params[client_id] = synced_params
        
        return aggregation_params
    
    def restore_aggregated_model_params(self, aggregated_params, client_models_dict, client_id=None):
        """
        将聚合后的参数恢复到客户端模型
        
        Args:
            aggregated_params: 聚合后的参数
            client_models_dict: 客户端模型字典
            client_id: 特定的客户端ID，如果为None则恢复所有客户端模型
            
        Returns:
            更新后的客户端模型字典
        """
        # 如果指定了客户端ID，只恢复该客户端的模型
        if client_id is not None:
            if client_id in client_models_dict:
                # 获取原始模型
                model = client_models_dict[client_id]
                
                # 获取目标设备
                target_device = self.get_client_device(client_id)
                
                # 将聚合参数移至目标设备并恢复数据类型
                device_params = self.to_original_device(aggregated_params, client_id, restore_type=True)
                
                # 更新模型参数
                if hasattr(model, 'state_dict') and hasattr(model, 'load_state_dict'):
                    # 获取当前模型的状态字典
                    current_state_dict = model.state_dict()
                    
                    # 只更新特征提取层参数
                    for name, param in device_params.items():
                        if name in current_state_dict:
                            current_state_dict[name] = param
                    
                    # 加载更新后的状态字典
                    model.load_state_dict(current_state_dict)
                else:
                    # 对于非标准模型，直接替换
                    client_models_dict[client_id] = device_params
                
            return client_models_dict
        
        # 否则恢复所有客户端模型
        for client_id, model in client_models_dict.items():
            # 获取目标设备
            target_device = self.get_client_device(client_id)
            
            # 将聚合参数移至目标设备并恢复数据类型
            device_params = self.to_original_device(aggregated_params, client_id, restore_type=True)
            
            # 更新模型参数
            if hasattr(model, 'state_dict') and hasattr(model, 'load_state_dict'):
                # 获取当前模型的状态字典
                current_state_dict = model.state_dict()
                
                # 只更新特征提取层参数
                for name, param in device_params.items():
                    if name in current_state_dict:
                        current_state_dict[name] = param
                
                # 加载更新后的状态字典
                model.load_state_dict(current_state_dict)
            else:
                # 对于非标准模型，直接替换
                client_models_dict[client_id] = device_params
        
        return client_models_dict
    
    def clear_cuda_cache(self):
        """清理CUDA缓存，释放内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("已清理CUDA缓存")
    
    def get_device_usage_stats(self):
        """
        获取设备使用统计
        
        Returns:
            设备使用统计信息
        """
        stats = {
            'device_usage': dict(self.device_usage),
            'aggregation_device': self.aggregation_device
        }
        
        # 如果有GPU，添加内存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = f'cuda:{i}'
                total_memory = torch.cuda.get_device_properties(i).total_memory
                used_memory = torch.cuda.memory_allocated(i)
                stats[f'{device}_memory_usage'] = used_memory / total_memory
        
        return stats
import torch
import torch.nn as nn
import torch.optim as optim
import logging

class OptimizedTrainingStrategy:
    """优化的训练策略，适用于GroupNorm的ResNet模型"""
    
    def __init__(self, learning_rate=0.00075, lr_decay=0.85, weight_decay=1e-4):
        """
        初始化训练策略
        
        Args:
            learning_rate: 初始学习率，为标准学习率的0.75倍（适应GroupNorm）
            lr_decay: 学习率衰减因子
            weight_decay: 权重衰减参数
        """
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.logger = logging.getLogger("TrainingStrategy")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"优化训练策略初始化完成，学习率: {learning_rate}, 衰减因子: {lr_decay}, 权重衰减: {weight_decay}")
    
    def create_optimizer(self, model, optimizer_type="Adam"):
        """
        创建优化器
        
        Args:
            model: 要优化的模型
            optimizer_type: 优化器类型，支持"Adam"和"SGD"
            
        Returns:
            适合GroupNorm的优化器
        """
        if optimizer_type == "Adam":
            # Adam优化器更适合GroupNorm
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),  # 使用默认值
                eps=1e-8
            )
        elif optimizer_type == "SGD":
            # SGD优化器适合大批量训练
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True  # 使用Nesterov动量
            )
        else:
            # 默认使用Adam
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        return optimizer
    
    def create_component_optimizers(self, model, global_classifier=None):
        """
        为模型不同组件创建单独的优化器
        
        Args:
            model: 客户端模型
            global_classifier: 全局分类器(可选)
            
        Returns:
            optimizers: 包含不同组件优化器的字典
        """
        optimizers = {}
        
        # 特征提取层和分类器使用不同的学习率乘数
        lr_multipliers = {
            'feature_extractor': 1.0,   # 特征提取层使用基准学习率
            'classifier': 1.2,          # 分类器使用更高学习率
            'global_classifier': 0.75    # 全局分类器使用更低学习率
        }
        
        # 不同组件的权重衰减
        weight_decay_values = {
            'feature_extractor': self.weight_decay,
            'classifier': self.weight_decay * 2,  # 分类器使用更强的权重衰减
            'global_classifier': self.weight_decay * 1.5  # 全局分类器使用适中的权重衰减
        }
        
        # 获取模型的特征提取层和非特征提取层参数
        if hasattr(model, 'get_feature_extraction_params') and hasattr(model, 'get_non_feature_extraction_params'):
            # 使用模型自带的方法
            feature_params = list(model.get_feature_extraction_params().values())
            non_feature_params = list(model.get_non_feature_extraction_params().values())
        else:
            # 手动分离参数
            feature_params = []
            non_feature_params = []
            
            for name, param in model.named_parameters():
                if any(substr in name for substr in ['conv', 'gn', 'layer', 'downsample']) and not any(substr in name for substr in ['classifier', 'projection', 'fc']):
                    feature_params.append(param)
                else:
                    non_feature_params.append(param)
        
        # 为特征提取层创建优化器
        if feature_params:
            optimizers['feature_extractor'] = optim.Adam(
                feature_params,
                lr=self.learning_rate * lr_multipliers['feature_extractor'],
                weight_decay=weight_decay_values['feature_extractor'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # 为分类器创建优化器
        if non_feature_params:
            optimizers['classifier'] = optim.Adam(
                non_feature_params,
                lr=self.learning_rate * lr_multipliers['classifier'],
                weight_decay=weight_decay_values['classifier'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # 为全局分类器创建优化器（如果提供）
        if global_classifier is not None:
            optimizers['global_classifier'] = optim.Adam(
                global_classifier.parameters(),
                lr=self.learning_rate * lr_multipliers['global_classifier'],
                weight_decay=weight_decay_values['global_classifier'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        return optimizers
    
    def create_scheduler(self, optimizer, patience=10, factor=0.85, min_lr=1e-6):
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器
            patience: 容忍多少个轮次不改善
            factor: 学习率衰减因子
            min_lr: 最小学习率
            
        Returns:
            适合GroupNorm的学习率调度器
        """
        # 使用ReduceLROnPlateau调度器，适合GroupNorm
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',       # 监控损失减少
            factor=factor,    # 学习率衰减因子
            patience=patience, # 容忍多少个轮次不改善
            verbose=True,     # 打印信息
            min_lr=min_lr     # 最小学习率
        )
        
        return scheduler
    
    def create_schedulers_dict(self, optimizers_dict, patience=10, factor=0.85, min_lr=1e-6):
        """
        为优化器字典创建学习率调度器字典
        
        Args:
            optimizers_dict: 优化器字典
            patience: 容忍多少个轮次不改善
            factor: 学习率衰减因子
            min_lr: 最小学习率
            
        Returns:
            学习率调度器字典
        """
        schedulers = {}
        
        for name, optimizer in optimizers_dict.items():
            schedulers[name] = self.create_scheduler(
                optimizer,
                patience=patience,
                factor=factor,
                min_lr=min_lr
            )
        
        return schedulers
    
    def get_training_config(self):
        """
        获取训练配置
        
        Returns:
            训练配置字典
        """
        return {
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'weight_decay': self.weight_decay,
            'optimizer': 'Adam',  # 默认优化器
            'scheduler': 'ReduceLROnPlateau'  # 默认调度器
        }


# 为客户端创建训练策略的函数
def create_training_strategy_for_client(tier, is_low_resource=False):
    """
    为客户端创建适合的训练策略
    
    Args:
        tier: 客户端tier级别
        is_low_resource: 是否为低资源客户端
        
    Returns:
        针对客户端优化的训练策略
    """
    # 根据tier级别调整学习率和权重衰减
    base_lr = 0.00075  # 基准学习率（标准的0.75倍，适应GroupNorm）
    
    # 高tier客户端（更复杂的模型）使用稍低的学习率，防止过拟合
    if tier <= 3:
        lr_multiplier = 0.8
        weight_decay = 2e-4  # 更强的正则化
    # 中tier客户端使用基准学习率
    elif tier <= 5:
        lr_multiplier = 1.0
        weight_decay = 1e-4  # 标准正则化
    # 低tier客户端（简单模型）使用更高的学习率，加速收敛
    else:
        lr_multiplier = 1.2
        weight_decay = 5e-5  # 更弱的正则化
    
    # 低资源客户端使用更保守的配置
    if is_low_resource:
        lr_multiplier *= 0.8
        
    # 创建训练策略
    return OptimizedTrainingStrategy(
        learning_rate=base_lr * lr_multiplier,
        lr_decay=0.85,
        weight_decay=weight_decay
    )
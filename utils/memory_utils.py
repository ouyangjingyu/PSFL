import gc
import torch
import psutil
import os

def free_memory():
    """释放未使用的内存，减轻内存压力"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_memory_usage():
    """打印当前内存使用情况"""
    # CPU内存
    mem = psutil.virtual_memory()
    print(f"CPU内存: 已用 {mem.percent}%, 可用 {mem.available / (1024**3):.2f} GB")
    
    # GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            free = reserved - allocated
            print(f"GPU:{i} 内存: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB, 可用 {free:.2f} GB")

def safe_model_copy(model, device=None):
    """安全地复制模型，避免使用深度复制"""
    if model is None:
        return None
        
    try:
        # 创建同类型的新模型实例
        new_model = type(model)()
        
        # 复制状态字典
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            new_model.load_state_dict(state_dict)
        
        # 移动到指定设备
        if device is not None:
            new_model = new_model.to(device)
            # 确保所有参数都在正确设备上
            for param in new_model.parameters():
                param.data = param.data.to(device)
            # 确保所有缓冲区都在正确设备上
            for buffer in new_model.buffers():
                buffer.data = buffer.data.to(device)
        
        return new_model
    except Exception as e:
        print(f"安全复制模型时出错: {str(e)}")
        # 回退到基本复制
        import copy
        result = copy.deepcopy(model)
        if device is not None:
            result = result.to(device)
        return result

def safe_to_device(model, device):
    """安全地将模型移动到指定设备，确保所有参数和缓冲区都在相同设备上"""
    if model is None:
        return None
        
    try:
        # 将模型移动到设备
        model = model.to(device)
        
        # 确保所有参数都在正确设备上
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
                
        # 确保所有缓冲区都在正确设备上
        for buffer in model.buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device)
                
        return model
    except Exception as e:
        print(f"安全移动模型到设备时出错: {str(e)}")
        return model
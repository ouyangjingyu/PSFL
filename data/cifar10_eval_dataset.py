import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset, random_split

def get_cifar10_proxy_dataset(option='test_subset', num_samples=1000, seed=42):
    """
    获取CIFAR-10共享评估测试集
    
    Args:
        option: 测试集选项 ['test_subset', 'balanced_test', 'class_samples', 'aug_test']
        num_samples: 样本数量
        seed: 随机种子，用于可重复性
        
    Returns:
        评估数据集
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 定义标准的CIFAR-10变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载CIFAR-10测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    if option == 'test_subset':
        # 选项1: 测试集随机子集 (快速且简单)
        indices = torch.randperm(len(testset))[:num_samples]
        return Subset(testset, indices)
    
    elif option == 'balanced_test':
        # 选项2: 类别平衡的测试集子集
        samples_per_class = num_samples // 10  # CIFAR-10有10个类别
        
        # 按类别索引图像
        class_indices = [[] for _ in range(10)]
        for idx, (_, label) in enumerate(testset):
            class_indices[label].append(idx)
        
        # 从每个类别中选择相等数量的样本
        balanced_indices = []
        for class_idx in range(10):
            selected = np.random.choice(class_indices[class_idx], 
                                        min(samples_per_class, len(class_indices[class_idx])), 
                                        replace=False)
            balanced_indices.extend(selected)
        
        return Subset(testset, balanced_indices)
    
    elif option == 'class_samples':
        # 选项3: 每个类别的代表性样本
        # 按类别索引图像
        class_indices = [[] for _ in range(10)]
        for idx, (_, label) in enumerate(testset):
            class_indices[label].append(idx)
            
        # 所有类别的代表性样本索引
        representative_indices = []
        
        # 从每个类别中获取代表性样本
        for class_idx, indices in enumerate(class_indices):
            # 每个类别选择num_samples // 10个样本
            selected = np.random.choice(indices, min(num_samples // 10, len(indices)), replace=False)
            representative_indices.extend(selected)
            
        return Subset(testset, representative_indices)
    
    elif option == 'aug_test':
        # 选项4: 增强的测试集 (更多样化的测试)
        # 先选择一个小的平衡子集
        subset_size = min(num_samples // 2, len(testset))
        indices = torch.randperm(len(testset))[:subset_size]
        test_subset = Subset(testset, indices)
        
        # 使用数据增强创建额外样本
        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # 收集原始数据和标签
        data, labels = [], []
        for image, label in test_subset:
            data.append(image)
            labels.append(label)
        
        # 增强数据
        aug_data = []
        for image in data:
            # 应用增强变换
            aug_image = aug_transform(image)
            aug_data.append(aug_image)
        
        # 合并原始和增强数据
        all_data = torch.stack(data + aug_data)
        all_labels = torch.tensor(labels + labels)  # 增强数据保持相同标签
        
        # 如果合并后超过要求数量，则随机选择子集
        if len(all_data) > num_samples:
            indices = torch.randperm(len(all_data))[:num_samples]
            all_data = all_data[indices]
            all_labels = all_labels[indices]
        
        return TensorDataset(all_data, all_labels)
    
    else:
        raise ValueError(f"不支持的选项: {option}")


def create_cifar10_tiny_testset(download_path='./data', num_samples=1000):
    """
    创建一个紧凑的CIFAR-10测试集并保存到文件
    
    Args:
        download_path: 数据下载和保存路径
        num_samples: 样本数量
        
    Returns:
        测试数据集路径
    """
    import os
    import pickle
    
    # 获取平衡的测试集子集
    test_dataset = get_cifar10_proxy_dataset(option='balanced_test', num_samples=num_samples)
    
    # 收集数据和标签
    data, labels = [], []
    for idx in test_dataset.indices:
        img, label = test_dataset.dataset[idx]
        data.append(img.numpy())
        labels.append(label)
    
    # 创建字典，模拟CIFAR-10格式
    tiny_testset = {
        'data': np.stack(data),
        'labels': labels
    }
    
    # 确保目录存在
    os.makedirs(download_path, exist_ok=True)
    
    # 保存到文件
    save_path = os.path.join(download_path, f'cifar10_tiny_testset_{num_samples}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(tiny_testset, f)
    
    print(f"已保存紧凑版CIFAR-10测试集({num_samples}样本)到: {save_path}")
    return save_path


def load_cifar10_tiny_testset(file_path, transform=None):
    """
    加载之前保存的紧凑版CIFAR-10测试集
    
    Args:
        file_path: 数据文件路径
        transform: 图像变换
        
    Returns:
        测试数据集
    """
    import pickle
    import torch
    from torch.utils.data import TensorDataset
    
    # 加载数据
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # 获取数据和标签
    data = data_dict['data']
    labels = data_dict['labels']
    
    # 转换为张量
    data_tensor = torch.tensor(data, dtype=torch.float32) / 255.0  # 标准化到[0,1]范围
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # 应用变换(如果提供)
    if transform is not None:
        transformed_data = []
        for img in data_tensor:
            transformed_data.append(transform(img))
        data_tensor = torch.stack(transformed_data)
    
    return TensorDataset(data_tensor, labels_tensor)


# 使用示例
if __name__ == "__main__":
    # 选项1: 直接获取评估数据集(不保存文件)
    eval_dataset = get_cifar10_proxy_dataset(
        option='balanced_test',  # 平衡的测试集
        num_samples=1000,        # 1000个样本
        seed=42                  # 随机种子
    )
    print(f"评估数据集大小: {len(eval_dataset)}")
    
    # 选项2: 创建并保存小型评估数据集
    save_path = create_cifar10_tiny_testset(
        download_path='./data',  # 保存路径
        num_samples=1000         # 样本数量
    )
    
    # 选项3: 加载之前保存的评估数据集
    # 定义标准化变换
    transform = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    loaded_dataset = load_cifar10_tiny_testset(
        file_path=save_path,     # 数据文件路径
        transform=transform      # 图像变换
    )
    print(f"加载的评估数据集大小: {len(loaded_dataset)}")
    
    # 创建数据加载器
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # 查看数据形状
    for images, labels in eval_loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签形状: {labels.shape}")
        break
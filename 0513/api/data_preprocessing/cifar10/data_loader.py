import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from PIL import Image

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    train_transform, test_transform = _data_transforms_cifar10()

    # 加载训练集和测试集
    train_dataset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    # 获取类别数和数据集大小
    class_num = 10
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    # 获取训练数据和标签
    train_data = train_dataset.data
    train_targets = np.array(train_dataset.targets)
    
    # 创建测试数据集的全局数据加载器
    test_data_global = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 按照partition_method分割数据
    if partition_method == "iid":
        # IID划分：随机打乱数据并平均分配给客户端
        total_num = train_data_num
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, client_number)
        client_idxs = batch_idxs
    elif partition_method == "hetero" or partition_method == "non-iid":
        # 非IID划分：使用Dirichlet分布分配数据
        min_size = 0
        K = class_num
        N = train_data_num
        client_idxs = [[] for _ in range(client_number)]
        
        # 使用Dirichlet分布生成各客户端对各类别的采样比例
        # 注意：这里使用partition_alpha参数控制异质性
        # alpha=0.5表示中等程度的异质性，较低的值表示更高的异质性
        proportions = np.random.dirichlet(np.repeat(partition_alpha, client_number), size=K)
        proportions = np.array([p * (len(np.where(train_targets == k)[0])) for p, k in zip(proportions, range(K))])
        proportions = proportions.astype(int)
        
        # 针对每个类别，将数据按比例分配给客户端
        for k in range(K):
            idx_k = np.where(train_targets == k)[0]
            np.random.shuffle(idx_k)
            
            # 计算每个客户端应分配到的该类别数据量
            proportions_k = proportions[k]
            # 确保总和不超过该类别的总数据量
            proportions_k = np.minimum(proportions_k, len(idx_k))
            proportions_k_normalized = proportions_k / np.sum(proportions_k)
            proportions_k = np.array([int(p * len(idx_k)) for p in proportions_k_normalized])
            
            # 处理舍入误差
            proportions_k[-1] = len(idx_k) - np.sum(proportions_k[:-1])
            
            # 分配数据索引给客户端
            index = 0
            for client_id in range(client_number):
                client_idxs[client_id].extend(idx_k[index:index + proportions_k[client_id]])
                index += proportions_k[client_id]
    else:
        raise ValueError(f"Unknown partition method: {partition_method}")

    # 统计分布情况（仅用于调试）
    traindata_cls_counts = []
    for client_id in range(client_number):
        traindata_cls_counts.append(np.bincount(train_targets[client_idxs[client_id]], minlength=class_num))

    # 为每个客户端创建数据集
    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_num_dict = {}
    
    for client_id in range(client_number):
        # 创建本地数据集
        client_idxs_np = np.array(client_idxs[client_id])
        client_train_data = train_data[client_idxs_np]
        client_train_targets = train_targets[client_idxs_np]
        
        # 转换为PyTorch数据集格式
        client_train_dataset = CIFAR10_Subset(client_train_data, client_train_targets, transform=train_transform)
        
        # 为每个客户端创建独立的测试集
        # 从全局测试集中选择与客户端训练数据分布类似的数据
        test_indices = []
        for class_id in range(class_num):
            # 计算该客户端在每个类上的数据比例
            if len(client_train_targets) > 0:
                class_ratio = np.sum(client_train_targets == class_id) / len(client_train_targets)
            else:
                class_ratio = 0
            
            # 按照这个比例从测试集中抽取数据
            test_idx_class = np.where(np.array(test_dataset.targets) == class_id)[0]
            test_idx_class_selected = np.random.choice(
                test_idx_class, 
                size=int(class_ratio * len(test_idx_class)),
                replace=False
            )
            test_indices.extend(test_idx_class_selected)
        
        # 创建客户端测试数据集
        client_test_dataset = data.Subset(test_dataset, test_indices)
        
        # 创建数据加载器
        train_data_local = data.DataLoader(
            client_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_data_local = data.DataLoader(
            client_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        
        # 保存结果
        train_data_local_dict[client_id] = train_data_local
        test_data_local_dict[client_id] = test_data_local
        train_data_local_num_dict[client_id] = len(client_train_dataset)
    
    # 创建全局训练数据加载器（仅用于评估）
    train_data_global = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts

# 自定义数据集类，用于处理CIFAR-10子集
class CIFAR10_Subset(data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # 转换图像格式
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target
    
    def __len__(self):
        return len(self.data)


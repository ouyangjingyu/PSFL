# import logging

# import numpy as np
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms

# from .datasets import CIFAR10_truncated

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# # only for test, to check the code, Only select part of samlpes

# SAMPLE_FRAC = 1 # mean every 1000 select one sample to test
# SAMPLE_FRAC_TEST = 1 # mean every 1000 select one sample to test


# # generate the non-IID distribution for all methods
# def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
#     distribution = {}
#     with open(filename, 'r') as data:
#         for x in data.readlines():
#             if '{' != x[0] and '}' != x[0]:
#                 tmp = x.split(':')
#                 if '{' == tmp[1].strip():
#                     first_level_key = int(tmp[0])
#                     distribution[first_level_key] = {}
#                 else:
#                     second_level_key = int(tmp[0])
#                     distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
#     return distribution


# def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
#     net_dataidx_map = {}
#     with open(filename, 'r') as data:
#         for x in data.readlines():
#             if '{' != x[0] and '}' != x[0] and ']' != x[0]:
#                 tmp = x.split(':')
#                 if '[' == tmp[-1].strip():
#                     key = int(tmp[0])
#                     net_dataidx_map[key] = []
#                 else:
#                     tmp_array = x.split(',')
#                     net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
#     return net_dataidx_map


# def record_net_data_stats(y_train, net_dataidx_map):
#     net_cls_counts = {}

#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp
#     logging.debug('Data statistics: %s' % str(net_cls_counts))
#     return net_cls_counts


# class Cutout(object):
#     def __init__(self, length):
#         self.length = length

#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)

#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)

#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img


# def _data_transforms_cifar10():
#     CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
#     CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

#     train_transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#     ])

#     train_transform.transforms.append(Cutout(16))

#     valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#     ])

#     return train_transform, valid_transform


# def load_cifar10_data(datadir):
#     train_transform, test_transform = _data_transforms_cifar10()

#     cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
#     cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

#     X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
#     X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

#     return (X_train, y_train, X_test, y_test)


# def partition_data(dataset, datadir, partition, n_nets, alpha):
#     logging.info("*********partition data***************")
#     X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
#     n_train = X_train.shape[0]
#     # n_test = X_test.shape[0]

#     if partition == "homo":
#         total_num = n_train
#         idxs = np.random.permutation(total_num)
#         batch_idxs = np.array_split(idxs, n_nets)
#         net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

#     elif partition == "hetero":
#         min_size = 0
#         K = 10
#         N = y_train.shape[0]
#         logging.info("N = " + str(N))
#         net_dataidx_map = {}

#         while min_size < 10:
#             idx_batch = [[] for _ in range(n_nets)]
#             # for each class in the dataset
#             for k in range(K):
#                 idx_k = np.where(y_train == k)[0]
#                 np.random.shuffle(idx_k)
#                 proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
#                 ## Balance
#                 proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
#                 proportions = proportions / proportions.sum()
#                 proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
#                 idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#                 min_size = min([len(idx_j) for idx_j in idx_batch])

#         for j in range(n_nets):
#             np.random.shuffle(idx_batch[j])
#             net_dataidx_map[j] = idx_batch[j]

#     elif partition == "hetero-fix":
#         dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
#         net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

#     if partition == "hetero-fix":
#         distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
#         traindata_cls_counts = read_data_distribution(distribution_file_path)
#     else:
#         traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

#     return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# # for centralized training
# def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
#     return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)


# # for local devices
# def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
#     return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


# def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
#     dl_obj = CIFAR10_truncated

#     transform_train, transform_test = _data_transforms_cifar10()

#     train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
#     test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
#     if not torch.cuda.is_available():  # to reduce for chekcing on CPU
#         test_ds = dl_obj(datadir, dataidxs=list(range(0,10000,SAMPLE_FRAC)), train=False, transform=transform_test, download=True)

#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

#     return train_dl, test_dl


# def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
#     dl_obj = CIFAR10_truncated

#     transform_train, transform_test = _data_transforms_cifar10()

#     train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
#     test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

#     return train_dl, test_dl


# def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
#                                             client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

#     # get global test data
#     if process_id == 0:
#         train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#         logging.info("train_dl_global number = " + str(len(train_data_global)))
#         logging.info("test_dl_global number = " + str(len(test_data_global)))
#         train_data_local = None
#         test_data_local = None
#         local_data_num = 0
#     else:
#         # get local dataset
#         dataidxs = net_dataidx_map[process_id - 1]
#         local_data_num = len(dataidxs)
#         logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             process_id, len(train_data_local), len(test_data_local)))
#         train_data_global = None
#         test_data_global = None
#     return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


# def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

#     train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#     logging.info("train_dl_global number = " + str(len(train_data_global)))
#     logging.info("test_dl_global number = " + str(len(test_data_global)))
#     test_data_num = len(test_data_global)

#     # get local dataset
#     data_local_num_dict = dict()
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()

#     for client_idx in range(client_number):
#         dataidxs = net_dataidx_map[client_idx]
#         if not torch.cuda.is_available():  # to reduce for chekcing on CPU
#             dataidxs = dataidxs[::SAMPLE_FRAC_TEST]
#         local_data_num = len(dataidxs)
#         data_local_num_dict[client_idx] = local_data_num
#         logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             client_idx, len(train_data_local), len(test_data_local)))
#         train_data_local_dict[client_idx] = train_data_local
#         test_data_local_dict[client_idx] = test_data_local
#     return train_data_num, test_data_num, train_data_global, test_data_global, \
#            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# only for test, to check the code, Only select part of samlpes

SAMPLE_FRAC = 1 # mean every 1000 select one sample to test
SAMPLE_FRAC_TEST = 1 # mean every 1000 select one sample to test


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


# def partition_data(dataset, datadir, partition, n_nets, alpha):
#     logging.info("*********partition data***************")
#     X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
#     n_train = X_train.shape[0]
#     # n_test = X_test.shape[0]

#     if partition == "homo":
#         total_num = n_train
#         idxs = np.random.permutation(total_num)
#         batch_idxs = np.array_split(idxs, n_nets)
#         net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

#     elif partition == "hetero":
#         min_size = 0
#         K = 10
#         N = y_train.shape[0]
#         logging.info("N = " + str(N))
#         net_dataidx_map = {}

#         while min_size < 10:
#             idx_batch = [[] for _ in range(n_nets)]
#             # for each class in the dataset
#             for k in range(K):
#                 idx_k = np.where(y_train == k)[0]
#                 np.random.shuffle(idx_k)
#                 proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
#                 ## Balance
#                 proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
#                 proportions = proportions / proportions.sum()
#                 proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
#                 idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#                 min_size = min([len(idx_j) for idx_j in idx_batch])

#         for j in range(n_nets):
#             np.random.shuffle(idx_batch[j])
#             net_dataidx_map[j] = idx_batch[j]

#     elif partition == "hetero-fix":
#         dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
#         net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

#     if partition == "hetero-fix":
#         distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
#         traindata_cls_counts = read_data_distribution(distribution_file_path)
#     else:
#         traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

#     return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
def partition_data(dataset, datadir, partition, n_nets, alpha):
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    
    if partition == "hetero":
        # 为训练集和测试集分别创建索引映射
        train_net_dataidx_map = {}
        test_net_dataidx_map = {}
        
        # 分割训练集
        min_size = 0
        K = 10
        N = y_train.shape[0]
        
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                
        # 同样方式分割测试集
        test_idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_batch, np.split(idx_k, proportions))]

        # 保存训练和测试索引映射
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            train_net_dataidx_map[j] = idx_batch[j]
            np.random.shuffle(test_idx_batch[j])
            test_net_dataidx_map[j] = test_idx_batch[j]
            
    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map

# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


# def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
#     dl_obj = CIFAR10_truncated

#     transform_train, transform_test = _data_transforms_cifar10()

#     train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
#     test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
#     if not torch.cuda.is_available():  # to reduce for chekcing on CPU
#         test_ds = dl_obj(datadir, dataidxs=list(range(0,10000,SAMPLE_FRAC)), train=False, transform=transform_test, download=True)

#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

#     return train_dl, test_dl

def get_dataloader_CIFAR10(datadir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = CIFAR10_truncated
    transform_train, transform_test = _data_transforms_cifar10()

    # 使用不同的索引创建训练集和测试集
    train_ds = dl_obj(datadir, dataidxs=train_dataidxs, train=True, 
                     transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=False, 
                    transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, 
                              shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, 
                             shuffle=False, drop_last=False)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


# def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

#     train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#     logging.info("train_dl_global number = " + str(len(train_data_global)))
#     logging.info("test_dl_global number = " + str(len(test_data_global)))
#     test_data_num = len(test_data_global)

#     # get local dataset
#     data_local_num_dict = dict()
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()

#     for client_idx in range(client_number):
#         dataidxs = net_dataidx_map[client_idx]
#         if not torch.cuda.is_available():  # to reduce for chekcing on CPU
#             dataidxs = dataidxs[::SAMPLE_FRAC_TEST]
#         local_data_num = len(dataidxs)
#         data_local_num_dict[client_idx] = local_data_num
#         logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             client_idx, len(train_data_local), len(test_data_local)))
#         train_data_local_dict[client_idx] = train_data_local
#         test_data_local_dict[client_idx] = test_data_local
#     return train_data_num, test_data_num, train_data_global, test_data_global, \
#            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    # 获取分割的数据
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    
    # 计算总的训练和测试样本数
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    
    class_num = len(np.unique(y_train))
    
    # 获取全局数据加载器
    train_data_global, test_data_global = get_dataloader_CIFAR10(
        data_dir, batch_size, batch_size)
    
    # 为每个客户端创建本地数据加载器和记录数据量
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        test_dataidxs = test_net_dataidx_map[client_idx]
        
        # 记录每个客户端的数据量
        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        
        # 创建本地数据加载器
        train_data_local, test_data_local = get_dataloader_CIFAR10(
            data_dir, batch_size, batch_size, 
            train_dataidxs, test_dataidxs)
            
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
        logging.info(f"Client {client_idx} - Training samples: {local_data_num}, "
                    f"Test samples: {len(test_dataidxs)}")
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
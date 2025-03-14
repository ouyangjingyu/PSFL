# ============================================================================
# Deployment Environment and Resource Profiles:
# The DTFL and the baselines are deployed on a server with the following specifications:
# - Dual-sockets Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz 
# - Four NVIDIA GeForce GTX 1080 Ti GPUs
# - 64 GB of memory

# Each client in the simulation is assigned a distinct simulated CPU and communication resource
# to replicate heterogeneous resources, simulating varying training times based on CPU/network profiles.
# We simulate a heterogeneous environment with varying client capacity in both cross-solo and cross-device FL settings.

# We consider 5 resource profiles:
# 1. 4 CPUs with 100 Mbps
# 2. 2 CPUs with 30 Mbps
# 3. 1 CPU with 30 Mbps
# 4. 0.2 CPU with 30 Mbps
# 5. 0.1 CPU with 10 Mbps communication speed to the server.

# In this implementaion number of tiers is 6 (M=6)
# ============================================================================

import torch
import torchvision
from torch import cosine_similarity, nn
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random
import numpy as np
import os

import time
import sys
import wandb
import argparse
import logging

import warnings
from sklearn.cluster import KMeans

# Ignore all warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from model.resnet import resnet56_SFL_local_tier_7
from model.resnet import resnet110_SFL_fedavg_base
from model.resnet import resnet110_SFL_local_tier_7

from utils.loss import PatchShuffle
from utils.loss import dis_corr
from utils.fedavg import aggregated_fedavg
from utils.fedavg import aggregate_clients_models
from utils.clustering import data_distribution_aware_clustering

from utils.TierScheduler import TierScheduler
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from data.cifar10_eval_dataset import create_cifar10_tiny_testset
from data.cifar10_eval_dataset import load_cifar10_tiny_testset
from data.cifar10_eval_dataset import get_cifar10_proxy_dataset



import matplotlib
matplotlib.use('Agg')
import copy
# from multiprocessing import Process
# import torch.multiprocessing as mp
# from multiprocessing import Pool


SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "Multi-Tier Splitfed Local Loss"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

def add_args(parser):
    
    parser.add_argument('--running_name', default="DTFL", type=str)
    
    # Optimization related arguments
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_patience', default=10, type=float)
    parser.add_argument('--lr_min', default=0, type=float)
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
 
    # Model related arguments
    parser.add_argument('--model', type=str, default='resnet110', metavar='N',
                        help='neural network used in training')
    
    
    # Data loading and preprocessing related arguments
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
        
    # Federated learning related arguments
    parser.add_argument('--client_epoch', default=1, type=int)
    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    # parser.add_argument('--batch_size', type=int, default=100, metavar='N',
    #                     help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N',  # 改为200，每个GPU处理100
                    help='input batch size for training')
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--whether_local_loss', default=True, type=bool)
    parser.add_argument('--tier', default=5, type=int)
        
    
    # Privacy related arguments
    # parser.add_argument('--whether_dcor', default=False, type=bool)
    # parser.add_argument('--dcor_coefficient', default=0.5, type=float)  # same as alpha in paper
    # parser.add_argument('--PatchShuffle', default=0, type=int)  
    
    
    
    # Add the argument for simulation like net_speed_list
    parser.add_argument('--net_speed_list', type=str, default=[100, 30, 30, 30, 10], 
                    metavar='N', help='list of net speeds in mega bytes')
    parser.add_argument('--delay_coefficient_list', type=str, default=[16, 20, 34, 130, 250],
                    metavar='N', help='list of delay coefficients')
    
    args = parser.parse_args()
    return args

DYNAMIC_LR_THRESHOLD = 0.0001
DEFAULT_FRAC = 1.0        # participation of clients


#### Initialization
# T_max = 1000

NUM_CPUs = os.cpu_count()

parser = argparse.ArgumentParser()
args = add_args(parser)
logging.info(args)

    
wandb.init(
    mode="online",
    project="ParallelSFL",
    name="ParallelSFL",# + str(args.tier),
    config=args,
    # tags="Tier1_5",
    # group="ResNet56",
)



SFL_local_tier = resnet56_SFL_local_tier_7

### model selection


if args.dataset == 'cifar10':
    class_num = 10
elif args.dataset == 'cifar100' or args.dataset == 'cinic10':
    class_num = 100


    
if args.model == 'resnet110':
    SFL_local_tier = resnet110_SFL_local_tier_7
    num_tiers = 7
    init_glob_model = resnet110_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)





whether_local_loss = args.whether_local_loss
# whether_dcor = args.whether_dcor
# dcor_coefficient = args.dcor_coefficient
tier = args.tier
client_epoch = args.client_epoch
client_epoch = np.ones(args.client_number,dtype=int) * client_epoch
client_number = args.client_number

# 在这一步，控制客户端初始化的分级tier为7
# client_type_percent = [0.0, 0.0, 0.0, 0.0, 1.0]

# if num_tiers == 7:
#     client_type_percent = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
#     tier = 1

    

# client_number_tier = (np.dot(args.client_number , client_type_percent))

# 设置客户端的存储能力，模拟设备异质性
def allocate_resources_to_clients(num_clients):
    """
    Allocate storage capabilities and device resources for N clients, simulating heterogeneity.
    Each client is assigned a storage tier, device resource tier, and storage capacity, based on a distribution strategy.
    
    Args:
        num_clients (int): Number of clients to allocate resources to.
    
    Returns:
        dict: A dictionary mapping each client index to its assigned storage tier, device resource tier, and storage capacity.
    """
    
    # Define the number of tiers for storage and device resources
    num_storage_tiers = 7
    client_resource_allocation = {}

    # Define probabilities for each tier (tiers 3, 4, 5 have higher probabilities)
    storage_tier_weights = [0.05, 0.05, 0.25, 0.25, 0.25, 0.1, 0.05]  # Probabilities for tiers 1 to 7

    # Assign each client to a storage tier (1 to 7) and a random resource tier (1 to 5)
    for client_id in range(num_clients):
        storage_tier = random.choices(range(1, num_storage_tiers + 1), weights=storage_tier_weights, k=1)[0]
        storage_capacity = random.choice([64, 128, 256, 512, 1024])  # Assign a random storage capacity in GB
        client_resource_allocation[client_id] = {
            "storage_tier": storage_tier,
            "storage_capacity": storage_capacity
        }
    
    return client_resource_allocation

client_resource_allocation = allocate_resources_to_clients(client_number)
client_tier,client_number_tier,tierschedule_time_clients = TierScheduler(client_resource_allocation)


########## network speed profile of clients
# 设置客户端网络传输速度异质性，compute_delay计算网络传输总时延
def compute_delay(data_transmitted_client:float, net_speed:float, delay_coefficient:float, duration) -> float:
    net_delay = data_transmitted_client / net_speed
    computation_delay = duration * delay_coefficient
    total_delay = net_delay + computation_delay
    simulated_delay = total_delay
    return simulated_delay

net_speed_list = np.array([100, 200, 500]) * 1024000 ** 2  # MB/s: speed for transmitting data
net_speed_weights = [0.5, 0.25, 0.25]  # weights for each speed level
net_speed = random.choices(net_speed_list, weights=net_speed_weights, k=args.client_number)


net_speed_list = list(np.array(args.net_speed_list) * 1024 ** 2)

net_speed = net_speed_list * (args.client_number // 5 + 1)

delay_coefficient_list = list(np.array(args.delay_coefficient_list) / 14.5)  # to scale on the GPU 

delay_coefficient = delay_coefficient_list * (args.client_number // 5 + 1)  # coeffieient list for simulation computational power
delay_coefficient = list(np.array(delay_coefficient))

############### Client profiles definitions ###############
client_cpus_gpus = [(0, 0.5, 0), (1, 0.5, 0), (2, 0, 1), (3, 2, 0), (4, 1, 0),
                    (5, 0.5, 0), (6, 0.5, 0), (7, 0, 1), (8, 2, 0), (9, 1, 0)]





total_time = 0 
# global avg_tier_time_list
avg_tier_time_list = []
max_time_list = pd.DataFrame({'time' : []})
    
client_delay_computing = 0.1
client_delay_net = 0.1



# tier = 1
#===================================================================
# No. of users
num_users = args.client_number
epochs = args.rounds
lr = args.lr

# data transmmission
global data_transmit
model_parameter_data_size = 0 # model parameter 
intermediate_data_size = 0 # intermediate data

# =====
#   load dataset
# ====



def load_data(args, dataset_name):

    # 添加GPU数量检查
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs, batch_size per GPU: {args.batch_size//n_gpu}")

    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    else:
        data_loader = load_partition_data_cifar10

    if dataset_name == "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
        
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset

if args.dataset != "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
if args.dataset == "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts] = dataset

    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
    dataset_size = {}
    for i in range(0,len(traindata_cls_counts)):
        dataset_size[i] = sum(traindata_cls_counts[i].values())
    avg_dataset = sum(dataset_size.values()) / len(dataset_size)

dataset_size = {}
if args.dataset != "cinic10":
    for i in range(0,args.client_number):
        dataset_size[i] = len(dataset_train[i].dataset.target)
    avg_dataset = sum(dataset_size.values()) / len(dataset_size)
    


# Functions

########################### Client Selection ##########################

def get_random_user_indices(num_users, DEFAULT_FRAC=0.1):
    m = max(int(DEFAULT_FRAC * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    return idxs_users, m


def calculate_data_size(w_model):
    """
    Calculate the data size (memory usage) of tensors in the w_glob_client_tier for a specific model

    Parameters:
        w_model (dict): Dictionary containing tensors for each model.

    Returns:
        int: Data size (memory usage) of tensors in bytes.
    """
    data_size = 0
    for k in w_model:
        data_size += sys.getsizeof(w_model[k].storage())
        # tensor = w_model[k]
        # data_size += tensor.numel() * tensor.element_size() # this calculate the tensor size, but a little smaller than with using sys
    return data_size

#####
#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side

    # # def __init__(self):
    # def __init__(self, block, num_layers, classes):
    #     super(ResNet18_client_side, self).__init__()
    #     self.input_planes = 64
    #     self.layer1 = nn.Sequential (
    #             nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
    #             nn.BatchNorm2d(64),
    #             nn.ReLU (inplace = True),
    #             nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
    #         )
            
    #     # Aux network  fedgkt
            
    #     self.layer2 = self._layer(block, 16, 1) # layers[0] =1

    #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    #     self.fc = nn.Linear(16 * 1, classes )  # block.expansion = 1 , classes

            
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
                    
    # def _layer(self, block, planes, num_layers, stride = 2):
    #     dim_change = None
    #     if stride != 1 or planes != self.input_planes * block.expansion:
    #         dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
    #                                     nn.BatchNorm2d(planes*block.expansion))
    #     netLayers = []
    #     netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
    #     self.input_planes = planes * block.expansion
    #     for i in range(1, num_layers):
    #         netLayers.append(block(self.input_planes, planes))
    #         self.input_planes = planes * block.expansion
                
    #     return nn.Sequential(*netLayers)
            
            
    # def forward(self, x):
    #     resudial1 = F.relu(self.layer1(x))   # here from fedgkt code extracted_features = x without maxpool
            
    #     # Aux Network output
    #     # extracted_features = resudial1

    #     x = self.layer2(resudial1)  # B x 16 x 32 x 32
    #     # x = self.layer2(x)  # B x 32 x 16 x 16
    #     # x = self.layer3(x)  # B x 64 x 8 x 8

    #     x = self.avgpool(x)  # B x 64 x 1 x 1
    #     x_f = x.view(x.size(0), -1)  # B x 64
    #     extracted_features = self.fc(x_f)  # B x num_classes

    #     return extracted_features, resudial1
    
net_glob_client_tier = {}


net_glob_client_tier[1],_ = SFL_local_tier(classes=class_num,tier=5)
net_glob_client,_ = SFL_local_tier(classes=class_num,tier=tier)
for i in range(1,num_tiers+1):
    net_glob_client_tier[i],_ = SFL_local_tier(classes=class_num,tier=i)

    


"""
    Note that we only initialize the client feature extractor to mitigate the difficulty of alternating optimization
"""

if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)
    # net_glob_client = nn.DataParallel(net_glob_client, device_ids=list(range(torch.cuda.device_count())))  
    # for i in range(1, num_tiers+1):
    #     net_glob_client_tier[i] = nn.DataParallel(net_glob_client_tier[i], device_ids=list(range(torch.cuda.device_count())))
    for i in range(1, num_tiers+1):
        net_glob_client_tier[i] = nn.DataParallel(net_glob_client_tier[i])
        

for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)


    
net_glob_client.to(device)


net_glob_server_tier = {}

_, net_glob_server = SFL_local_tier(classes=class_num,tier=tier) # local loss SplitFed
for i in range(1,num_tiers+1):
    _, net_glob_server_tier[i] = SFL_local_tier(classes=class_num,tier=i)
    
    
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)
    # net_glob_server = nn.DataParallel(net_glob_server, device_ids=list(range(torch.cuda.device_count())))   # to use the multiple GPUs 
    # for i in range(1, num_tiers+1):
    #     net_glob_server_tier[i] = nn.DataParallel(net_glob_server_tier[i], device_ids=list(range(torch.cuda.device_count())))
    for i in range(1, num_tiers+1):
        net_glob_server_tier[i] = nn.DataParallel(net_glob_server_tier[i])
        
        
for i in range(1, num_tiers+1):
    net_glob_server_tier[i].to(device)

net_glob_server.to(device)


#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []


criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

time_train_server_train = 0
time_train_server_train_all = 0

#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
# def FedAvg(w):
#     len_min = float('inf')
#     index_len_min = 0
#     for j in range(0, len(w)):
#         if len(w[j]) < len_min:
#             len_min = len(w[j])
#             index_len_min = j
#     w[0],w[index_len_min] = w[index_len_min],w[0]
            
            
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         c = 1
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#             c += 1
#         w_avg[k] = torch.div(w_avg[k], c)
#     return w_avg

def FedAvg_wighted(w, client_sample):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # w_avg[k] += w[i][k] * client_sample[i]  # to solve long error
            w_avg[k] += w[i][k] * client_sample[i].to(w_avg[k].dtype)  # maybe other method can be used
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
best_acc = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_glob_server_tier ={}
net_glob_server_tier[tier].load_state_dict(w_glob_server)
for i in range(1, num_tiers+1):
   w_glob_server_tier[i] = net_glob_server_tier[i].state_dict()
w_locals_server = []
w_locals_server_tier = {}
for i in range(1,num_tiers+1):
    w_locals_server_tier[i]=[]


#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server_tier = {}
net_model_client_tier = {}
client_tier = {}
for i in range (0, num_users):
    client_tier[i] = num_tiers
k = 0
net_model_server = [net_glob_server for i in range(num_users)]
for i in range(len(client_number_tier)):
    for j in range(int(client_number_tier[i])):
        net_model_server_tier[k] = net_glob_server_tier[i+1]
        # net_model_client_tier[k] = net_glob_client_tier[i+1]
        client_tier[k] = i+1
        k +=1
net_server = copy.deepcopy(net_model_server[0]).to(device)
net_server = copy.deepcopy(net_model_server_tier[0]).to(device)
        


#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
optimizer_server_glob =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
scheduler_server = ReduceLROnPlateau(optimizer_server_glob, 'max', factor=0.8, patience=0, threshold=0.0000001)

patience = args.lr_patience
factor= args.lr_factor
wait=0
new_lr = lr
min_lr = args.lr_min

times_in_server = []
        
        
# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, extracted_features):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server, time_train_server_train, time_train_server_train_all, w_glob_server_tier, w_locals_server_tier, w_locals_tier
    global loss_train_collect_user, acc_train_collect_user, lr, total_time, times_in_server, new_lr
    # global server_lr_scheduler
    time_train_server_s = time.time()
    
    net_server = copy.deepcopy(net_model_server_tier[idx]).to(device)
    # 获取原始模型以访问 tier 属性
    actual_net = net_server.module if isinstance(net_server, torch.nn.DataParallel) else net_server
    server_tier = actual_net.tier

    # 开始训练
    net_server.train()
    lr = new_lr
    if args.optimizer == "Adam":
        optimizer_server =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
    elif args.optimizer == "SGD":
        optimizer_server =  torch.optim.SGD(net_server.parameters(), lr=lr, momentum=0.9,
                                              nesterov=True,
                                              weight_decay=args.wd)
    
    
    time_train_server_s = time.time()
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    # 确保输入需要梯度
    fx_client.requires_grad_(True)

    # 初始化统计指标
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    #---------forward prop-------------

    if server_tier in [3, 4]:
        # # 扩展到128通道
        fx_client_reduced = fx_client[:, :128, :, :]  # 只取前128个通道
        fx_client_reduced.requires_grad_(True)
        fx_client_to_use = fx_client_reduced
    elif server_tier == 7:
        # 缩减到16通道
        fx_client_reduced = fx_client[:, :16, :, :]
        fx_client_reduced.requires_grad_(True)
        fx_client_to_use = fx_client_reduced
    else:
        fx_client.requires_grad_(True)
        fx_client_to_use = fx_client


    # 前向传播
    net_server.zero_grad()
    fx_server = net_server(fx_client_to_use)
    y = y.to(torch.long)
    loss = criterion(fx_server, y)

    # 计算准确率
    _, predicted = torch.max(fx_server.data, 1)
    batch_correct = (predicted == y).sum().item()
    batch_total = y.size(0)

    loss.backward()

    # 处理梯度
    if server_tier in [3, 4] and fx_client_to_use.grad is not None:
        dfx_client = fx_client_to_use.grad[:, :64, :, :].clone().detach()
    elif server_tier == 7 and fx_client_to_use.grad is not None:
        # 对于tier=7，需要将16通道的梯度扩展回64通道
        dfx_client = torch.zeros_like(fx_client)
        dfx_client[:, :16, :, :] = fx_client_to_use.grad
    else:
        dfx_client = fx_client_to_use.grad.clone().detach() if fx_client_to_use.grad is not None else torch.zeros_like(fx_client)
    # dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    # 累积统计量
    epoch_loss += loss.item()
    epoch_correct += batch_correct
    epoch_total += batch_total

    
    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)
    net_model_server_tier[idx] = copy.deepcopy(net_server)
    time_train_server_train += time.time() - time_train_server_s
    # count1: to track the completion of the local batch associated with one client
    # like count1 , aggregate time_train_server_train
    count1 += 1
    if count1 == len_batch:
        # 计算平均指标
        acc_avg_train = 100 * epoch_correct / epoch_total
        loss_avg_train = epoch_loss / len_batch
       
        count1 = 0
        
        # wandb.log({"Client{}_Training_Time_in_Server".format(idx): time_train_server_train, "epoch": l_epoch_count}, commit=False)
        times_in_server.append(time_train_server_train)
        time_train_server_train_all += time_train_server_train
        total_time += time_train_server_train
        time_train_server_train = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.2f} \tLoss: {:.3f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict() 
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            if idx not in idx_collect:
                idx_collect.append(idx)

            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            w_locals_server.append(copy.deepcopy(w_server))
            w_locals_server_tier[client_tier[idx]].append(copy.deepcopy(w_server))
            
            # 记录训练指标
            loss_train_collect_user.append(loss_avg_train)
            acc_train_collect_user.append(acc_avg_train)
            
        # This is for federation process--------------------

        if len(idx_collect) == m:  # federation after evfery epoch not when all clients complete thier process like splitfed
            fed_check = True 
                                                             # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display 

            w_locals_tier = w_locals_server
            w_locals_server = []
            w_locals_server_tier = {}
            for i in range(1,num_tiers+1):
                w_locals_server_tier[i]=[]
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
            # 更新服务器端的学习率
            scheduler_server.step(loss_avg_all_user_train)

            wandb.log({"Server_Training_Time": time_train_server_train_all, "epoch": l_epoch_count}, commit=False)
            print("Server LR: ", optimizer_server.param_groups[0]['lr'])
            new_lr = optimizer_server.param_groups[0]['lr']
            wandb.log({"Server_LR": optimizer_server.param_groups[0]['lr'], "epoch": l_epoch_count}, commit=False)
            
    
    # print(time_train_server_copy, time_train_server_train)
    # send gradients to the client               
    return dfx_client
    # return dfx_client, new_lr  # output of server 

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server, net_glob_server_tier 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check, w_glob_server_tier
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, acc_avg_all_user, loss_avg_all_user_train, best_acc
    global wait, new_lr
    
    net = copy.deepcopy(net_model_server_tier[idx]).to(device)
    # 获取原始模型以访问 tier 属性
    actual_net = net_server.module if isinstance(net_server, torch.nn.DataParallel) else net_server
    server_tier = actual_net.tier
    net.eval()

    # 初始化统计指标
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        # print(f"Evaluating for tier {net.tier}")
    
        if server_tier in [3, 4]:
            # 扩展输入通道
            # fx_client_expanded = torch.cat([fx_client, torch.zeros_like(fx_client)], dim=1)
            fx_client_to_use = fx_client[:, :128, :, :]  # 截取前128个通道
            fx_server = net(fx_client_to_use)
            
        elif server_tier == 7:
            # 缩减到16通道
            fx_client_reduced = fx_client[:, :16, :, :]
            fx_server = net(fx_client_reduced)
        else:
            fx_server = net(fx_client)
            
        # fx_server = net(fx_client)
        # print(f"Server output shape: {fx_server.shape}")
    
        y = y.reshape(-1).to(torch.long)  # 确保标签是一维的
        # print(f"Reshaped labels shape: {y.shape}")
        # calculate loss
        # y = y.to(torch.long)
        loss = criterion(fx_server, y)
        # acc = calculate_accuracy(fx_server, y)
        _, predicted = torch.max(fx_server.data, 1)
        batch_correct = (predicted == y).sum().item()
        batch_total = y.size(0)
        
        
        # batch_loss_test.append(loss.item())
        # batch_acc_test.append(acc.item())
        # 累积统计量
        epoch_loss += loss.item()
        epoch_correct += batch_correct
        epoch_total += batch_total
    
        count2 += 1
        if count2 == len_batch:
            # acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            # loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            # batch_acc_test = []
            # batch_loss_test = []
            # 计算平均指标
            acc_avg_test = 100 * epoch_correct / epoch_total
            loss_avg_test = epoch_loss / len_batch
            count2 = 0
            
            prGreen('Global Model Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(acc_avg_test, loss_avg_test))


            if loss_avg_test > 100:
                print(loss_avg_test)
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                
                
                if (acc_avg_all_user/100) > best_acc  * ( 1 + DYNAMIC_LR_THRESHOLD ):
                    print("- Found better accuracy")
                    best_acc = (acc_avg_all_user/100)
                    wait = 0
                else:
                     wait += 1 
                     print('wait', wait)
                if wait > patience:   #https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
                    new_lr = max(float(optimizer_server.param_groups[0]['lr']) * factor, min_lr)
                    wait = 0
                    
                    
                              
                print("==========================================================")
                print("{:^58}".format("DTFL Performance"))
                print("----------------------------------------------------------")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
                
                wandb.log({"Server_Training_Accuracy": acc_avg_all_user_train, "epoch": ell}, commit=False)
                wandb.log({"Server_Test_Accuracy": acc_avg_all_user, "epoch": ell}, commit=False)

         
    return 

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================


# Client-side functions associated with Training and Testing

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None,pretrain_rounds=5):
        self.idx = idx
        self.device = device
        self.sl_lr = lr  # 用于 SL 训练的学习率
        self.local_lr = 0.001  # 用于本地训练的学习率
        self.local_ep = client_epoch[idx]
        self.ldr_train = dataset_train[idx]
        self.ldr_test = dataset_test[idx]
        self.pretrain_rounds = pretrain_rounds
        # 添加学习率调度器作为类属性
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = 0  # 追踪总训练轮数

    def init_optimizer_scheduler(self, net):
        """初始化或更新优化器和调度器"""
        if args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(net.parameters(), 
                                            lr=self.local_lr, 
                                            weight_decay=args.wd, 
                                            amsgrad=True)
        elif args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(net.parameters(), 
                                           lr=self.local_lr, 
                                           momentum=0.9,
                                           nesterov=True,
                                           weight_decay=args.wd)
        
        # 只有在首次调用时创建调度器
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=20,  # 每20轮调整一次
                gamma=0.9      # 调整为原来的0.9倍
            )
            
    # 客户端在预训练阶段，首先要初始训练N轮，向聚合服务器上传预训练模型，根据模型相似度对客户端进行分类。
    def pre_train(self,net):      
        net.train()

        if args.optimizer == "Adam":
            optimizer_client =  torch.optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
        elif args.optimizer == "SGD":
            optimizer_client =  torch.optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9,
                                                        nesterov=True,
                                                        weight_decay=args.wd)
        
        time_pretrained = 0

        for round in range(self.pretrain_rounds):
            len_batch = len(self.ldr_train)
            epoch_pretrain_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_start = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                    
                        
                #---------pretrained-------------

                output = net(images)

                # 计算本地损失值，进行反向传播   
                labels = labels.to(torch.long)
                if isinstance(output, tuple):
                    # 如果输出是元组，使用第一个元素（通常是预测结果）
                    output = output[0]
                loss_pretrained = criterion(output, labels)
                loss_pretrained.backward()       
                optimizer_client.step()
                time_pretrained += time.time() - time_start

                epoch_pretrain_loss += loss_pretrained.item()

            avg_epoch_loss = epoch_pretrain_loss / len_batch
        # 计算预训练阶段，客户端上传的模型数据大小       
        total_size = 0
        for param in net.parameters():
            total_size += param.nelement() * param.element_size()
        self.model_size = total_size / (1024 * 1024)
        print(f"Model size after pre-training: {self.model_size:.2f} MB")
                   
        # clients log
        print(f"Client {self.idx} finished pre-training for {self.pretrain_rounds} rounds in {time_pretrained:.2f} seconds.")
            
        return net.state_dict(), time_pretrained


    # 客户端每轮训练开始前，初始化的本地模型net = copy.deepcopy(net_glob_client).to(device)，现在我们要对每一个客户端添加一个分类器

    # 客户端本地训练
    def local_train(self, net, local_train_epoch):
        net.train()
        # self.lr , lr = new_lr, new_lr
        
        # 客户端专用的学习率参数
        # 确保优化器和调度器已初始化
        self.init_optimizer_scheduler(net)

        time_client_local_trained = 0
        all_losses = []  # 记录所有epoch的损失值
        
        print(f"Client {self.idx} initial learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Current total epochs trained: {self.current_epoch}")

        for iter in range(local_train_epoch):
            len_batch = len(self.ldr_train)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client = self.optimizer
                optimizer_client.zero_grad()
                
                #---------forward prop-------------
                output, extracted_features = net(images)
                if isinstance(output, tuple):
                    output = output[0]
                    
                # 计算本地损失值，进行反向传播   
                labels = labels.to(torch.long)
                loss_local_trained = criterion(output, labels)
                loss_local_trained.backward()       
                optimizer_client.step()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
                
                # Accumulate batch loss
                epoch_loss += loss_local_trained.item()
                
                time_client_local_trained += time.time() - time_s

            # Calculate average metrics for this epoch
            avg_epoch_loss = epoch_loss / len(self.ldr_train)
            epoch_accuracy = 100 * epoch_correct / epoch_total
            
            # 记录损失值
            all_losses.append(avg_epoch_loss)

            # 更新总训练轮数
            self.current_epoch += 1
            
            # # 调用学习率调度器并更新本地学习率
            # old_lr = self.optimizer.param_groups[0]['lr']
            # self.lr_scheduler.step()
            current_local_lr = self.optimizer.param_groups[0]['lr']
            # self.local_lr = current_local_lr  # 更新类的本地学习率
            
            # if old_lr != current_local_lr:
            #     print(f'Client {self.idx} Epoch {self.current_epoch}: '
            #           f'Learning rate changed: {old_lr:.6f} -> {current_local_lr:.6f}')
            

        # clients log
        wandb.log({"Client{}_Local_train_Loss".format(idx): float(avg_epoch_loss), "epoch": self.current_epoch}, commit=False)
        wandb.log({"Client{}_Local_train_Acc".format(idx): float(epoch_accuracy), "epoch": self.current_epoch}, commit=False)
        wandb.log({"Client{}_local_time_not_scaled (s)".format(idx): time_client_local_trained, "epoch": self.current_epoch}, commit=False)
        wandb.log({"Client{}_local_train_lr (s)".format(idx): current_local_lr, "epoch": self.current_epoch}, commit=False)
        return net.state_dict(), time_client_local_trained 
    
    # 拆分学习训练过程
    def train(self, net):
        # 开始训练
        net.train()
        # self.sl_lr = new_lr  # 使用服务器端更新的学习率
        self.sl_lr , lr = new_lr, new_lr

        if args.optimizer == "Adam":
            optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
        elif args.optimizer == "SGD":
            optimizer_client =  torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                                      nesterov=True,
                                                      weight_decay=args.wd)
        
        
        
        time_client=0
        client_intermediate_data_size = 0
        CEloss_client_train = []
        Dcorloss_client_train = []

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                
                    
                #---------forward prop-------------
                output,extracted_features = net(images)

                client_fx = extracted_features.clone().detach().requires_grad_(True)
                # Sending activations to server and receiving gradients from server
                time_client += time.time() - time_s
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, _)
                
                
                #--------backward prop -------------
                time_s = time.time()
                client_fx.backward(dfx)    
                optimizer_client.step()
                time_client += time.time() - time_s
                
                
                client_intermediate_data_size += (sys.getsizeof(client_fx.storage()) + 
                                      sys.getsizeof(labels.storage()))


        global intermediate_data_size
        intermediate_data_size += client_intermediate_data_size          
            
        
        # clients log
        # wandb.log({"Client{}_DcorLoss".format(idx): float(sum(Dcorloss_client_train)), "epoch": iter}, commit=False)
        wandb.log({"Client{}_time_not_scaled (s)".format(idx): time_client, "epoch": iter}, commit=False)
        wandb.log({"Client{}_sl_train_lr (s)".format(idx): optimizer_client.param_groups[0]['lr'], "epoch": iter}, commit=False)
        return net.state_dict(), time_client, client_intermediate_data_size 
    
    def evaluate(self, net, ell):
        net.eval()
        # 初始化客户端本地的评估指标
        local_correct = 0
        local_total = 0

           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------

                output,extracted_features = net(images)

                # 计算客户端本地模型的准确率
                _, local_predicted = torch.max(output.data, 1)
                local_total += labels.size(0)
                local_correct += (local_predicted == labels).sum().item()
            # Sending activations to server 
                evaluate_server(extracted_features, labels, self.idx, len_batch, ell)

            # 计算整体的本地准确率和损失
            local_accuracy = 100 * local_correct / local_total

        # 记录到wandb
        wandb.log({"Client{}_Local_Test_Accuracy".format(idx):local_accuracy, "epoch": ell}, commit=False)       
        return 

    # def evaluate_glob(self, net, ell): # I wrote this part
    #     net.eval()
    #     epoch_acc = []
    #     epoch_loss = []
           
    #     with torch.no_grad():
    #         batch_acc = []
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_test):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             #---------forward prop-------------
    #             fx = net(images)
    #             labels = labels.to(torch.long)
    #             loss = criterion(fx, labels)
    #             acc = calculate_accuracy(fx, labels)
    #             batch_loss.append(loss.item())
    #             batch_acc.append(acc.item())
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #         epoch_acc.append(sum(batch_acc)/len(batch_acc))
    #         prGreen('Model Test =>                     \tAcc: {:.3f} \tLoss: {:.4f}'
    #                 .format(epoch_acc[-1], epoch_loss[-1])) # After model update the test for all agent should be same. because the test dataset is same and after convergence all agents model are same
                
    #         return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
#=====================================================================================================        

def evaluate_global_model_on_all_test(net_glob, dataset_test, device, client_number,ell):
    """
    在所有客户端的测试集上评估全局模型
    Args:
        net_glob: 聚合后的全局模型
        dataset_test: 包含所有客户端测试集的字典
        device: 运行设备
        client_number: 客户端数量
    """
    net_glob.to(device)
    net_glob.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    # 移到函数开始处并初始化到正确的设备
    predictions_distribution = torch.zeros(10).to(device)  # 假设10个类别
    # 用于记录每个客户端的性能
    client_metrics = {}
    
    with torch.no_grad():
        for idx in range(client_number):
            client_correct = 0
            client_samples = 0
            client_loss = 0.0
            test_loader = dataset_test[idx]
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                output = net_glob(images)
                if isinstance(output, tuple):
                    output = output[0]
                
                # 计算损失
                loss = criterion(output, labels)
                client_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                client_samples += labels.size(0)
                client_correct += (predicted == labels).sum().item()
                predictions_distribution += torch.bincount(predicted, minlength=10)
            
            # 计算当前客户端的平均性能
            client_accuracy = 100.0 * client_correct / client_samples
            client_avg_loss = client_loss / len(test_loader)
            
            # 存储客户端的指标
            client_metrics[idx] = {
                'accuracy': client_accuracy,
                'loss': client_avg_loss,
                'samples': client_samples
            }
            
            # 累积总体指标
            total_correct += client_correct
            total_samples += client_samples
            total_loss += client_loss
            
            # 打印每个客户端的结果
            print(f"Client {idx} Test Accuracy: {100.0 * client_correct / client_samples:.2f}%")
    
    # 计算全局平均性能
    global_accuracy = 100.0 * total_correct / total_samples
    global_avg_loss = total_loss / total_samples
    
    # 打印总体结果
    print('\nGlobal Model Performance on All Test Sets:')
    print(f'Average Accuracy: {global_accuracy:.2f}%')
    print(f'Average Loss: {global_avg_loss:.4f}')
    print(f'Total Samples: {total_samples}')
    print(f"\nEpoch {ell} Global Test Accuracy: {global_accuracy:.2f}%")
    # 将分布转移到CPU并计算百分比
    dist_percent = (predictions_distribution / predictions_distribution.sum() * 100).cpu().numpy()
    print("Predictions distribution (%):")
    for i, percent in enumerate(dist_percent):
        print(f"Class {i}: {percent:.2f}%")
    # 记录到wandb
    wandb.log({"Global_Test_Accuracy":global_accuracy, "epoch": ell}, commit=False) 
    return 

               
#=====================================================================================================

def load_state_dict_for_aggregated_model(model, state_dict):
    """
    处理状态字典加载的函数，添加类型转换和结构匹配
    """
    print("\n=== 开始加载聚合后的模型权重 ===")
    
    # 1. 转换数据类型
    def convert_tensor_type(tensor):
        if tensor.dtype == torch.long:
            return tensor.float()
        return tensor

    # 2. 处理并打印原始状态字典信息
    print("\n原始状态字典信息:")
    processed_state_dict = {}
    for k, v in state_dict.items():
        # 转换数据类型
        v = convert_tensor_type(v)
        
        # 移除'module.'前缀
        name = k[7:] if k.startswith('module.') else k
        
        processed_state_dict[name] = v
        print(f"{k}: shape={v.shape}, "
              f"mean={v.mean().item():.6f}, std={v.std().item():.6f}, "
              f"max={v.max().item():.6f}, min={v.min().item():.6f}")

    # 3. 获取并打印当前模型状态
    current_dict = model.state_dict()
    print("\n当前模型状态字典信息:")
    for k, v in current_dict.items():
        print(f"{k}: shape={v.shape}, "
              f"mean={v.mean().item():.6f}, std={v.std().item():.6f}")

    # 4. 创建新的状态字典，保持模型结构
    new_state_dict = {}
    for k in current_dict.keys():
        if k in processed_state_dict:
            # 检查形状是否匹配
            if processed_state_dict[k].shape == current_dict[k].shape:
                new_state_dict[k] = processed_state_dict[k]
            else:
                print(f"\n警告: {k} 形状不匹配")
                print(f"期望: {current_dict[k].shape}")
                print(f"实际: {processed_state_dict[k].shape}")
                new_state_dict[k] = current_dict[k]  # 保持原始参数
        else:
            print(f"\n警告: 找不到键 {k}")
            new_state_dict[k] = current_dict[k]  # 保持原始参数

    try:
        # 5. 加载新的状态字典
        model.load_state_dict(new_state_dict)
        print("\n成功加载新的状态字典")
        
        # 6. 验证更新后的参数
        print("\n更新后的模型参数:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.mean().item():.6f}, "
                      f"std={param.std().item():.6f}")
                
                # 检查是否有异常值
                if torch.isnan(param).any():
                    print(f"警告: {name} 包含 NaN")
                if torch.isinf(param).any():
                    print(f"警告: {name} 包含 Inf")

    except Exception as e:
        print(f"\n错误: 加载状态字典失败 - {str(e)}")
        print("\n模型结构:")
        print(model)
        raise e

    print("\n=== 模型权重加载完成 ===")
    return model


#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          

# Data transmission
client_tier_all = []
client_tier_all.append(copy.deepcopy(client_tier))
total_training_time = 0
time_train_server_train_all_list = []

client_sample = np.ones(num_users)

def calculate_client_samples(train_data_local_num_dict, idxs_users, dataset):
    """
    Calculates the number of samples for each client in a federated dataset.

    Args:
        train_data_local_num_dict (dict): A dictionary mapping client indices to the number
            of samples in their local training dataset.
        num_users (int): The total number of clients in the federated dataset.
        dataset (str): The name of the federated dataset.

    Returns:
        A list of length num_users, where the i-th element represents the number of samples
        that the i-th client should use for training.
    """
    num_users = len(idxs_users)
    client_sample = []
    total_samples = sum(train_data_local_num_dict.values())
    for idx in idxs_users:
        client_sample.append(train_data_local_num_dict[idx] / total_samples * num_users)
    return client_sample
# 调试代码，分析客户端和服务器端模型架构
def debug_model_dimensions(w_locals_client_tier, w_locals_server_tier, client_sample):
    """
    调试函数：打印各层参数维度信息，添加类型检查
    """
    print("\n=== 模型维度调试信息 ===")
    
    # 检查输入类型
    print(f"\n数据结构类型:")
    print(f"w_locals_client_tier type: {type(w_locals_client_tier)}")
    print(f"w_locals_server_tier type: {type(w_locals_server_tier)}")
    
    # 检查客户端模型
    if isinstance(w_locals_client_tier, dict):
        for tier_idx, client_weights in w_locals_client_tier.items():
            print(f"\n客户端 Tier {tier_idx} 的参数维度:")
            if isinstance(client_weights, dict):
                for key, tensor in client_weights.items():
                    print(f"Layer: {key}, Shape: {tensor.shape if hasattr(tensor, 'shape') else 'No shape'}")
            else:
                print(f"Warning: client_weights for tier {tier_idx} is not a dict, type: {type(client_weights)}")
    else:
        print(f"Warning: w_locals_client_tier is not a dict, type: {type(w_locals_client_tier)}")
        # 如果是列表，尝试直接遍历
        if isinstance(w_locals_client_tier, list):
            for tier_idx, client_weights in enumerate(w_locals_client_tier):
                print(f"\n客户端 index {tier_idx} 的参数维度:")
                if isinstance(client_weights, dict):
                    for key, tensor in client_weights.items():
                        print(f"Layer: {key}, Shape: {tensor.shape if hasattr(tensor, 'shape') else 'No shape'}")
                else:
                    print(f"Warning: client_weights at index {tier_idx} is not a dict, type: {type(client_weights)}")
    
    # 检查服务器端模型
    if isinstance(w_locals_server_tier, dict):
        for tier_idx, server_weights in w_locals_server_tier.items():
            print(f"\n服务器端 Tier {tier_idx} 的参数维度:")
            if isinstance(server_weights, dict):
                for key, tensor in server_weights.items():
                    print(f"Layer: {key}, Shape: {tensor.shape if hasattr(tensor, 'shape') else 'No shape'}")
            else:
                print(f"Warning: server_weights for tier {tier_idx} is not a dict, type: {type(server_weights)}")
    else:
        print(f"Warning: w_locals_server_tier is not a dict, type: {type(w_locals_server_tier)}")
        # 如果是列表，尝试直接遍历
        if isinstance(w_locals_server_tier, list):
            for tier_idx, server_weights in enumerate(w_locals_server_tier):
                print(f"\n服务器端 index {tier_idx} 的参数维度:")
                if isinstance(server_weights, dict):
                    for key, tensor in server_weights.items():
                        print(f"Layer: {key}, Shape: {tensor.shape if hasattr(tensor, 'shape') else 'No shape'}")
                else:
                    print(f"Warning: server_weights at index {tier_idx} is not a dict, type: {type(server_weights)}")

    # 打印client_sample信息
    print("\nClient Sample信息:")
    print(f"Type: {type(client_sample)}")
    print(f"Content: {client_sample}")



def safe_shape_print(tensor):
    """安全地打印张量形状"""
    if hasattr(tensor, 'shape'):
        return str(tensor.shape)
    return f"No shape (type: {type(tensor)})"

def fix_state_dict_prefix(state_dict, target_model):
    """修复状态字典中的module前缀问题"""
    from collections import OrderedDict
    import torch.nn as nn
    
    # 确定目标模型是否为DataParallel
    is_data_parallel = isinstance(target_model, nn.DataParallel)
    
    # 检查状态字典是否包含module前缀
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    new_state_dict = OrderedDict()
    
    # 进行前缀调整
    if is_data_parallel and not has_module_prefix:
        # 目标是DataParallel但状态字典没有前缀，添加前缀
        print("添加module.前缀到状态字典")
        for k, v in state_dict.items():
            new_state_dict[f'module.{k}'] = v
    elif not is_data_parallel and has_module_prefix:
        # 目标不是DataParallel但状态字典有前缀，移除前缀
        print("从状态字典移除module.前缀")
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
    else:
        # 前缀状态已匹配，不需要修改
        return state_dict
    
    return new_state_dict


def load_model_safely(model, state_dict, preserve_classifier=True):
    """
    安全加载状态字典，处理模型结构不匹配问题
    
    Args:
        model: 目标模型
        state_dict: 源状态字典
        preserve_classifier: 是否保留原模型的分类器层参数
        
    Returns:
        加载结果
    """
    import torch
    import torch.nn as nn
    from collections import OrderedDict
    
    # 检查是否是DataParallel模型
    if isinstance(model, nn.DataParallel):
        target_model = model.module
    else:
        target_model = model
    
    # 获取当前模型的状态字典
    current_state = target_model.state_dict()
    
    # 创建新的状态字典
    new_state = OrderedDict()
    
    # 找出哪些键需要保留
    if preserve_classifier:
        classifier_keys = [k for k in current_state.keys() if 'classifier' in k or 'fc' in k]
    else:
        classifier_keys = []
    
    # 收集projection层的键
    projection_keys = [k for k in current_state.keys() if 'projection' in k]
    
    # 遍历当前模型的所有参数
    for key in current_state.keys():
        # 如果是分类器参数且需要保留，使用当前模型的参数
        if key in classifier_keys:
            new_state[key] = current_state[key]
        # 如果是projection参数
        elif key in projection_keys:
            # 如果源状态字典也有这个参数，使用源的
            if key in state_dict and state_dict[key].shape == current_state[key].shape:
                new_state[key] = state_dict[key]
            # 否则保留当前模型的参数
            else:
                new_state[key] = current_state[key]
        # 对于其他参数，尝试从源状态字典获取
        # elif key in state_dict and state_dict[key].shape == current_state[key].shape:
        elif key in state_dict :
            new_state[key] = state_dict[key]
        # 如果源状态字典中没有或形状不匹配，保留当前模型的参数
        else:
            new_state[key] = current_state[key]
            print(f"保留原参数: {key}")
    
    # 加载新的状态字典
    try:
        result = target_model.load_state_dict(new_state)
        print("模型参数安全加载完成")
        return result
    except Exception as e:
        print(f"加载状态字典时发生错误: {str(e)}")
        raise e
def analyze_model_layers(model_dict, name="未命名模型"):
    """
    分析模型的层结构，输出层前缀统计
    
    Args:
        model_dict: 模型状态字典
        name: 模型名称
        
    Returns:
        层前缀集合
    """
    if not isinstance(model_dict, dict):
        print(f"{name} 不是有效的状态字典")
        return set()
    
    # 统计层前缀
    layer_prefixes = {}
    
    for k in model_dict.keys():
        parts = k.split('.')
        if len(parts) >= 2:
            # 处理module前缀
            if parts[0] == 'module':
                if len(parts) >= 3:
                    prefix = f"{parts[0]}.{parts[1]}"
                    layer_name = parts[1]
                else:
                    prefix = parts[0]
                    layer_name = parts[0]
            else:
                prefix = parts[0]
                layer_name = parts[0]
                
            if layer_name not in layer_prefixes:
                layer_prefixes[layer_name] = 0
            layer_prefixes[layer_name] += 1
    
    print(f"\n{name} 层结构分析:")
    print(f"总参数数量: {len(model_dict)}")
    
    print("层统计:")
    for layer_name, count in sorted(layer_prefixes.items()):
        print(f"  {layer_name}: {count} 个参数")
    
    # 返回唯一层前缀集合
    return set(layer_prefixes.keys())
# for i in range(0, num_users):
#     wandb.log({"Client{}_Tier".format(i): num_tiers - client_tier[i] + 1, "epoch": -1}, commit=False)

#------------ Training And Testing  -----------------
net_glob_client.train()
w_glob_client_tier ={}


#copy weights
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()


# net_glob_client_tier[tier].load_state_dict(w_glob_client)
w_glob_client_tier[tier] = net_glob_client_tier[tier].state_dict()
     

# to start with same weigths 
for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)

# init_glob_model方法构造了一个初始化的全局模型，固定了tier=1，权重记录在W_glob中
w_glob = copy.deepcopy(init_glob_model.state_dict())
# net_glob_fed构造一个初始化的全局模型，用来记录聚合后的完整模型
net_glob_fed = copy.deepcopy(init_glob_model)

# 对于w_glob_client_tier[tier] 进行初始化操作，让不同tier的模型参数一致，但是我们改变了classifier的构造，因此w_glob的参数设置不能适用于所有客户端或服务器
for t in range(1, num_tiers+1):
    for k in w_glob_client_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            k1 = k1[7:] # remove the 'module.' prefix
                    
        # # if k1.startswith('classifier'):
        # if 'classifier' in k1:
        #     continue 
        if 'projection' in k1:
            continue
        
        w_glob_client_tier[t][k] = w_glob[k1]
    for k in w_glob_server_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            k1 = k1[7:]

        # # if k1.startswith('classifier'):
        # if 'classifier' in k1:
        #     continue 

        w_glob_server_tier[t][k] = w_glob[k1]

  
    net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
    net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
    
w_locals_tier, w_locals_client, w_locals_server = [], [], []


# w_glob_client = init_glob_model.state_dict() # copy weights
# net_glob_client_tier[tier].load_state_dict(w_glob_client) # copy weights
net_model_client_tier = {}
for i in range(1, num_tiers+1):
    net_model_client_tier[i] = net_glob_client_tier[i]
    net_model_client_tier[i].train()
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()


# optimizer for every client
optimizer_client_tier = {}
for i in range(0, num_users): # one optimizer for every tier/ client
    if args.optimizer == "Adam":
        optimizer_client_tier[i] =  torch.optim.Adam(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
    elif args.optimizer == "SGD":
        optimizer_client_tier[i] =  torch.optim.SGD(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, momentum=0.9,
                                                          nesterov=True,
                                                          weight_decay=args.wd)


# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

simulated_delay_historical_df = pd.DataFrame()
start_time = time.time() 

client_observed_times = pd.DataFrame()
torch.manual_seed(SEED)
simulated_delay= np.zeros(num_users)

for i in range(0, num_users): # maybe remove this part
    # continue
    data_server_to_client = 0
    for k in w_glob_client_tier[client_tier[i]]:
        data_server_to_client += sys.getsizeof(w_glob_client_tier[client_tier[i]][k].storage())
    simulated_delay[i] = data_server_to_client / net_speed[i]

# Generate a list of randomly chosen user indices based on the number of users and the default fraction
idxs_users, m = get_random_user_indices(num_users, DEFAULT_FRAC)
print(f"users idx in train is{idxs_users}")

# record all data transmitted in last involeved epoch
data_transmitted_client_all = {}

# computation_time_clients = {}
# for k in range(num_users):
#     computation_time_clients[k] = []

# net_glob_client_cluster 记录了各个cluster下的客户端全局模型
net_glob_client_cluster = {}
w_glob_client_cluster = {}
num_clusters = 3  # Example number of clusters
# 初始化net_glob_client_cluster，假设初始化的客户端模型tier=1
for i in range (num_clusters) :
    net_glob_client_cluster[i],_ = SFL_local_tier(classes=class_num,tier=1) 
    w_glob_client_cluster[i] = net_glob_client_cluster[i].state_dict()

print("-----------------------------------------------------------")
print("{:^59}".format("Client Pre-train"))
print("-----------------------------------------------------------")
# client_states记录了各个客户端预训练结束后返回的本地模型参数
client_states = []
# w_locals_client_idx记录了每个各个客户端的本地模型参数
net_locals_client_idx = {}
w_locals_client_idx = {}
w_locals_client_idx = {i: [] for i in range(num_users)}

# 客户端初始化（只初始化一次）
# 首先初始化clients列表大小为最大客户端数
max_clients = max(idxs_users) + 1
clients = [None] * max_clients

for i, idx in enumerate(idxs_users):
    data_server_to_client = calculate_data_size(w_glob_client_tier[client_tier[idx]])
    simulated_delay[idx] = data_server_to_client / net_speed[idx]            
           
    client_model_parameter_data_size = 0
    time_train_test_s = time.time()
    
    # 按照客户端tier级别分配客户端模型
    net_glob_client = net_model_client_tier[client_tier[idx]]
    net_locals_client_idx[idx] = copy.deepcopy(net_glob_client)
    # w_glob_client_tier[client_tier[idx]] = net_glob_client_tier[client_tier[idx]].state_dict()
    
    # 创建client对象并直接放入对应索引位置
    client = Client(
        net_glob_client, 
        idx=idx,  # 保持idx不变
        lr=lr, 
        device=device, 
        dataset_train=dataset_train, 
        dataset_test=dataset_test, 
        idxs=[], 
        idxs_test=[]
    )
    
    # 验证客户端对象创建是否正确
    print(f"Created client object with idx: {client.idx}")
    print(f"Client tier: {client_tier[idx]}")
    
    # 直接将client放入其idx对应的位置
    clients[idx] = client
    
    # 预训练阶段
    [w_pretrained, time_pretrained] = client.pre_train(
        net=copy.deepcopy(net_glob_client).to(device)
    )
        
    client_states.append(w_pretrained)
    w_locals_client_idx[idx] = copy.deepcopy(w_pretrained)
    net_locals_client_idx[idx].load_state_dict(w_pretrained)

# 创建评估数据集
eval_dataset = get_cifar10_proxy_dataset(
    option='balanced_test',  # 类别平衡的测试集
    num_samples=1000,        # 使用1000个样本
    seed=42                  # 固定随机种子确保可重复性
)

# 将字典转换为模型列表，仅包含已训练的客户端模型
client_models = []
client_indices = []  # 保存对应的客户端索引，以便于之后映射回去

for idx in idxs_users:
    if idx in net_locals_client_idx:
        # 确保模型在列表中的位置与客户端索引对应关系记录下来
        client_models.append(net_locals_client_idx[idx])
        client_indices.append(idx)

# 使用模型列表进行数据分布感知的聚类
labels, cluster_info = data_distribution_aware_clustering(
    client_models,          # 正确的模型对象列表
    None,                   # 不使用客户端数据集
    eval_dataset,           # 使用创建的评估数据集
    n_clusters=3,                   # 聚类数量
    device=device
)

# Record classification results记录客户端分类结果cluster_count记录每个类别的客户端总数，client_clusters记录每个类别的客户端索引
cluster_count = {}
client_clusters = {}

# 初始化字典
for cluster_id in set(labels):
    cluster_count[cluster_id] = 0
    client_clusters[cluster_id] = []

# 将聚类结果映射回客户端索引
client_cluster_map = {}
for i, label in enumerate(labels):
    client_idx = client_indices[i]
    client_cluster_map[client_idx] = label
    
    # 更新聚类计数
    cluster_count[label] += 1
    
    # 将客户端添加到对应的聚类列表中
    client_clusters[label].append(client_idx)

# 打印聚类结果
print("\n客户端聚类结果:")
for cluster_id in sorted(client_clusters.keys()):
    print(f"聚类 {cluster_id}: 包含 {cluster_count[cluster_id]} 个客户端")
    print(f"  客户端索引: {client_clusters[cluster_id]}")
    
    # 打印每个聚类中客户端的tier分布
    tier_distribution = {}
    for idx in client_clusters[cluster_id]:
        tier = client_tier[idx]
        if tier not in tier_distribution:
            tier_distribution[tier] = 0
        tier_distribution[tier] += 1
    
    print(f"  Tier分布: {tier_distribution}")


# 聚合预训练阶段的客户端模型，作为训练阶段的初始化本地模型
# calculate the number of samples in each client
client_sample = calculate_client_samples(train_data_local_num_dict, idxs_users, args.dataset) # same order as appended weights

# 对客户端本地模型，按照分类情况进行聚合，结果保留在aggregated_clients_models中
aggregated_pretrained_models = {}
aggregated_pretrained_models = aggregate_clients_models(client_clusters, w_locals_client_idx, num_clusters, whether_local_loss, client_sample, idxs_users)

# 更新预训练阶段的聚合结果，记录在net_glob_client_cluster中，在训练阶段会下发给各个客户端
for cluster_id in range(num_clusters):
    print(f"\nProcessing aggregation for cluster {cluster_id}")
    if cluster_id not in aggregated_pretrained_models:
        print(f"Warning: Cluster {cluster_id} not found in aggregated models")
    if cluster_id in aggregated_pretrained_models:
        # 获取当前模型的状态字典结构
        current_model = net_glob_client_cluster[cluster_id]
        current_state_dict = current_model.state_dict()
        
        # 修复前缀问题
        fixed_state_dict = fix_state_dict_prefix(
            aggregated_pretrained_models[cluster_id],
            current_model
        )

        # 使用安全加载函数加载聚合后的状态字典
        load_model_safely(
            net_glob_client_cluster[cluster_id], 
            fixed_state_dict,
            preserve_classifier=True
        )

    # 更新存储的状态字典
    w_glob_client_cluster[cluster_id] = net_glob_client_cluster[cluster_id].state_dict()



for iter in range(epochs):

    
    # Initialize empty lists for client weights
    w_locals_client = []
    w_locals_client_tier = {}
    
    # Initialize a dictionary to store client weights based on their tiers
    w_locals_client_tier = {i: [] for i in range(1, num_tiers+1)}
    
    # Initialize a numpy array to store client time
    client_observed_time = np.zeros(num_users)
    
    # processes = []
    
    simulated_delay= np.zeros(num_users)

    
    #清空w_locals_client_idx，开始正式训练阶段
    w_locals_client_idx = {}

    

    for idx in idxs_users:
        print(f"\nClient {idx}: tier = {client_tier[idx]}")
        data_server_to_client = calculate_data_size(w_glob_client_tier[client_tier[idx]])
        simulated_delay[idx] = data_server_to_client / net_speed[idx]
        
        # 获取客户端所在的聚类和模型
        current_cluster = client_cluster_map[idx]
        net_glob_client = net_locals_client_idx[idx]
        
        if current_cluster not in net_glob_client_cluster:
            print(f"Warning: Cluster {current_cluster} not found in net glob client cluster")
            continue
        
        # 使用安全加载函数直接从聚类模型加载参数到客户端模型
        print(f"为客户端 {idx} (聚类 {current_cluster}) 加载聚类模型参数")
        load_model_safely(
            net_glob_client,
            net_glob_client_cluster[cluster_id].state_dict(),
            preserve_classifier=True  # 保留分类器参数
        )
        
        # 更新全局模型参数（这样w_glob_client_tier也保持最新状态）用于计算服务器向客户端传输的数据量
        w_glob_client_tier[client_tier[idx]] = net_glob_client.state_dict()
        
        client = clients[idx]  # 使用已存在的客户端实例

        # 本地训练5轮
        local_train_epoch = 5
        [Client_local_model, Client_train_time] = client.local_train(
                net=copy.deepcopy(net_glob_client).to(device), 
                local_train_epoch=local_train_epoch
        )
        # 更新客户端模型
        net_glob_client.load_state_dict(Client_local_model)
        # Training ------------------
        # 拆分学习训练
        print(f"client{idx} train is start")
        [w_client, duration, client_intermediate_data_size] = client.train(
            net=copy.deepcopy(net_glob_client).to(device)
        )
        print(f"client{idx} train is done")

        w_locals_client_idx[idx] = copy.deepcopy(w_client)    
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_client_tier[client_tier[idx]].append(copy.deepcopy(w_client))
        
        # Testing -------------------  
        net = copy.deepcopy(net_glob_client)
        w_previous = copy.deepcopy(net.state_dict())  # to test for updated model
        net.load_state_dict(w_client)
        net.to(device)
            
        # local.evaluate(net, ell= iter)
        client.evaluate(net, ell= iter)
        net.load_state_dict(w_previous)

        client_observed_time[idx] = duration
        
        
        client_model_parameter_data_size = calculate_data_size(w_client)
        model_parameter_data_size += client_model_parameter_data_size         
        
        data_transmitted_client = client_intermediate_data_size + client_model_parameter_data_size
        
        # add to dic last observation
        data_transmitted_client_all[idx] = data_transmitted_client
        
        simulated_delay[idx] += compute_delay(data_transmitted_client, net_speed[idx]
                                              , delay_coefficient[idx], duration) # this is simulated delay

        wandb.log({"Client{}_Total_Delay".format(idx): simulated_delay[idx], "epoch": iter}, commit=False)
        
    server_wait_first_to_last_client = (max(simulated_delay * client_epoch) - min(simulated_delay * client_epoch))
    training_time = (max(simulated_delay)) 
    total_training_time += training_time
    if iter == 0:
        first_training_time = training_time
    wandb.log({"Training_time_clients": total_training_time, "epoch": iter}, commit=False)
    times_in_server = []
    time_train_server_train_all_list.append(time_train_server_train_all)
    time_train_server_train_all = 0
     
    simulated_delay[simulated_delay==0] = np.nan  # convert zeros to nan, for when some clients not involved in the epoch
    simulated_delay_historical_df = pd.concat([simulated_delay_historical_df, pd.DataFrame(simulated_delay).T], ignore_index=True)
    client_observed_times = pd.concat([client_observed_times, pd.DataFrame(client_observed_time).T], ignore_index=True)
    client_epoch_last = client_epoch.copy()
    
    # idxs_users, m = get_random_user_indices(num_users, DEFAULT_FRAC)

                                                    
    client_tier_all.append(copy.deepcopy(client_tier))
    
    # 定义需要跳过的分类器层
    classifier_layers = ['classifier.fc1.weight', 'classifier.fc1.bias', 
                    'classifier.fc2.weight', 'classifier.fc2.bias',
                    'classifier.fc3.weight', 'classifier.fc3.bias']
    
    # for i in client_tier.keys():  # assign each server-side to its tier model
    #     net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]

    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("{:^59}".format("Model Aggregation"))
    print("-----------------------------------------------------------")
    
    # calculate the number of samples in each client
    client_sample = calculate_client_samples(train_data_local_num_dict, idxs_users, args.dataset) # same order as appended weights

    # 对客户端本地模型，按照分类情况进行聚合，结果保留在aggregated_clients_models中
    aggregated_clients_models = {}
    aggregated_clients_models = aggregate_clients_models(client_clusters, w_locals_client_idx, num_clusters, whether_local_loss, client_sample, idxs_users)
    
    # 更新 w_glob_client_cluster，将类内的客户端聚合模型加载到net_glob_client_cluster中
    for cluster_id in range(num_clusters):
        print(f"\nLoading aggregation client model for cluster {cluster_id}")
        if cluster_id not in aggregated_clients_models:
            print(f"Warning: Cluster {cluster_id} not found in aggregated models")
        if cluster_id in aggregated_clients_models:
            # 获取当前模型的状态字典结构
            current_model = net_glob_client_cluster[cluster_id]
            current_state_dict = current_model.state_dict()
            
            # 修复前缀问题
            fixed_state_dict = fix_state_dict_prefix(
                aggregated_clients_models[cluster_id],
                current_model
            )

            # 使用安全加载函数加载聚合后的状态字典
            load_model_safely(
                net_glob_client_cluster[cluster_id], 
                fixed_state_dict,
                preserve_classifier=True
            )
        # 更新存储的状态字典
        w_glob_client_cluster[cluster_id] = net_glob_client_cluster[cluster_id].state_dict()

    # print("\nBefore aggregation:")
    # # 抽样打印一些客户端模型的参数
    # for idx in idxs_users[:2]:  # 打印前两个客户端的信息
    #     print(f"\nClient {idx} parameters:")
    #     for k in list(w_locals_client[idx].keys())[:3]:  # 打印前三个层的参数
    #         print(f"{k}: mean={w_locals_client[idx][k].mean().item():.4f}, std={w_locals_client[idx][k].std().item():.4f}")

    
    # # 在聚合前分析模型结构
    # if len(w_locals_tier) > 0:
    #     analyze_model_layers(w_locals_tier[0], "服务器模型[0]")

    # 聚合函数
    w_glob = aggregated_fedavg(
        w_locals_tier, 
        w_locals_client, 
        num_tiers, 
        num_users, 
        whether_local_loss, 
        client_sample, 
        idxs_users,
        target_device='cpu'  # 或根据需要选择其他设备
    )
    
    # # 分析聚合后的模型
    # analyze_model_layers(w_glob, "聚合后的全局模型")


    # 更新之前
    print("\nBefore updating global model:")
    for t in range(1, num_tiers+1):
        print(f"\nTier {t} parameters:")
        for k in list(net_glob_server_tier[t].state_dict().keys())[:3]:
            param = net_glob_server_tier[t].state_dict()[k]
            print(f"{k}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")


    # 聚合后的模型更新到服务器端
    
    for t in range(1, num_tiers+1):
        for k in w_glob_server_tier[t].keys():
            if k in w_glob:
                if w_glob_server_tier[t][k].shape == w_glob[k].shape:
                    w_glob_server_tier[t][k] = w_glob[k]
                else:
                    print(f"警告: w_glob_server_tier {t} 形状不匹配 (key {k})，保留原参数")
                    continue
            else:
                print(f"警告: w_glob 中没有键 {k}，保留原参数")
                continue
        
        net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])

    # for i in client_tier.keys():  # assign each server-side to its tier model
    #     net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]
    clients_updated = 0
    for client_idx, tier in client_tier.items():
        if client_idx not in net_model_server_tier:
            print(f"警告: 客户端 {client_idx} 在net_model_server_tier中不存在，跳过")
            continue
            
        # 从更新后的全局tier服务器模型复制
        if tier in net_glob_server_tier:
            # 深拷贝更新后的tier服务器模型
            net_model_server_tier[client_idx] = copy.deepcopy(net_glob_server_tier[tier])
            clients_updated += 1
            # 只打印前几个客户端的更新信息，避免过多输出
            if clients_updated <= 5:
                print(f"更新客户端 {client_idx} (tier {tier}) 的服务器模型")
    
    print(f"总共更新了 {clients_updated} 个客户端的服务器模型")
    print("=== 服务器模型更新完成 ===")

    # 更新之后
    print("\nAfter updating global model:")
    for t in range(1, num_tiers+1):
        print(f"\nTier {t} parameters:")
        for k in list(net_glob_server_tier[t].state_dict().keys())[:3]:
            param = net_glob_server_tier[t].state_dict()[k]
            print(f"{k}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

    # 在加载全局模型前
    print("\nBefore testing global model:")
    for k in list(net_glob_fed.state_dict().keys())[:3]:
        param = net_glob_fed.state_dict()[k]
        print(f"{k}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    # 获取当前模型的状态字典结构
    current_model = net_glob_fed
    current_state_dict = current_model.state_dict()
            
    # 修复前缀问题
    fixed_state_dict = fix_state_dict_prefix(
        w_glob,
        current_model
    )
    # 测试聚合后的全局模型 在测试数据集上的准确率
    load_model_safely(
        net_glob_fed,
        fixed_state_dict,
        preserve_classifier=True
    )
    # 在加载全局模型后
    print("\nAfter testing global model:")
    for k in list(net_glob_fed.state_dict().keys())[:3]:
        param = net_glob_fed.state_dict()[k]
        print(f"{k}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
#     
#     try:
#         net_glob_fed = load_state_dict_for_aggregated_model(net_glob_fed, w_glob)
#         print("Successfully loaded aggregated state dict")

#         # 验证权重是否更新
#         print("\nVerifying weight updates:")
#         for name, param in net_glob_fed.named_parameters():
#             if param.requires_grad:
#                 print(f"{name}: mean={param.mean().item():.6f}, "
#                 f"std={param.std().item():.6f}")
#     except Exception as e:
#         print(f"Error in loading aggregated state dict: {str(e)}")
    
#     print("\nTesting aggregated model on all clients' test sets...")

#     # 在评估全局模型前
#     print("\nBefore testing global model:")
#     for k in list(net_glob_fed.state_dict().keys())[:3]:
#         param = net_glob_fed.state_dict()[k]
#         print(f"{k}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")


    evaluate_global_model_on_all_test(net_glob=net_glob_fed,
        dataset_test=dataset_test,
        device=device,
        client_number=client_number,
        ell=iter
    )

 

    
    print(f'Size of Total Model Parameter Data Transferred {(model_parameter_data_size/1024**2):,.2f} Mega Byte')
    print(f'Size of Total Intermediate Data Transferred {(intermediate_data_size/1024**2):,.2f} Mega Byte')

    wandb.log({"Model_Parameter_Data_Transmission(MB) ": model_parameter_data_size/1024**2, "epoch": iter}, commit=False)
    wandb.log({"Intermediate_Data_Transmission(MB) ": intermediate_data_size/1024**2, "epoch": iter}, commit=True)
    
    
elapsed = (time.time() - start_time)/60
    
#===================================================================================     

print("Training and Evaluation completed!")    
    

#=============================================================================
#                         Program Completed
#=============================================================================

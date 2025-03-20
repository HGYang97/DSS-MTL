import argparse
import os
import sys
from datetime import datetime
import torch.nn as nn

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model, base_Model_addV
from trainer.trainer import Trainer, Trainer_ft, Trainer_self_supervised_decoder, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics, copy_Files
from utils import _logger, set_requires_grad
from generate import optimize_matrices

start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='DC_experiments', type=str, help='Experiment Description')
parser.add_argument('--run_description', default='test1_multi_task', type=str, help='Experiment Description')
parser.add_argument('--ft_perc', default='100p', type=str, help='Proportion of labeled data')
parser.add_argument('--seed', default=0, type=int, help='seed value')
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels')

parser.add_argument('--selected_dataset', default='CER_300_450_4704', type=str,
                    help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--data_path', default=r'data/', type=str, help='Path containing dataset')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description

ft_perc = args.ft_perc


logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = os.path.join(args.data_path, data_type)  # 定义数据集地址
train_dl, valid_dl, test_dl = data_generator(data_path, configs,
                                             training_mode)  # 生成三个数据读取器,每个读取器返回的是x，y1,y2,y3，aug1和aug2
logger.debug("Data loaded ...")

# Load Model
'''
model是自监督的encoder部分，模型的输出有两项logits, x。
# logits 的形状应该是 (batch_size, num_classes)，其中 batch_size 是输入数据的批次大小，num_classes 是分类的类别数量。这个输出通常用于计算交叉熵损失。
#这是输入数据经过三个卷积块处理后的结果，它保留了一些中间层的信息。这个输出的形状应该是 (batch_size, final_out_channels, features_len)，
其中 final_out_channels 是最后一个卷积块的输出通道数，features_len 是特征长度。这个输出可能用于某些需要中间层输出的场景
'''

if (training_mode == 'self_supervised') or (training_mode == f'train_linear_{ft_perc}') or (training_mode == f'ft_{ft_perc}_1p_and_generateV') or (training_mode == f'random_init_{ft_perc}'):
    model = base_Model(configs).to(device)

if (training_mode == 'self_supervised_decoder') or (training_mode == f're_ft_{ft_perc}') or (training_mode == 'gen_pseudo_labels') or (training_mode == 'SupCon') or (training_mode == f'train_linear_SupCon_{ft_perc}'):
    # ft_perc = '1p'
    A_optimized = torch.load(
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}",
                     'V1.pt')).to(device)
    B_optimized = torch.load(
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}",
                     'V2.pt')).to(device)
    C_optimized = torch.load(
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}",
                     'V3.pt')).to(device)
    model = base_Model_addV(configs, V1=A_optimized, V2=B_optimized, V3=C_optimized).to(device)

temporal_contr_model = TC(configs, device).to(device)

if "fine_tune" in training_mode or "ft_" in training_mode:
    # load saved model of this experiment
    # ft_perc = '1p'
    if training_mode == f"ft_{ft_perc}_and_generateV":
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                           weight_decay=3e-4)

        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                    betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        target_cosine_similarity_12, target_cosine_similarity_13, target_cosine_similarity_23 = Trainer_ft(model,
                                                                                                           temporal_contr_model,
                                                                                                           model_optimizer,
                                                                                                           temporal_contr_optimizer,
                                                                                                           train_dl,
                                                                                                           valid_dl,
                                                                                                           test_dl,
                                                                                                           device,
                                                                                                           logger,
                                                                                                           configs,
                                                                                                           experiment_log_dir,
                                                                                                           training_mode)
        # print(type(target_cosine_similarity_12))
        target_cosine_similarity_12 = target_cosine_similarity_12.detach().numpy()
        target_cosine_similarity_13 = target_cosine_similarity_13.detach().numpy()
        target_cosine_similarity_23 = target_cosine_similarity_23.detach().numpy()

        A_optimized, B_optimized, C_optimized = optimize_matrices(
            target_cosine_similarity_12, target_cosine_similarity_13, target_cosine_similarity_23, configs.features_len,
            configs.batch_size)

        torch.save(A_optimized, os.path.join(experiment_log_dir, 'V1.pt'))
        torch.save(B_optimized, os.path.join(experiment_log_dir, 'V2.pt'))
        torch.save(C_optimized, os.path.join(experiment_log_dir, 'V3.pt'))

        '''
        下面是测试V，x融合时的代码
        '''
        # A_optimized = torch.load(os.path.join(experiment_log_dir, 'V1.pt'))
        # B_optimized = torch.load(os.path.join(experiment_log_dir, 'V2.pt'))
        # C_optimized = torch.load(os.path.join(experiment_log_dir, 'V3.pt'))
        # print("V is ",A_optimized.shape)
        #
        # model = base_Model_addV(configs, V1=A_optimized, V2=B_optimized, V3=C_optimized).to(device)
        # Trainer_ft(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl,
        #         device,
        #         logger, configs, experiment_log_dir, training_mode)

        # print("Matrix A:", A_optimized)
        # print("Matrix B:", B_optimized)
        # print("Matrix C:", C_optimized)
        # print("Cosine Similarity AB:", cosine_similarity_AB)
        # print("Cosine Similarity AC:", cosine_similarity_AC)
        # print("Cosine Similarity BC:", cosine_similarity_BC)
    elif training_mode == f"re_ft_{ft_perc}":
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_decoder_seed_{SEED}",
                         "saved_models"))

        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

if training_mode == "gen_pseudo_labels":
    # ft_perc = "1p"
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"re_ft_{ft_perc}_seed_{SEED}",
                     "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)

    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)
    gen_pseudo_labels(model, train_dl, device, data_path)
    sys.exit(0)

if "train_linear" in training_mode or "tl" in training_mode:
    '''
    这两种模式都是加载encoder模型，然后训练分类层
    '''
    if 'SupCon' not in training_mode:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"),
                          map_location=device)  # 读取training_mode == "self_supervised"自监督中训练的encoder模型
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()  # 读取模型的状态或者说是权重

    # 1. filter out unnecessary keys 这段代码的作用是过滤掉预训练模型状态字典中那些在当前模型状态字典中不存在的键值对，以确保只有相匹配的键值对被保留下来
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 删除这些参数（例如：末端的线性层）
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    '''
    1. model_dict.update(pretrained_dict)将预训练模型的状态字典pretrained_dict更新到当前模型的状态字典model_dict中。这将用预训练的权重更新当前模型的权重。
    2. model.load_state_dict(model_dict)将更新后的状态字典加载到模型中，以更新模型的权重。

    def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

    3. set_requires_grad(model, pretrained_dict, requires_grad=False)调用set_requires_grad函数，将模型中除了最后一层以外的所有层的requires_grad属性设置为False，即冻结这些层的参数。
    4. set_requires_grad(model, dict_, requires_grad=True)是一个自定义函数，用于设置模型中特定参数的requires_grad属性。
    5. 在set_requires_grad函数中，通过model.named_parameters()遍历模型中的参数。
    6. 对于每个参数，判断其名称是否存在于dict_中。
    7. 如果参数的名称存在于dict_中，即该参数是需要进行梯度更新的参数，那么将其requires_grad属性设置为requires_grad参数指定的值
    这段代码的作用是将预训练模型的权重更新到当前模型中，并冻结除最后一层以外的所有层，以便在训练过程中只更新最后一层的参数。
    '''
    model_dict.update(pretrained_dict)  #
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == f"random_init_{ft_perc}":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "SupCon":
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"re_ft_{ft_perc}_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)

model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=3e-4)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised" or training_mode == "SupCon":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)
if training_mode == "self_supervised_decoder":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)
    # ft_perc = "1p"
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}",
                     "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)

    for param in model.conv_block1.parameters():
        param.requires_grad = False
    for param in model.conv_block2.parameters():
        param.requires_grad = False
    for param in model.conv_block3.parameters():
        param.requires_grad = False

    Trainer_self_supervised_decoder(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl,
                                    valid_dl, test_dl, device,
                                    logger, configs, experiment_log_dir, training_mode)
# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode)

if training_mode != "self_supervised" and training_mode != "SupCon" and training_mode != "SupCon_pseudo" and training_mode != "self_supervised_decoder":
    # Testing

    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    # total_loss, total_acc, pred_label1, true_label1 = outs
    # _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")

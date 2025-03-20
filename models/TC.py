import numpy as np
import torch
import torch.nn as nn

from .attention import Seq_Transformer


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        '''
        定义投影头，作用是将输入数据从 configs.TC.hidden_dim 维度映射到 configs.final_out_channels // 4 维度
        '''
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )
        '''
        文中自定义的那个transformer模块，接受特征序列并输出一个预测值c_t
        '''
        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4,
                                               heads=4, mlp_dim=64)

    def forward(self, z_aug1, z_aug2):
        seq_len = z_aug1.shape[2] #(batch_size,chennel_num,seq_len)

        z_aug1 = z_aug1.transpose(1, 2)#(batch_size,seq_len,chennel_num)
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]

        '''
        函数生成一个随机整数张量，范围在 [0, seq_len - self.timestep) 之间。这个随机整数表示了从输入数据中随机选择的时间戳的起始位置
        '''
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(
            self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch 时间步长和批次的平均值

        '''
        创建了一个空的三维张量 encode_samples，形状为 (self.timestep, batch, self.num_channels)。这个张量将用于存储encoder层的输出
        '''
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        '''
        生成编码样本
        for i in np.arange(1, self.timestep + 1):：这个循环从 1 到 self.timestep（包括 self.timestep）遍历。
        self.timestep 表示编码样本的时间步数。
        encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)：
        在每次循环迭代中，将 z_aug2 中的特定时间戳的数据切片（[:, t_samples + i, :]），并使用 view 方法将其形状调整为 (batch, self.num_channels)。
        然后，将调整后的数据存储在 encode_samples 的相应位置（i - 1）上。
        forward_seq = z_aug1[:, :t_samples + 1, :]：这行代码将 z_aug1 中的数据切片，从起始位置到 t_samples + 1 的时间戳，
        并将结果存储在 forward_seq 中。这个切片将用作后续处理的输入。
        综合起来，这个循环的作用是根据随机选择的时间戳，从输入数据中提取编码样本。
        这些编码样本将用于后续的计算和处理。同时，forward_seq 是一个切片，用于存储输入数据的子序列，以便在模型的前向传播过程中使用。
        '''
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :] #aug的前n个时间戳的数据

        c_t = self.seq_transformer(forward_seq)#aug的前n个时间戳的数据送入trans，输出一个C_t


        '''
        linear = self.Wk[i]：在每次循环迭代中，从 self.Wk 中获取第 i 个线性层对象，并将其赋值给变量 linear。
        pred[i] = linear(c_t)：使用线性层对象 linear 对输入 c_t 进行线性变换，得到预测结果。将预测结果存储在 pred 的相应位置（i）上,i为第一个维度。
        '''
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        '''
        计算 NCE 损失，并将计算得到的损失值和经过投影头处理后的结果作为输出返回
        '''
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)

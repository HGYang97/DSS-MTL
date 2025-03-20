from torch import nn
import torch
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits1 = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes1)
        self.logits2 = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes2)
        self.logits3 = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes3)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.shape)

        x_flat = x.reshape(x.shape[0], -1)
        logits1 = self.logits1(x_flat)
        logits2 = self.logits2(x_flat)
        logits3 = self.logits3(x_flat)
        return logits1, logits2, logits3, x

class base_Model_addV(nn.Module):
    def __init__(self, configs, V1, V2, V3):
        super(base_Model_addV, self).__init__()

        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits1 = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes1)
        self.logits2 = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes2)
        self.logits3 = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes3)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.shape)
        # print("V is ", self.V1.shape)
        # print("x is ", x.shape)
        # self.V1 = self.V1.repeat(x.shape[0], 1, 1)
        V1 = self.V1[:x.size(0)]
        V2 = self.V2[:x.size(0)]
        V3 = self.V3[:x.size(0)]
        x1 = torch.mul(x, V1)
        x2 = torch.mul(x, V2)
        x3 = torch.mul(x, V3)

        # print("V is ", self.V1.shape)
        # print("x is ", x.shape)
        # print("x1 is ", x1.shape)

        x1_flat = x1.reshape(x1.shape[0], -1)
        x2_flat = x2.reshape(x2.shape[0], -1)
        x3_flat = x3.reshape(x3.shape[0], -1)

        logits1 = self.logits1(x1_flat)
        logits2 = self.logits2(x2_flat)
        logits3 = self.logits3(x3_flat)

        return logits1, logits2, logits3, x
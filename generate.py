import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import minimize

import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.optimize import minimize

def optimize_matrices(AB, AC, BC, n_size, batch_size):
    # 目标余弦相似度
    target_cosine_similarity_AB = AB
    target_cosine_similarity_AC = AC
    target_cosine_similarity_BC = BC

    # print(type(AB))

    # 初始矩阵A和B，以及空的矩阵C
    n = n_size
    A = np.random.rand(n)
    B = np.random.rand(n)
    C = np.random.rand(n)

    # A = A.detach().numpy()
    # B = B.detach().numpy()
    # C = C.detach().numpy()

    print(type(A))

    # 定义损失函数，即余弦相似度与目标相似度之间的差值的平方
    def lossddd(x):
        A, B, C = x[:n], x[n:2*n], x[2*n:]
        loss_AB = (cosine(A, B) - target_cosine_similarity_AB) ** 2
        loss_AC = (cosine(A, C) - target_cosine_similarity_AC) ** 2
        loss_BC = (cosine(B, C) - target_cosine_similarity_BC) ** 2
        return loss_AB + loss_AC + loss_BC

    # 使用优化算法来最小化损失函数，以调整矩阵A、B和C的值
    result = minimize(lossddd, np.concatenate((A, B, C)), method='L-BFGS-B')

    # 从优化结果中获取生成的矩阵A、B和C
    A_optimized = result.x[:n]
    B_optimized = result.x[n:2*n]
    C_optimized = result.x[2*n:]

    # Convert numpy arrays to tensors
    A_optimized = torch.tensor(A_optimized).float()
    A_optimized = A_optimized.unsqueeze(0)
    A_optimized = A_optimized.unsqueeze(0).expand(batch_size, -1, -1)

    B_optimized = torch.tensor(B_optimized).float()
    B_optimized = B_optimized.unsqueeze(0)
    B_optimized = B_optimized.unsqueeze(0).expand(batch_size, -1, -1)

    C_optimized = torch.tensor(C_optimized).float()
    C_optimized = C_optimized.unsqueeze(0)
    C_optimized = C_optimized.unsqueeze(0).expand(batch_size, -1, -1)

    # 返回生成的矩阵A、B和C以及它们之间的余弦相似度
    return A_optimized, B_optimized, C_optimized



# 打印生成的矩阵A、B和C以及它们之间的余弦相似度
# A_optimized, B_optimized, C_optimized = optimize_matrices(0.3, 0.4, 0.5, 44, 128)
# print("Matrix A:", A_optimized)
# print("Matrix B:", B_optimized)
# print("Matrix C:", C_optimized)
#
# print(A_optimized.shape)

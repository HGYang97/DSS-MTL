import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import minimize
import numpy as np
import torch

def generate_vmatrix(target_cosine_similarity_AB,target_cosine_similarity_AC,target_cosine_similarity_BC,matrix_size):
    '''
    后期根据实际情况将函数的输入变成一个列表
    '''
# 目标余弦相似度
# target_cosine_similarity_AB = 0.3
# target_cosine_similarity_AC = 0.4
# target_cosine_similarity_BC = 0.5
#
# # 初始矩阵A和B，以及空的矩阵C
# n = 30
    A = np.random.rand(matrix_size)
    B = np.random.rand(matrix_size)
    C = np.zeros(matrix_size)

# 定义损失函数，即余弦相似度与目标相似度之间的差值的平方
    def loss(x):
        A, B, C = x[:matrix_size], x[matrix_size:2*matrix_size], x[2*matrix_size:]
        loss_AB = (cosine(A, B) - target_cosine_similarity_AB) ** 2
        loss_AC = (cosine(A, C) - target_cosine_similarity_AC) ** 2
        loss_BC = (cosine(B, C) - target_cosine_similarity_BC) ** 2
        return loss_AB + loss_AC + loss_BC

# 使用优化算法来最小化损失函数，以调整矩阵A、B和C的值
    result = minimize(loss, np.concatenate((A, B, C)), method='L-BFGS-B')

# 从优化结果中获取生成的矩阵A、B和C
    A_optimized = result.x[:matrix_size]
    B_optimized = result.x[matrix_size:2*matrix_size]
    C_optimized = result.x[2*matrix_size:]

# 打印生成的矩阵A、B和C以及它们之间的余弦相似度
    print("Matrix A:", A_optimized)
    print("Matrix B:", B_optimized)
    print("Matrix C:", C_optimized)
    print("Cosine Similarity AB:", cosine(A_optimized, B_optimized))
    print("Cosine Similarity AC:", cosine(A_optimized, C_optimized))
    print("Cosine Similarity BC:", cosine(B_optimized, C_optimized))

    return A_optimized, B_optimized, C_optimized

A,B,C = generate_vmatrix(0.2,0.4,0.8,4)
print(type(A))
# A = torch.tensor(A)
# print(type(A))
# print(A)
# A = A.view(5,5)
# print(A.shape)
# A = A.unsqueeze(0)
# print(A.shape)
# n = 6
A= A.unsqueeze(0).expand(n, -1, -1, -1)
print(A.shape)
print(A)


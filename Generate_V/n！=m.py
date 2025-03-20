import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import minimize
import torch

def generate_vmatrix(target_cosine_similarity,matrix_size,batch_size):
    # 目标余弦相似度

    A = np.random.rand(matrix_size)
    B = np.random.rand(matrix_size)

    # 定义损失函数，即余弦相似度与目标相似度之间的差值的平方
    def loss(x):
        A, B = x[:matrix_size], x[matrix_size:]
        return (cosine(A, B) - target_cosine_similarity) ** 2

    # 使用优化算法来最小化损失函数，以接近目标余弦相似度
    result = minimize(loss, np.concatenate((A, B)), method='L-BFGS-B')

    # 从优化结果中获取生成的矩阵A和B
    A_optimized = result.x[:matrix_size]
    B_optimized = result.x[matrix_size:]

    A_optimized = torch.tensor(A_optimized)
    A_optimized = A_optimized.view(2, 2)
    A_optimized = A_optimized.unsqueeze(0)
    A_optimized = A_optimized.unsqueeze(0).expand(batch_size, -1, -1, -1)


    # 打印生成的矩阵A和B以及它们的余弦相似度
    print("Matrix B:", B_optimized)
    print("Cosine Similarity:", cosine(A_optimized, B_optimized))
    return A_optimized, B_optimized

A, _ = generate_vmatrix(0.8 ,4 ,16)
print(type(A))
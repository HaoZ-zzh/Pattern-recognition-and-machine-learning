import numpy as np

# 定义样本数据
class1 = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
class2 = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])

# 计算每个类别的均值向量
mean1 = class1.mean(axis=0)
mean2 = class2.mean(axis=0)

print(f"类别1的均值向量为: {mean1}")
print(f"类别2的均值向量为: {mean2}")

# print(class1.shape)


# 计算类内散度矩阵
cov1 = np.cov(class1.T) * (class1.shape[0] - 1)
cov2 = np.cov(class2.T) * (class2.shape[0] - 1)
print(f"类别1的协方差矩阵为: \n{cov1}")
print(f"类别2的协方差矩阵为: \n{cov2}")
sw = cov1 + cov2
print(f"类内散度矩阵为: \n{sw}")

# 计算Sw的逆矩阵
sw_inv = np.linalg.pinv(sw)
print(f"类内散度矩阵的逆矩阵为: \n{sw_inv}")

# # 计算类间散度矩阵
# mean_diff = (mean1 - mean2).reshape(2, 1)
# print(f"类别均值向量的差为: \n{mean_diff}")
# sb = mean_diff * mean_diff.T
# print(f"类间散度矩阵为: \n{sb}")

# 计算Fisher判别投影向量
w = np.dot(sw_inv, mean1 - mean2)
print(f"Fisher判别投影向量为: {w}")

# 求判别面方程：x^Tw = z0 = 0.5 * w^T(m1 + m2)
print(w)
print(cov1)
z0 = 0.5 * np.dot(w.T, (cov1 + cov2).T)
print(f"判别面方程为: x^T{w} = {z0}")

# 输出判别函数字符串f(x)=……，系数保留两位小数
print(f"f(x) = {w[0]:.2f}x1 + {w[1]:.2f}x2")
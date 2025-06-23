import numpy as np
import matplotlib.pyplot as plt

# 生成数据集
# np.random.seed(1) 用来保证每次生成的随机数相同与否
X = np.random.rand(100, 1)  # 100个样本，(100,1)
y = np.sin(X) + np.random.randn(100, 1) * 0.1  # y = sin(x) + 噪声

# 初始化网络参数
input_size = 1  # 输入层feaure vector
hidden_size = 1000  # 隐藏层neurons
output_size = 1  # 输出层label

# 初始化权重
W1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层weight (1,1000)
b1 = np.zeros((input_size, hidden_size))  # 隐藏层bias (1,1000)
W2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层weight (1000,1)
b2 = np.zeros((input_size, output_size))  # 输出层bias (1,1)

# 参数
learning_rate = 0.01
n_iterations = 1000
m = len(X)

# 记录loss用于可视化
loss_history = []

# 激活函数 activation function
def relu(x):
    return np.maximum(0, x)

# 前向传播
def forward(X):
    z1 = X.dot(W1) + b1  # 隐藏层的线性组合  (100,1000)=(100,1)*(1,1000)+(1,1000)
    a1 = relu(z1)  # 隐藏层的激活 (100,1000)
    z2 = a1.dot(W2) + b2  # 输出层的线性组合 (100,1)=(100,1000)*(1000,1)+(1,1)
    return z2

# MSE(loss function)
def compute_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)

# vanilla SGD
for iteration in range(n_iterations):
    # 随机选择一个样本
    random_index = np.random.randint(m)
    xi = X[random_index:random_index+1] # (1,1)
    yi = y[random_index:random_index+1] # (1,1)
    
    # 前向传播
    z1 = xi.dot(W1) + b1  # (1,1000)=(1,1)*(1,1000)+(1,1000)
    a1 = relu(z1)  # (1,1000)
    z2 = a1.dot(W2) + b2  # (1,1)=(1,1000)*(1000,1)+(1,1)
    y_hat = z2  # 输出结果

    # 计算损失
    loss = compute_loss(y_hat, yi)
    
    # 记录loss
    loss_history.append(loss)

    # 反向传播：计算梯度
    # 输出层梯度
    dz2 = y_hat - yi  # (1,1)
    dW2 = a1.T.dot(dz2) / m  # (1000,1)=(1000,1)*(1,1)
    db2 = np.sum(dz2, axis=0, keepdims=True) / m  # (1,1)

    # 隐藏层梯度
    da1 = dz2.dot(W2.T)
    dz1 = da1 * (z1 > 0)  # ReLU的梯度
    dW1 = xi.T.dot(dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # 更新参数
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # 每100步输出loss
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 第一个子图：拟合结果
ax1.scatter(X, y, c='blue', alpha=0.6, label='Original Data', s=30)

# 生成更密集的测试点来绘制平滑的拟合曲线
X_test = np.linspace(0, 1, 200).reshape(-1, 1)
y_pred = forward(X_test)

# 绘制拟合曲线
ax1.plot(X_test, y_pred, 'r-', linewidth=2, label='Neural Network Fit')

# 绘制训练后的预测点
y_train_pred = forward(X)
ax1.scatter(X, y_train_pred, c='red', alpha=0.8, s=20, label='Training Predictions')

ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax1.set_title("Vanilla SGD Neural Network Fitting Results")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 第二个子图：Loss下降曲线
ax2.plot(loss_history, 'b-', linewidth=1, alpha=0.7)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss")
ax2.set_title("Training Loss Over Time")
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # 使用对数坐标更好地显示loss变化

plt.tight_layout()
plt.show()

# 输出最终的loss
final_loss = compute_loss(forward(X), y)
print(f"Final Loss: {final_loss:.6f}")

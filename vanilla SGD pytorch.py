import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子确保结果可重现
# torch.manual_seed(42)
# np.random.seed(42)

# 生成数据集
X = torch.rand(100, 1)  # 100个样本
y = torch.sin(X) + torch.randn(100, 1) * 0.1  # y = sin(x) + 噪声

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=1000, output_size=1): # 输入层，隐藏层，输出层
        super(SimpleNN, self).__init__() 
        self.fc1 = nn.Linear(input_size, hidden_size) # 输入层到隐藏层
        self.relu = nn.ReLU() # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size) # 隐藏层到输出层
        
    def forward(self, x):
        x = self.fc1(x) # 输入层到隐藏层
        x = self.relu(x) # 激活函数
        x = self.fc2(x) # 隐藏层到输出层
        return x

# 初始化模型
model = SimpleNN() 
criterion = nn.MSELoss()

# 参数
learning_rate = 0.01
n_iterations = 1000
m = len(X)

# 记录loss用于可视化
loss_history = []

# vanilla SGD训练
for iteration in range(n_iterations):
    # 随机选择一个样本
    random_index = torch.randint(0, m, (1,)).item()
    xi = X[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    
    # 前向传播
    y_hat = model(xi)
    
    # 计算损失
    loss = criterion(y_hat, yi)
    
    # 记录loss
    loss_history.append(loss.item())
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 手动更新参数（vanilla SGD）
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    
    # 每100步输出loss
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item():.6f}")

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 第一个子图：拟合结果
ax1.scatter(X.numpy(), y.numpy(), c='blue', alpha=0.6, label='Original Data', s=30)

# 生成更密集的测试点来绘制平滑的拟合曲线
X_test = torch.linspace(0, 1, 200).reshape(-1, 1)
with torch.no_grad():
    y_pred = model(X_test)

# 绘制拟合曲线
ax1.plot(X_test.numpy(), y_pred.numpy(), 'r-', linewidth=2, label='Neural Network Fit')

# 绘制训练后的预测点
with torch.no_grad():
    y_train_pred = model(X)
ax1.scatter(X.numpy(), y_train_pred.numpy(), c='red', alpha=0.8, s=20, label='Training Predictions')

ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax1.set_title("Vanilla SGD Neural Network Fitting Results (PyTorch)")
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
with torch.no_grad():
    final_loss = criterion(model(X), y)
print(f"Final Loss: {final_loss.item():.6f}") 
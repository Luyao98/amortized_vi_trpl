import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn

# 定义CVXPY优化问题
batch_size = 10
n_components = 5  # 假设组件数为5
p = cp.Variable(n_components, nonneg=True)
q = cp.Parameter(n_components)
r = cp.Parameter(n_components)
epsilon = cp.Parameter(nonneg=True)

# 目标函数：最小化KL散度
objective = cp.Minimize(cp.sum(cp.kl_div(p, q)))

# 约束条件
constraints = [
    cp.sum(cp.kl_div(p, r)) <= epsilon,  # KL散度约束
    cp.sum(p) == 1  # 归一化约束
]

problem = cp.Problem(objective, constraints)

# 确保问题是DPP的
assert problem.is_dpp()

# 创建CVXPY层
cvxpylayer = CvxpyLayer(problem, parameters=[q, r, epsilon], variables=[p])

# 定义PyTorch模型
class MyModel(nn.Module):
    def __init__(self, n_components, epsilon):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(n_components, n_components)
        self.cvxpylayer = cvxpylayer
        self.epsilon = epsilon
        self.n_components = n_components
        self.r_prev = None  # 初始化上一循环的输出

    def forward(self, x):
        q = self.linear(x)  # 线性层的输出作为q
        if self.r_prev is None:
            self.r_prev = torch.ones_like(q) / self.n_components  # 初始r为均匀分布
        epsilon_tch = torch.tensor(self.epsilon, dtype=torch.float32)
        # 使用CVXPY层求解得到p
        try:
            p, = self.cvxpylayer(q, self.r_prev, epsilon_tch)
        except cp.error.SolverError as e:
            print(f"Solver failed: {e}")
            # 尝试增加求解器迭代次数或其他设置
            p = torch.ones_like(q) / self.n_components  # 回退到均匀分布
        self.r_prev = q.detach()  # 更新上一循环的输出
        return p

# 初始化模型和优化器
n_components = 5  # 假设组件数为5
epsilon = 0.1  # 假设epsilon为0.1
model = MyModel(n_components, epsilon)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 示例输入数据

input_tensor = torch.randn(batch_size, n_components, requires_grad=True)

# 模拟训练循环
for i in range(100):
    optimizer.zero_grad()  # 清除梯度

    # 前向传播：计算模型输出
    p = model(input_tensor)

    # 定义损失函数（例如解的和）
    loss = p.sum()

    # 反向传播：计算损失相对于模型参数的梯度
    loss.backward()

    # 更新模型参数
    optimizer.step()

    print(f"Iteration {i}, Loss: {loss.item()}")

# 打印最终的参数
print("Optimized model parameters:", list(model.parameters()))

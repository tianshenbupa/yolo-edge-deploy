import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 0. 核心：指定使用你的 RTX 5060 Ti 显卡
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前正在使用的设备: {device} ({torch.cuda.get_device_name(0)})")

# ==========================================
# 1. 准备数据
# ==========================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 2. 构建神经网络
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型，并【推送到 GPU】
model = SimpleNet().to(device)

# ==========================================
# 3. 定义损失函数和优化器
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==========================================
# 4. 开始训练
# ==========================================
print("\n开始在 GPU 上训练模型...")
model.train()

for batch_idx, (data, target) in enumerate(train_loader):
    # 【核心：将数据也推送到 GPU】，否则模型和数据不在一个地方会报错
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 200 == 0:
        print(f"训练进度: [{batch_idx * len(data)}/{len(train_loader.dataset)}] \t损失 (Loss): {loss.item():.4f}")

# ==========================================
# 5. 测试模型
# ==========================================
model.eval()
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        # 【测试数据同样推送到 GPU】
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"\n🎉 训练完成！测试集上的准确率 (Accuracy): {accuracy:.2f}%")

# ==========================================
# 6. 保存模型（为后续边缘部署做准备）
# ==========================================
# 转换为 TorchScript 格式并保存
scripted_model = torch.jit.script(model.to("cpu")) # 部署时通常转回 CPU 格式以便移植
scripted_model.save("mnist_edge_model.pt")
print("💾 边缘部署模型 'mnist_edge_model.pt' 已成功保存到当前目录下！")
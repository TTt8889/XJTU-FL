import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 假设你已经定义好了模型，这里以一个简单的卷积神经网络为例
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道为1，输出通道为32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道为32，输出通道为64
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，步长为2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层，10个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 卷积层 + 激活函数 + 池化层
        x = self.pool(torch.relu(self.conv2(x)))  # 卷积层 + 激活函数 + 池化层
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = torch.relu(self.fc1(x))  # 全连接层 + 激活函数
        x = self.fc2(x)  # 输出层
        return x

# 实例化模型
model = SimpleCNN()

# 实例化模型并移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is :", device)
model = SimpleCNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载CIFAR10数据集
data_dir = "/home/ymh/wxy/FLPoison/data"
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 训练模型
num_epochs = 50
train_losses = []
test_losses = []
test_error_rates = []

stop_epoch = num_epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_losses.append(running_loss / len(test_loader))
    test_error_rate = 1 - correct / total
    test_error_rates.append(test_error_rate)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Error Rate: {test_error_rate:.4f}')

    if test_error_rate < 0.001:
        stop_epoch = epoch
        break

# 保存模型的状态字典
torch.save(model.state_dict(), 'mnist_model_state_dict.pth')

# 绘制错误率图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, stop_epoch + 1), test_error_rates, label='Test Error Rate', color='blue', marker='o')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Error Rate', fontsize=14)
plt.title('Test Error Rate over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('test_error_rate.png')  # 保存图像
plt.close()
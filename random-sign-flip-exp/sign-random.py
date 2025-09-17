import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import datetime
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
model = SimpleCNN().to(device)

# 加载模型的状态字典
model.load_state_dict(torch.load('mnist_model_state_dict.pth', map_location=device))
model.eval()  # 设置为评估模式

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载CIFAR10数据集
data_dir = "/home/ymh/wxy/FLPoison/data"
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 多次测试
# num_tests = 500
# test_error_rates = []


# for test in range(num_tests):

#     if test > 10:
#         # 保存原始参数
#         original_params = {name: param.clone() for name, param in model.named_parameters()}
        
#         # 将参数乘以随机的正负1
#         for name, param in model.named_parameters():
#             param.data = param.data * torch.randint_like(param.data, low=0, high=2, device=device) * 2 - 1
    
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     test_error_rate = 1 - correct / total
#     test_error_rates.append(test_error_rate)

#     if test > 10:
#         # 恢复原始参数
#         for name, param in model.named_parameters():
#             param.data = original_params[name]
    
    # print(f'Test {test + 1}/{num_tests}, Test Error Rate: {test_error_rate:.4f}')

# # 测试符号反转比例从 0 到 1，步长为 0.1
# flip_ratios = [i * 0.1 for i in range(11)]
# num_tests = 100  # 每个比例测试的次数
# all_test_error_rates = []

# for flip_ratio in flip_ratios:
#     test_error_rates = []
    
#     for _ in range(num_tests):
#         # 保存原始参数
#         original_params = {name: param.clone() for name, param in model.named_parameters()}
        
#         # 将参数乘以随机的正负1，比例为 flip_ratio
#         for name, param in model.named_parameters():
#             mask = torch.rand_like(param).uniform_(0, 1) > flip_ratio
#             param.data = param.data * (2 * mask.float() - 1)
        
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         test_error_rate = 1 - correct / total
#         test_error_rates.append(test_error_rate)
        
#         # 恢复原始参数
#         for name, param in model.named_parameters():
#             param.data = original_params[name]
    
#     all_test_error_rates.append(test_error_rates)
#     print(f'Flip Ratio: {flip_ratio:.1f}, Average Test Error Rate: {np.mean(test_error_rates):.4f}, Std Dev: {np.std(test_error_rates):.4f}')

# # 绘制错误率图像
# plt.figure(figsize=(10, 5))
# palette = plt.get_cmap('Set1')

# # 计算均值和标准差
# means = [np.mean(test_error_rates) for test_error_rates in all_test_error_rates]
# stds = [np.std(test_error_rates) for test_error_rates in all_test_error_rates]

# # 绘制均值和标准差
# plt.errorbar(flip_ratios, means, yerr=stds, fmt='o', color='blue', ecolor='black', capsize=5, label='Test Error Rate')

# plt.xlabel('Flip Ratio', fontsize=14)
# plt.ylabel('Error Rate', fontsize=14)
# plt.title('Test Error Rate with Different Flip Ratios', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig('test_error_rate_with_random_flip_ratios_mnist.png')  # 保存图像
# plt.close()


# 测试符号反转比例从 0 到 1，步长为 0.1
flip_ratios = [i * 0.1 for i in range(11)]
num_tests = 50  # 每个比例测试的次数
all_test_acc_list = []

for flip_ratio in flip_ratios:
    test_acc_list = []
    
    for _ in range(num_tests):
        # 保存原始参数
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 将参数乘以随机的正负1，比例为 flip_ratio
        for name, param in model.named_parameters():
            mask = torch.rand_like(param).uniform_(0, 1) < flip_ratio
            param.data = -param.data * (2 * mask.float() - 1)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        test_acc_list.append(test_acc)
        
        # 恢复原始参数
        for name, param in model.named_parameters():
            param.data = original_params[name]
    
    all_test_acc_list.append(test_acc_list)
    # print(f'Flip Ratio: {flip_ratio:.1f}, Average Test Error Rate: {np.mean(test_error_rates):.4f}, Std Dev: {np.std(test_error_rates):.4f}')

array_data = np.array(all_test_acc_list)     
formatted_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")     
np.save(f"/home/ymh/wxy/Sign-Fed/random-sign-flip-exp/random-rate-sign-flip-acc_{formatted_time}.npy", array_data)


# # 绘制错误率图像
# plt.figure(figsize=(10, 5))
# palette = plt.get_cmap('Set1')

# # 计算均值和标准差
# means = [np.mean(test_error_rates) for test_error_rates in all_test_error_rates]
# stds = [np.std(test_error_rates) for test_error_rates in all_test_error_rates]

# # 绘制均值和标准差
# plt.errorbar(flip_ratios, means, yerr=stds, fmt='-o', color='blue', ecolor='black', capsize=5, label='Test Error Rate')

# # 自定义 y 轴标签格式为百分比形式
# def to_percent(y, position):
#     # 将 y 转换为百分比形式
#     s = f"{100 * y:.0f}%"
#     return s

# # 设置 y 轴的格式化器
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# # 设置 y 轴标签
# plt.ylabel('Test Error Rate',  fontsize=14)
# plt.xlabel('Flip Ratio', fontsize=14)
# plt.title('Test Error Rate with Different Flip Ratios', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig('test_error_rate_with_fix_flip_ratios_mnist.png')  # 保存图像
# plt.close()
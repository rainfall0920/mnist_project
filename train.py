import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

batch_size = 32
lr = 0.001
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备:{device}")

#???数据预处理规则：把图片变成模型能吃的格式
transform = transforms.Compose([
    # 1. 转成张量：把0-255的像素值，转成0-1之间的PyTorch张量
    #    形状从 [28, 28] 变成 [1, 28, 28]，1代表灰度图单通道
    transforms.ToTensor(),
    # 2. 标准化：让数据分布更规整，模型学起来更快，收敛更稳
    #    0.1307和0.3081是MNIST数据集官方算好的均值和标准差，固定用就行
    transforms.Normalize((0.1307,), (0.3081,))
])

#???加载训练集和测试集
train_dataset = datasets.MNIST(
    root = './data',
    train = True,   #?
    transform = transform,
    download = True
)
test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    transform = transform,
    download = True
)
#========== 用DataLoader批量加载 ==========
train_loader =  DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader =  DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# ========== 定义卷积神经网络 ==========
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一部分：卷积+池化，负责提取图片里的特征（线条、圆圈、拐角等）
        self.conv_layers = nn.Sequential(
            # 第1层卷积：输入1通道，输出16通道，卷积核3×3，边缘填充1圈
            # 填充1圈的作用：卷积后图片尺寸不变，还是28×28
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第2层卷积：输入16通道，输出32通道
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 第二部分：全连接层，负责根据提取的特征做最终分类
        self.fc_layers = nn.Sequential(
            nn.Linear(32*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    # 前向传播：定义数据在模型里的流动路线
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layers(x)
        return x
    # 创建模型对象，并移到对应的设备（CPU/GPU）
model = MNIST_CNN().to(device)

# ========== 损失函数：评判模型预测错了多少 ==========
# 多分类任务，固定用交叉熵损失，内部自带Softmax，模型最后一层不用加激活
criterion = nn.CrossEntropyLoss()
# ========== 优化器：负责更新模型参数 ==========
# Adam是最常用的优化器，稳定好调，传入模型所有可训练参数和学习率
optimizer = optim.Adam(model.parameters(),lr=lr)

print("\n===== 开始训练 =====")
# 外层循环：控制训练多少轮
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 内层循环：逐个批次训练
    for inputs,labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 标准训练5步，永远不变
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # 统计本轮的损失和准确率
        total_loss += loss.item()
        _,predicted = torch.max(outputs.data,dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 每轮结束，打印本轮的平均损失和准确率
    avg_loss = total_loss/len(train_loader)
    train_acc = 100*correct/total
    print(f"Epoch{epoch+1}/{epochs} | 平均损失：{avg_loss:.4f} | 训练准确率：{train_acc:.2f}%")

# 只保存模型的权重参数，不保存模型结构，文件小、通用性强
torch.save(model.state_dict(),"./models/mnist_cnn.pth")
print("\n模型已保存到 models/mnist_cnn.pth")

print("\n===== 开始测试 =====")
# 1. 先新建一个同结构的模型
test_model = MNIST_CNN().to(device)
# 2. 加载刚才保存的权重参数
test_model.load_state_dict(torch.load("./models/mnist_cnn.pth",weights_only=True))
# 3. 切换到评估模式
test_model.eval()

test_correct = 0
test_total = 0

# 测试时不计算梯度，节省资源、加快速度
with torch.no_grad():
    for inputs,labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = test_model(inputs)
        _,predicted = torch.max(outputs.data,1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# 计算测试集准确率
test_acc = 100 * test_correct / test_total
print(f"测试集准确率：{test_acc:.2f}%")
print("\n🎉 项目全部完成！")
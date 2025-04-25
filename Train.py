import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

# 数据目录
data_dir = '/mnt/dataset/train'

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f"[init] == device: {device} ==")


# 数据预处理：调整图像大小，转换为灰度图，转为tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.ToTensor(),  # 转为Tensor，并归一化到[0, 1]
])

# 使用ImageFolder读取数据，注意要将训练集的子文件夹作为分类标签
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 按照7:3划分训练集和测试集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建训练和测试集的DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 输出数据集信息
dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
print(f"训练集样本数: {dataset_sizes['train']}")
print(f"测试集样本数: {dataset_sizes['test']}")

# 获取类别名称
class_names = dataset.classes
for index, class_name in enumerate(class_names):
    print(f"Label: {index}, Class Name: {class_name}")



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from Ours import Get_BG_ResNet18, Get_BG_ResNet34
from Plot import evaluate_and_plot

model = Get_BG_ResNet18().to(device)
# model = Get_BG_ResNet34().to(device)


# # 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
Max = 0
for i in range(20):
    model.train()
    p = 0
    sum_loss = torch.zeros((30))
    for inputs, labels in train_loader:
        labels = one_hot_encode(labels, 4)
        inputs, labels = inputs.to(device), labels.to(device)
        output, _, _ = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(torch.mean(sum_loss))
    if i>=10:
        model.eval()
        all_test_target = []
        all_test_output = []
        all_score = []
        m = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            all_test_target.append(labels)
            output, _, _ = model(inputs)
            all_score.append(output)
            predicted_class = torch.argmax(output, dim=1).to(device)
            all_test_output.append(predicted_class)
            m = m + 1

        all_test_target = torch.cat(all_test_target)
        all_test_output = torch.cat(all_test_output)
        all_score = torch.cat(all_score)
        acu = torch.sum(all_test_output == all_test_target).item() / len(all_test_output)
        acu_percent = acu * 100
        all_test_target = all_test_target.cpu().numpy()
        all_test_output = all_test_output.cpu().numpy()
        print(f'Accuracy: {acu_percent:.2f}%')
        if acu > 0.955:
            evaluate_and_plot(model, test_loader, class_names, device)
            torch.save(model.state_dict(), f"{acu_percent:.2f}%_model.pth")
            print(f"Model saved as {acu_percent:.2f}%_model.pth")

print("跑完了")


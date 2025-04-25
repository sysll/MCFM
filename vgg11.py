import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# 数据目录
data_dir = '/mnt/dataset/train'

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f"[init] == device: {device} ==")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 变成3通道灰度图（复制3份）
    transforms.ToTensor(),
])

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return acc, prec, recall, f1


# 读取数据
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 划分训练/测试集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 输出数据集信息
print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 类别名称
class_names = dataset.classes
for index, class_name in enumerate(class_names):
    print(f"Label: {index}, Class Name: {class_name}")

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载 VGG11 模型并修改输出层
model = models.vgg11(weights=None)
model.classifier[6] = nn.Linear(4096, 4)  # 4 类输出
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(20):
    model.train()
    sum_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {sum_loss / len(train_loader):.4f}")

    # 从第10轮开始评估
    if epoch >= 10:
        print(f"[Epoch {epoch + 1}] Evaluation:")
        evaluate_model(model, test_loader, device)

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

# 图像预处理（Swin Transformer 输入为 224x224）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# 模型评估函数
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

# 加载数据
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 类别信息
print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
class_names = dataset.classes
for index, class_name in enumerate(class_names):
    print(f"Label: {index}, Class Name: {class_name}")

# 加载 Swin Transformer 模型
model = models.swin_t(weights=None)  # 使用预训练则改为 weights=models.Swin_T_Weights.DEFAULT
model.head = nn.Linear(model.head.in_features, 4)  # 修改输出层
model = model.to(device)

# 损失函数 & 优化器
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

    if epoch >= 10:
        print(f"[Epoch {epoch + 1}] Evaluation:")
        evaluate_model(model, test_loader, device)

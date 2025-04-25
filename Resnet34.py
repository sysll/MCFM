import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# 数据目录
data_dir = '/mnt/dataset/train'
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f"[init] == device: {device} ==")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

#绘制曲线
def plot_evaluation_curves(true_labels, pred_probs, pred_classes, class_names):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy()
    if torch.is_tensor(pred_probs):
        pred_probs = pred_probs.cpu().numpy()
    if torch.is_tensor(pred_classes):
        pred_classes = pred_classes.cpu().numpy()

    # 1️⃣ 混淆矩阵（数量 + 识别率）
    cm = confusion_matrix(true_labels, pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 每行归一化为识别率

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')


    ax.set_title("Confusion Matrix", fontsize=24)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(len(class_names))], fontsize=20)
    ax.set_yticklabels([str(i) for i in range(len(class_names))], fontsize=20)
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)
    thresh = cm_normalized.max() / 2.

    # 每个格子显示为两行：数量 + 百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            ax.text(j, i, f'{count}\n{percent:.1f}%',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=18)

    plt.tight_layout()
    plt.show()

    # 2️⃣ ROC 和 PR 曲线
    plt.figure(figsize=(14, 6))
    num_classes = pred_probs.shape[1]
    true_onehot = np.eye(num_classes)[true_labels]

    # ROC 曲线
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_onehot[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve', fontsize=24)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=16)  # ✅ 加上这行
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)

    # PR 曲线
    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(true_onehot[:, i], pred_probs[:, i])
        ap = average_precision_score(true_onehot[:, i], pred_probs[:, i])  # 计算 AP 值
        plt.plot(recall, precision, label=f'Class {i} (AP = {ap:.4f})', linewidth=2)  # 在标签中显示 AP
    plt.title('Precision-Recall Curve', fontsize=24)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)

    plt.tight_layout()
    plt.show()


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
    if 0.91 <= acc < 0.93:
        print(f"Accuracy 达到 {acc * 100:.2f}%，开始绘制混淆矩阵、ROC 和 PR 曲线...")

        # 将 logits 重算出来用于概率图（假设你用 softmax 输出）
        all_probs = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        # 调用绘图函数
        plot_evaluation_curves(
            true_labels=np.array(all_labels),
            pred_probs=np.array(all_probs),
            pred_classes=np.array(all_preds),
            class_names=class_names
        )
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

# 加载 ResNet-34 模型
model = models.resnet34(weights=None)  # 可换成 weights=models.ResNet34_Weights.DEFAULT
model.fc = nn.Linear(model.fc.in_features, 4)  # 修改输出层为4类
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


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import logging
from configs import *
from utils import *
import time

device = f'cuda:{device_index}'
expname = 'experiment'

# 创建日志目录
cur_time = get_timestamp()
log_folder = f'logs/{cur_time}_{expname}/'
os.makedirs(log_folder, exist_ok=True)

# 日志设置
logger = logging.getLogger(__name__)
log_file = os.path.join(log_folder, "log.txt")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# 模型与数据初始化
wrapClip = WrapClip(device)
male_emb = wrapClip.get_emb("male").to(device)[:, 0:1, :]
female_emb = wrapClip.get_emb("female").to(device)[:, 0:1, :]

dataset = GenderDataset(dataset_file, wrapClip, configs.clip_cache, device)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

test_dataset = GenderDataset("/home/DC/zenghang/vedio_safety/CIC/LatentGuard/test.json", wrapClip, configs.clip_cache, device)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 构建模型
base_layer = EmbeddingMappingLayer(num_heads, head_dim, out_dim)
model = GenderPromptClassifier(base_layer, num_heads, head_dim, out_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === 新增 === 保存路径
checkpoint_dir = os.path.join(log_folder, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
final_model_path = os.path.join(checkpoint_dir, "final_model.pth")

num_epochs = 100
loss_history = []
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for prompt_emb, labels, _ in train_dataloader:
        prompt_emb, labels = prompt_emb.to(device), labels.to(device)
        B = prompt_emb.size(0)
        male_emb_batch = male_emb.repeat(B, 1, 1)
        female_emb_batch = female_emb.repeat(B, 1, 1)

        logits = model(prompt_emb, male_emb_batch, female_emb_batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    loss_history.append(avg_loss)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # === 新增：保存检查点 ===
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, checkpoint_path)

    # === 新增：保存最优模型 ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Best model updated at epoch {epoch+1} (loss={best_loss:.4f})")

# === 新增：保存最终模型 ===
torch.save(model.state_dict(), final_model_path)
logger.info(f"Final model saved at: {final_model_path}")

# 绘制 loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='blue')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_folder, "loss_curve.png"), dpi=300)
plt.show()

# =======================
#       测试部分
# =======================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for prompt_emb, labels, _ in test_dataloader:
        prompt_emb, labels = prompt_emb.to(device), labels.to(device)
        B = prompt_emb.size(0)
        male_emb_batch = male_emb.repeat(B, 1, 1)
        female_emb_batch = female_emb.repeat(B, 1, 1)

        logits = model(prompt_emb, male_emb_batch, female_emb_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print(f"\nTest Results:\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

logger.info(f"Test Accuracy: {acc:.4f}")
logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
logger.info(f"Confusion Matrix:\n{cm}")

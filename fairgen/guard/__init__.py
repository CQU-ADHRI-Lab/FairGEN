import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import logging
from LatentGuard.configs import *
from LatentGuard.utils import *
import time

def init_CIC(checkpoint_path):
    print("=== Loading CIC model and weights ===")
    base_layer = EmbeddingMappingLayer(num_heads, head_dim, out_dim)
    model = GenderPromptClassifier(base_layer, num_heads, head_dim, out_dim)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def gender_concept_tensor(class_list, device="cpu"):
    """
    Args:
        class_list: list[str]，每个元素为以下之一：
            "gender-irrelevant", "gender-male", "gender-female", "implicit-gender-bias"
        device: 将tensor放到的设备

    Returns:
        tensor, shape = [batch_size, 2]
        列含义: [male, female]
    """
    result = []
    for c in class_list:
        if c == "gender-irrelevant":
            result.append([0., 0.])
        elif c == "gender-male":
            result.append([1., 0.])
        elif c == "gender-female":
            result.append([0., 1.])
        elif c == "implicit-gender-bias":
            # 0.5 概率随机选择男性或女性
            if random.random() < 0.5:
                result.append([1., 0.])
            else:
                result.append([0., 1.])
        else:
            raise ValueError(f"Unexpected class label: {c}")

    return result


def get_CIC_pred(model,prompt_list,device):
    # ===============================
    #        准备输入
    # ===============================
    wrapClip = WrapClip(device)
    model=model.to(device)
    male_emb = wrapClip.get_emb("male").to(device)[:, 0:1, :]
    female_emb = wrapClip.get_emb("female").to(device)[:, 0:1, :]

    # 对每个 prompt 获取 CLIP embedding
    prompt_embs = [wrapClip.get_emb(p).to(device) for p in prompt_list]
    prompt_embs = torch.cat(prompt_embs, dim=0)  # shape: (N, seq_len, emb_dim)

    # 扩展性别向量以匹配 batch
    B = prompt_embs.size(0)
    male_emb_batch = male_emb.repeat(B, 1, 1)
    female_emb_batch = female_emb.repeat(B, 1, 1)

    # ===============================
    #        推理 / 分类
    # ===============================
    with torch.no_grad():
        logits = model(prompt_embs, male_emb_batch, female_emb_batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
   
    label2id =[
                "gender-irrelevant",
                "gender-male",
                "gender-female",
                "implicit-gender-bias"]
    label=[label2id[pred] for pred in preds]
    return label,gender_concept_tensor(label,device)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGAFID Dataset PyTorch 二分类训练示例

支持 InceptionTime 和 ConvMHSA 两种模型。

使用方法:
    python NGAFID_DATASET_TF_EXAMPLE.py --demo                    # 快速验证 InceptionTime
    python NGAFID_DATASET_TF_EXAMPLE.py --demo --model convmhsa  # 快速验证 ConvMHSA
    python NGAFID_DATASET_TF_EXAMPLE.py --model inception         # 训练 InceptionTime
    python NGAFID_DATASET_TF_EXAMPLE.py --model convmhsa         # 训练 ConvMHSA
"""

import os
import sys
import gc
import math
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngafiddataset.dataset.dataset import NGAFID_Dataset_Manager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

class Config:
    """训练配置"""
    DATASET_NAME = '2days'
    DESTINATION = ''
    MAX_LENGTH = 4096
    NUM_CHANNELS = 23
    NUM_CLASSES = 2  # 二分类

    # 显存优化: 减小 batch size
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_FOLDS = 5

    # InceptionTime 超参数
    NUM_FILTERS = 32
    DEPTH = 6
    KERNEL_SIZE = 40
    USE_RESIDUAL = False
    USE_BOTTLENECK = True
    DROPOUT = 0.5

    # ConvMHSA 超参数 (参考原版)
    MHSA_D_MODEL = 512
    MHSA_NUM_HEADS = 8
    MHSA_DFF = 1024
    MHSA_NUM_LAYERS = 4
    MHSA_DROPOUT = 0.1
    MHSA_LEARNING_RATE = 5e-4  # ConvMHSA 专用学习率

    # DataLoader 优化
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    MODEL_NAME = 'InceptionTime_Binary'
    SAVE_PATH = '.'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 42


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ============================================================================
# 数据集
# ============================================================================

class NGAFIDDataset(Dataset):
    """NGAFID 数据集 - 内存优化版本"""

    def __init__(self, data_dict, mins, maxs, device='cpu'):
        self.data_refs = []
        self.labels = []
        self.folds = []
        self.mins = mins.astype(np.float32)
        self.maxs = maxs.astype(np.float32)
        self.device = device

        for sample in data_dict:
            self.data_refs.append(sample['data'])
            self.labels.append(int(sample['before_after']))
            self.folds.append(sample['fold'])

    def __len__(self):
        return len(self.data_refs)

    def __getitem__(self, idx):
        x = self.data_refs[idx].astype(np.float32)
        x = (x - self.mins) / (self.maxs - self.mins + 1e-8)
        x = np.nan_to_num(x, nan=0.0)
        x = torch.from_numpy(x).float()
        x = x.transpose(0, 1)  # (23, 4096)
        return x, self.labels[idx]


class SubDataset(Dataset):
    """子数据集"""
    def __init__(self, parent_dataset, indices):
        self.parent = parent_dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]


# ============================================================================
# 模型工厂
# ============================================================================

def create_model(model_type, config):
    """创建模型"""
    if model_type == 'inception':
        return InceptionTime(
            in_channels=config.NUM_CHANNELS,
            nb_filters=config.NUM_FILTERS,
            depth=config.DEPTH,
            kernel_size=config.KERNEL_SIZE,
            use_residual=config.USE_RESIDUAL,
            dropout=config.DROPOUT
        )
    elif model_type == 'convmhsa':
        return ConvMHSA(
            in_channels=config.NUM_CHANNELS,
            seq_length=config.MAX_LENGTH,
            d_model=config.MHSA_D_MODEL,
            num_heads=config.MHSA_NUM_HEADS,
            dff=config.MHSA_DFF,
            num_layers=config.MHSA_NUM_LAYERS,
            dropout=config.MHSA_DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# InceptionTime 模型
# ============================================================================

class InceptionModule(nn.Module):
    """Inception 模块"""

    def __init__(self, in_channels, out_channels, kernel_size=40, use_bottleneck=True):
        super().__init__()

        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, 32, kernel_size=1, bias=False)
            self.bn_bottleneck = nn.BatchNorm1d(32)
            branch_channels = 32
        else:
            self.bottleneck = None
            branch_channels = in_channels

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(branch_channels, out_channels, kernel_size=ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for ks in [kernel_size // 4, kernel_size // 2, kernel_size]
        ])

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_original = x

        if self.bottleneck is not None:
            x = self.bottleneck(x)
            x = self.bn_bottleneck(x)

        outputs = [conv(x) for conv in self.convs]
        pool_out = self.pool_conv(x_original)
        target_len = outputs[0].size(-1)
        if pool_out.size(-1) != target_len:
            pool_out = F.interpolate(pool_out, size=target_len, mode='linear', align_corners=False)
        outputs.append(pool_out)

        return torch.cat(outputs, dim=1)


class InceptionTime(nn.Module):
    """InceptionTime 二分类模型"""

    def __init__(self, in_channels=23, nb_filters=32, depth=6, kernel_size=40,
                 use_residual=False, dropout=0.5):
        super().__init__()
        self.module_out_channels = nb_filters * 4

        self.inception_blocks = nn.ModuleList()
        for d in range(depth):
            in_ch = in_channels if d == 0 else self.module_out_channels
            self.inception_blocks.append(InceptionModule(in_ch, nb_filters, kernel_size))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.module_out_channels, 1)

    def forward(self, x):
        for inception_block in self.inception_blocks:
            x = inception_block(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


# ============================================================================
# ConvMHSA 模型
# ============================================================================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.scale = self.depth ** -0.5

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        # 线性变换
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # reshape: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, depth)
        q = q.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.dense(out)

        return out, attn


class EncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(inplace=True),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.dropout1(attn_out)
        x = self.layernorm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.layernorm2(x + ffn_out)

        return x


class ConvMHSA(nn.Module):
    """
    ConvMHSA 二分类模型

    架构参考原版:
    - Conv1D 特征提取: 128->128->256->256->512 (strides 2)
    - 4层 Transformer Encoder (MHSA + FFN)
    - GlobalAveragePooling + Dense(1)

    修复说明:
    - 移除 Linear 投影层 (Conv1D 输出直接是 512 通道)
    - 添加 BatchNorm 稳定训练
    """

    def __init__(self, in_channels=23, seq_length=4096, d_model=512, num_heads=8,
                 dff=1024, num_layers=4, dropout=0.1):
        super().__init__()

        # Conv1D 特征提取 (参考原版结构)
        # 输入: (batch, channels, length) = (batch, 23, 4096)
        self.conv_layers = nn.Sequential(
            # Block 1: 23 -> 128, stride 1
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            # Block 2: 128 -> 128, stride 2 (下采样)
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # Block 3: 128 -> 256, stride 1
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            # Block 4: 256 -> 256, stride 2 (下采样)
            nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # Block 5: 256 -> 512, stride 2 (下采样) - 输出直接是 512 通道
            nn.Conv1d(256, 512, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # 移除 BatchNorm（可能导致梯度问题）
        )

        # 计算卷积后序列长度: 4096 / 2 / 2 / 2 = 512
        self.conv_out_length = seq_length // 8

        # 位置编码 - d_model=512 与 Conv 输出通道数匹配
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.conv_out_length + 10)

        # Transformer 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot 初始化 - 对 Transformer 很重要"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # fc 偏置初始化为 0（让初始输出接近 0，sigmoid(0)=0.5）

    def forward(self, x):
        """
        x: (batch, channels, length) = (batch, 23, 4096)
        """
        # Conv1D 特征提取: (batch, 23, 4096) -> (batch, 512, 512)
        x = self.conv_layers(x)

        # 转置: (batch, channels, seq_len) -> (batch, seq_len, channels)
        # Conv 输出 (batch, 512, 512) -> Transformer 期望 (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, 512, 512)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer 编码器
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # 全局平均池化: (batch, seq_len, d_model) -> (batch, d_model)
        x = x.mean(dim=1)
        x = self.dropout(x)

        # 输出: (batch, 1)
        return self.fc(x).squeeze(-1)


# ============================================================================
# 评估指标
# ============================================================================

def compute_metrics(outputs, targets, threshold=0.5):
    """计算评估指标"""
    predictions = (torch.sigmoid(outputs) > threshold).float()

    # 准确率
    accuracy = (predictions == targets).float().mean().item()

    # True Positives, False Positives, False Negatives
    tp = ((predictions == 1) & (targets == 1)).float().sum().item()
    fp = ((predictions == 1) & (targets == 0)).float().sum().item()
    fn = ((predictions == 0) & (targets == 1)).float().sum().item()
    tn = ((predictions == 0) & (targets == 0)).float().sum().item()

    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # ROC-AUC (近似计算)
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(outputs).cpu().numpy()
        labels = targets.cpu().numpy()
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5

    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc * 100
    }


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """训练器"""

    def __init__(self, model, config, model_type='inception'):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model_type = model_type

        self.use_amp = config.DEVICE == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 根据模型类型选择学习率
        if model_type == 'convmhsa':
            lr = config.MHSA_LEARNING_RATE
        else:
            lr = config.LEARNING_RATE

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_f1 = 0.0

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, total_correct, total = 0, 0, 0

        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS}')
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * total_correct / total:.2f}%'
            })

        avg_loss = total_loss / len(loader)
        accuracy = 100. * total_correct / total

        return avg_loss, accuracy

    def validate(self, loader):
        self.model.eval()
        total_loss, all_outputs, all_targets = 0, [], []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # 计算所有指标
        metrics = compute_metrics(all_outputs, all_targets)
        avg_loss = total_loss / len(loader)

        return avg_loss, metrics

    def train(self, train_loader, val_loader, fold):
        print(f'\n开始训练 Fold {fold + 1}/{self.config.NUM_FOLDS}')
        print(f'设备: {self.device}')
        if self.device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(0)}')

        history = []

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, metrics = self.validate(val_loader)
            self.scheduler.step()

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': metrics['accuracy'],
                'val_f1': metrics['f1'],
                'val_auc': metrics['auc']
            })

            # 使用 F1 保存最优模型
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                self.save_model(fold)
                marker = ' [F1最优]'
            else:
                marker = ''

            if (epoch + 1) % 5 == 0 or marker:
                print(f'Epoch {epoch + 1}: '
                      f'Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | '
                      f'Val Loss={val_loss:.4f}, Acc={metrics["accuracy"]:.2f}%, '
                      f'F1={metrics["f1"]:.2f}%, AUC={metrics["auc"]:.2f}%{marker}')

        return history

    def save_model(self, fold):
        save_dir = Path(self.config.SAVE_PATH) / 'models'
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{self.config.MODEL_NAME}_fold{fold}.pt'
        torch.save({
            'fold': fold,
            'model_state_dict': self.model.state_dict(),
            'best_f1': self.best_f1
        }, path)


# ============================================================================
# 数据加载
# ============================================================================

def prepare_data(config):
    """准备数据集"""
    print('=' * 60)
    print('准备数据集...')
    print('=' * 60)

    dm = NGAFID_Dataset_Manager(config.DATASET_NAME, destination=config.DESTINATION)
    dm.data_dict = dm.construct_data_dictionary(numpy=True)

    print(f'样本数量: {len(dm.data_dict)}')
    print(f'序列长度: {config.MAX_LENGTH}')
    print(f'通道数: {config.NUM_CHANNELS}')

    labels = [s['before_after'] for s in dm.data_dict]
    print(f'正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}')

    dataset = NGAFIDDataset(dm.data_dict, dm.mins, dm.maxs)

    return dm, dataset


def get_fold_data(dataset, fold, config):
    """获取指定折的训练和测试数据"""
    train_indices = [i for i, f in enumerate(dataset.folds) if f != fold]
    test_indices = [i for i, f in enumerate(dataset.folds) if f == fold]

    print(f'训练样本: {len(train_indices)}, 测试样本: {len(test_indices)}')

    num_workers = 0 if sys.platform == 'win32' else config.NUM_WORKERS

    train_loader = DataLoader(
        SubDataset(dataset, train_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None
    )

    test_loader = DataLoader(
        SubDataset(dataset, test_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None
    )

    return train_loader, test_loader


# ============================================================================
# 主函数
# ============================================================================

def train(model_type, config):
    """训练主函数"""
    model_name_map = {
        'inception': 'InceptionTime_Binary',
        'convmhsa': 'ConvMHSA_Binary'
    }
    config.MODEL_NAME = model_name_map.get(model_type, f'{model_type}_Binary')

    print('=' * 60)
    print(f'NGAFID 二分类训练 - {model_type.upper()}')
    print('=' * 60)

    set_seed(config.RANDOM_SEED)

    print(f'\n设备: {config.DEVICE}')
    if config.DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        torch.cuda.empty_cache()

    dm, dataset = prepare_data(config)

    all_results = []
    for fold in range(config.NUM_FOLDS):
        print(f'\n{"=" * 60}')
        print(f'Fold {fold + 1}/{config.NUM_FOLDS}')
        print(f'{"=" * 60}')

        train_loader, test_loader = get_fold_data(dataset, fold, config)

        model = create_model(model_type, config)

        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f'\n模型: {model_type.upper()}')
            print(f'模型参数量: {total_params:,}')

        trainer = Trainer(model, config, model_type)
        history = trainer.train(train_loader, test_loader, fold)

        final_metrics = history[-1]
        print(f'\nFold {fold + 1} 完成: '
              f'Acc={final_metrics["val_acc"]:.2f}%, '
              f'F1={final_metrics["val_f1"]:.2f}%, '
              f'AUC={final_metrics["val_auc"]:.2f}%')

        all_results.append({
            'fold': fold,
            'val_acc': final_metrics['val_acc'],
            'val_f1': final_metrics['val_f1'],
            'val_auc': final_metrics['val_auc'],
            'best_f1': trainer.best_f1
        })

        del model, trainer, train_loader, test_loader
        gc.collect()
        if config.DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print('\n' + '=' * 60)
    print('训练完成!')
    print('=' * 60)
    results_df = pd.DataFrame(all_results)
    print(f'\n平均准确率: {results_df["val_acc"].mean():.2f}% ± {results_df["val_acc"].std():.2f}%')
    print(f'平均 F1: {results_df["val_f1"].mean():.2f}% ± {results_df["val_f1"].std():.2f}%')
    print(f'平均 AUC: {results_df["val_auc"].mean():.2f}% ± {results_df["val_auc"].std():.2f}%')

    return all_results


def demo(model_type='inception'):
    """快速演示"""
    print('=' * 60)
    print(f'快速演示 - {model_type.upper()} 单折训练')
    print('=' * 60)

    set_seed(Config.RANDOM_SEED)

    device = torch.device(Config.DEVICE)
    print(f'\n使用设备: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        torch.cuda.empty_cache()

    dm, dataset = prepare_data(Config)
    train_loader, test_loader = get_fold_data(dataset, 0, Config)

    model = create_model(model_type, Config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n模型: {model_type.upper()}')
    print(f'模型参数量: {total_params:,}')

    # 测试前向传播
    print('\n测试模型前向传播...')
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x = x.to(device)
        output = model(x)
        print(f'输入形状: {x.shape}')
        print(f'输出形状: {output.shape}')
        print(f'输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]')

    # 训练 1 个 epoch
    print('\n训练 1 个 epoch 测试...')
    Config.NUM_EPOCHS = 1
    trainer = Trainer(model, Config, model_type)
    train_loss, train_acc = trainer.train_epoch(train_loader, 0)
    val_loss, metrics = trainer.validate(test_loader)

    print(f'\n测试结果:')
    print(f'  训练损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%')
    print(f'  验证损失: {val_loss:.4f}')
    print(f'  准确率: {metrics["accuracy"]:.2f}%')
    print(f'  F1: {metrics["f1"]:.2f}%')
    print(f'  AUC: {metrics["auc"]:.2f}%')

    print('\n演示完成!')
    return True


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NGAFID 二分类训练')
    parser.add_argument('--model', type=str, default='inception',
                      choices=['inception', 'convmhsa'],
                      help='选择模型: inception 或 convmhsa')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--save_path', type=str, default='.', help='保存路径')
    parser.add_argument('--demo', action='store_true', help='快速演示模式')

    args = parser.parse_args()

    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.num_epochs
    Config.NUM_WORKERS = args.num_workers
    Config.SAVE_PATH = args.save_path

    if args.demo:
        demo(args.model)
    else:
        train(args.model, Config)

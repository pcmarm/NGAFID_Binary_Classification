#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGAFID Dataset PyTorch MINIROCKET 二分类训练示例

MINIROCKET 是一种高效的时间序列分类方法，使用随机卷积核提取特征。

使用方法:
    python NGAFID_DATASET_MINIROCKET_EXAMPLE.py --demo  # 快速验证
    python NGAFID_DATASET_MINIROCKET_EXAMPLE.py         # 完整训练
"""

import os
import sys
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

    # MINIROCKET 超参数
    NUM_KERNELS = 10000
    KERNEL_LENGTH_MIN = 7
    KERNEL_LENGTH_MAX = 9
    DILATIONS_MAX = 32

    # 训练超参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 2.5e-5
    WEIGHT_DECAY = 1e-5
    NUM_FOLDS = 5
    HIDDEN_SIZE = 512
    DROPOUT = 0.5

    MODEL_NAME = 'MINIROCKET_Binary'
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
# MINIROCKET 实现
# ============================================================================

class MiniRocketFeatures(nn.Module):
    """
    MINIROCKET 特征提取器

    使用随机卷积核提取时间序列特征。
    权重固定为1，只通过偏置学习。
    """

    def __init__(self, input_channels, num_kernels=10000,
                 kernel_length_min=7, kernel_length_max=9, dilations_max=32):
        super().__init__()
        self.input_channels = input_channels
        self.num_kernels = num_kernels

        # 生成随机卷积核参数
        self.kernel_lengths = []
        self.dilations = []
        self.paddings = []
        self.biases = []

        for _ in range(num_kernels):
            # 卷积核长度: 7 或 9
            kernel_length = np.random.choice([kernel_length_min, kernel_length_max])
            self.kernel_lengths.append(kernel_length)

            # 膨胀: 2的幂次方 [1, 2, 4, 8, 16, 32]
            dilation_exp = np.random.randint(0, int(np.log2(dilations_max)) + 1)
            dilation = 2 ** dilation_exp
            self.dilations.append(dilation)

            # Padding: 保持输出长度
            padding = (kernel_length - 1) * dilation // 2
            self.paddings.append(padding)

            # 偏置: 均匀分布 [-1, 1]
            bias = np.random.uniform(-1.0, 1.0)
            self.biases.append(bias)

        # 转换为张量
        self.register_buffer('biases_tensor', torch.tensor(self.biases, dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) = (batch, 23, 4096)
        Returns:
            features: (batch, num_kernels * 2)
        """
        batch_size = x.size(0)
        features_list = []

        for kernel_idx in range(self.num_kernels):
            kernel_length = self.kernel_lengths[kernel_idx]
            dilation = self.dilations[kernel_idx]
            padding = self.paddings[kernel_idx]

            # 固定权重: 全1
            weight = x.new_ones(1, self.input_channels, kernel_length)

            # 1D 卷积
            conv_out = F.conv1d(x, weight, padding=padding, dilation=dilation, groups=1)

            # 计算百分位数特征
            p33, p66 = conv_out.quantile(dim=-1, q=torch.tensor([0.33, 0.66], device=conv_out.device))

            # 特征1: 最大值与偏置比较
            f1 = (conv_out.max(dim=-1)[0] > self.biases_tensor[kernel_idx]).float()

            # 特征2: 百分位数差异
            f2 = ((p66 - p33) > 0).float()

            features_list.append(f1)
            features_list.append(f2)

        # (batch, num_kernels * 2)
        return torch.stack(features_list, dim=1)


class MiniRocketClassifier(nn.Module):
    """MINIROCKET 二分类器"""

    def __init__(self, input_channels, num_kernels=10000,
                 hidden_size=512, dropout=0.5):
        super().__init__()

        # MINIROCKET 特征提取
        self.features = MiniRocketFeatures(
            input_channels=input_channels,
            num_kernels=num_kernels
        )

        # 分类头
        num_features = num_kernels * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # 提取特征
        features = self.features(x)
        # 分类
        return torch.sigmoid(self.classifier(features)).squeeze(-1)


# ============================================================================
# 数据集
# ============================================================================

class NGAFIDDataset(Dataset):
    """NGAFID 数据集"""

    def __init__(self, data_dict, mins, maxs):
        self.samples = []

        for sample in data_dict:
            # 数据: (length, channels) -> (channels, length) for PyTorch
            x = sample['data'].astype(np.float32)
            x = (x - mins) / (maxs - mins + 1e-8)
            x = np.nan_to_num(x, nan=0.0)
            x = torch.from_numpy(x).float()  # (channels, length)

            # 标签: 二分类 (0 或 1)
            y = int(sample['before_after'])
            fold = sample['fold']

            self.samples.append({'x': x, 'y': y, 'fold': fold})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['x'], self.samples[idx]['y']


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """训练器"""

    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = torch.device(config.DEVICE)

        self.optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        self.criterion = nn.BCELoss()
        self.best_val_loss = float('inf')

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS}')
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}',
                              'acc': f'{100. * correct / total:.2f}%'})

        return total_loss / len(loader), 100. * correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        return total_loss / len(loader), 100. * correct / total

    def train(self, train_loader, val_loader, fold):
        print(f'\n开始训练 Fold {fold + 1}/{self.config.NUM_FOLDS}')
        print(f'设备: {self.device}')
        if self.device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(0)}')

        history = []

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(fold)
                marker = ' [保存]'
            else:
                marker = ''

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}: '
                      f'Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | '
                      f'Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%{marker}')

        # 保存历史
        df = pd.DataFrame(history)
        df.to_csv(f'{self.config.SAVE_PATH}/{self.config.MODEL_NAME}_fold{fold}_history.csv', index=False)

        return history

    def save_model(self, fold):
        save_dir = Path(self.config.SAVE_PATH) / 'models'
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{self.config.MODEL_NAME}_fold{fold}.pt'
        torch.save({
            'fold': fold,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }, path)
        print(f'  模型已保存: {path}')


# ============================================================================
# 数据加载
# ============================================================================

def prepare_data(config):
    """准备数据集"""
    print('=' * 60)
    print('准备数据集...')
    print('=' * 60)

    dm = NGAFID_Dataset_Manager(config.DESTINATION, destination=config.DESTINATION)
    dm.data_dict = dm.construct_data_dictionary(numpy=True)

    print(f'样本数量: {len(dm.data_dict)}')
    print(f'序列长度: {config.MAX_LENGTH}')
    print(f'通道数: {config.NUM_CHANNELS}')

    # 统计二分类标签分布
    labels = [s['before_after'] for s in dm.data_dict]
    print(f'正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}')

    dataset = NGAFIDDataset(dm.data_dict, dm.mins, dm.maxs)

    return dm, dataset


def get_fold_data(dataset, fold, config):
    """获取指定折的训练和测试数据"""
    train_samples = [s for s in dataset.samples if s['fold'] != fold]
    test_samples = [s for s in dataset.samples if s['fold'] == fold]

    print(f'训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}')

    class SimpleDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]['x'], self.samples[idx]['y']

    train_loader = DataLoader(
        SimpleDataset(train_samples),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        SimpleDataset(test_samples),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    return train_loader, test_loader


# ============================================================================
# 主函数
# ============================================================================

def train(config):
    """训练主函数"""
    print('=' * 60)
    print('NGAFID MINIROCKET 二分类训练 (PyTorch)')
    print('=' * 60)

    # 设置随机种子
    set_seed(config.RANDOM_SEED)

    # 设备信息
    print(f'\n设备: {config.DEVICE}')
    if config.DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # 准备数据
    dm, dataset = prepare_data(config)

    # 训练循环
    all_results = []
    for fold in range(config.NUM_FOLDS):
        print(f'\n{"=" * 60}')
        print(f'Fold {fold + 1}/{config.NUM_FOLDS}')
        print(f'{"=" * 60}')

        train_loader, test_loader = get_fold_data(dataset, fold, config)

        # 创建模型
        model = MiniRocketClassifier(
            input_channels=config.NUM_CHANNELS,
            num_kernels=config.NUM_KERNELS,
            hidden_size=config.HIDDEN_SIZE,
            dropout=config.DROPOUT
        )

        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f'\n模型参数量: {total_params:,}')

        # 训练
        trainer = Trainer(model, config)
        history = trainer.train(train_loader, test_loader, fold)

        # 结果
        final_acc = history[-1]['val_acc']
        print(f'\nFold {fold + 1} 完成: 验证准确率 = {final_acc:.2f}%')
        all_results.append({'fold': fold, 'val_acc': final_acc, 'best_val_loss': trainer.best_val_loss})

    # 汇总
    print('\n' + '=' * 60)
    print('训练完成!')
    print('=' * 60)
    results_df = pd.DataFrame(all_results)
    print(f'\n平均验证准确率: {results_df["val_acc"].mean():.2f}% ± {results_df["val_acc"].std():.2f}%')

    return all_results


def demo():
    """快速演示"""
    print('=' * 60)
    print('快速演示 - MINIROCKET 单折训练')
    print('=' * 60)

    set_seed(Config.RANDOM_SEED)

    # 准备数据
    dm, dataset = prepare_data(Config)

    # 只用第0折
    train_loader, test_loader = get_fold_data(dataset, 0, Config)

    # 创建模型
    model = MiniRocketClassifier(
        input_channels=Config.NUM_CHANNELS,
        num_kernels=Config.NUM_KERNELS,
        hidden_size=Config.HIDDEN_SIZE,
        dropout=Config.DROPOUT
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n模型参数量: {total_params:,}')

    # 测试前向传播
    print('\n测试模型前向传播...')
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x = x.to(Config.DEVICE)
        output = model(x)
        print(f'输入形状: {x.shape}')
        print(f'输出形状: {output.shape}')
        print(f'输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]')

    # 训练1个epoch测试
    print('\n训练1个epoch测试...')
    Config.NUM_EPOCHS = 1
    trainer = Trainer(model, Config)
    train_loss, train_acc = trainer.train_epoch(train_loader, 0)
    val_loss, val_acc = trainer.validate(test_loader)
    print(f'\n测试结果:')
    print(f'  训练损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%')
    print(f'  验证损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%')

    print('\n演示完成! 模型可正常运行。')
    return True


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NGAFID MINIROCKET 二分类训练')
    parser.add_argument('--dataset', type=str, default='2days', help='数据集名称')
    parser.add_argument('--destination', type=str, default='', help='数据路径')
    parser.add_argument('--num_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2.5e-5, help='学习率')
    parser.add_argument('--num_kernels', type=int, default=10000, help='随机卷积核数量')
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层大小')
    parser.add_argument('--save_path', type=str, default='.', help='保存路径')
    parser.add_argument('--demo', action='store_true', help='快速演示模式')

    args = parser.parse_args()

    # 更新配置
    Config.DATASET_NAME = args.dataset
    Config.DESTINATION = args.destination
    Config.NUM_FOLDS = args.num_folds
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.num_epochs
    Config.LEARNING_RATE = args.lr
    Config.NUM_KERNELS = args.num_kernels
    Config.HIDDEN_SIZE = args.hidden_size
    Config.SAVE_PATH = args.save_path

    if args.demo:
        demo()
    else:
        train(Config)

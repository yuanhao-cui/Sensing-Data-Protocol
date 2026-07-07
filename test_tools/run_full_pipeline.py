"""
完整 pipeline 演示脚本（使用 data/xrf55 真实数据）
===========================================
说明：
- 使用 XRF55Reader (XrfReader) 自动解析 .npy 文件
- 使用 ConfigurableProcessor 自定义算法 pipeline
- 使用 GroupShuffleSplit 划分训练/验证/测试集
- 训练并评估模型

如何更换 algorithm：
    修改 pipeline_steps 字典即可。可用方法可通过 wsdp.algorithms.list_algorithms() 查看，
    或使用预设：wsdp.algorithms.apply_preset('high_quality'/'fast'/'robust')

如何更换 model：
    修改 MODEL_NAME 变量即可。可用模型可通过 wsdp.models.list_models() 查看。
    例如：'mlpmodel', 'cnn1dmodel', 'resnet1d', 'CSIModel' 等。
"""

import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from wsdp import readers
from wsdp.processors import ConfigurableProcessor
from wsdp.models import create_model, list_models
from wsdp.datasets import CSIDataset
from wsdp.utils import resize_csi_to_fixed_length, train_model
from wsdp.algorithms import apply_preset

# ==================== 配置区 ====================
DATA_PATH = "data/xrf55"
DATASET_NAME = "xrf55"

# 更换算法：修改下面的 pipeline_steps
# 可用预设：apply_preset('high_quality') / apply_preset('fast') / apply_preset('robust')
pipeline_steps = apply_preset('robust')
# pipeline_steps = {
#     'denoise': {'method': 'bandpass'},
#     'calibrate': {'method': 'stc'},
#     'normalize': {'method': 'z-score'},
# }

# 更换模型：修改 MODEL_NAME
# 可用模型：print(list_models())
# 推荐基线：'mlpmodel', 'cnn1dmodel', 'cnn2dmodel', 'lstmmodel', 'resnet1d'
# 推荐 SOTA：'CSIModel', 'attentiongru', 'mambacsi'
MODEL_NAME = "cnn1dmodel"

# 训练超参数
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 5
PADDING_LENGTH = 1000  # xrf55 的默认 padding_length 为 1000
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 42
# ===============================================


def main():
    print("=" * 60)
    print("WSDP 完整 Pipeline 演示")
    print("=" * 60)

    # Step 1: 加载数据（自动使用 XRF55Reader / XrfReader）
    print(f"\n📦 Step 1: 加载数据 ({DATA_PATH})")
    print("   说明: readers.load_data() 会根据 dataset='xrf55' 自动选择 XrfReader")
    csi_data_list = readers.load_data(DATA_PATH, DATASET_NAME)
    print(f"   共加载 {len(csi_data_list)} 个 CSI 样本")

    # Step 2: 使用 ConfigurableProcessor 处理数据
    print("\n🔧 Step 2: 算法处理")
    print(f"   当前 pipeline: {pipeline_steps}")
    print("   💡 更换算法：修改上面的 pipeline_steps 字典")
    processor = ConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(csi_data_list, dataset=DATASET_NAME)

    print(f"   处理完成: {len(all_data)} 个样本")
    print(f"   标签分布: {dict((x, all_labels.count(x)) for x in set(all_labels))}")
    print(f"   分组分布: {dict((x, all_groups.count(x)) for x in set(all_groups))}")
    print(f"   样本形状: {all_data[0].shape}")

    # Step 3: 数据规整化
    print(f"\n📐 Step 3: 长度归一化 (padding_length={PADDING_LENGTH})")
    processed_data = resize_csi_to_fixed_length(all_data, target_length=PADDING_LENGTH)
    processed_data = np.array(processed_data)
    labels = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"   数据 shape: {processed_data.shape}")

    # 标签重新编码为 0-based
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = np.array([label_map[label] for label in labels])
    num_classes = len(unique_labels)
    print(f"   类别数: {num_classes} (原始标签: {unique_labels})")

    # Step 4: 数据划分
    print("\n✂️ Step 4: 数据集划分")
    n_groups = len(set(groups))
    if n_groups < 3:
        print(f"   警告: 只有 {n_groups} 个 group，使用简单随机划分替代 GroupShuffleSplit")
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            processed_data, zero_indexed_labels, test_size=TEST_SPLIT, random_state=SEED
        )
        test_data, val_data, test_labels, val_labels = train_test_split(
            temp_data, temp_labels, test_size=VAL_SPLIT, random_state=SEED
        )
    else:
        splitter_1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
        train_idx, temp_idx = next(
            splitter_1.split(processed_data, zero_indexed_labels, groups=groups)
        )
        train_data = processed_data[train_idx]
        train_labels = zero_indexed_labels[train_idx]

        temp_data = processed_data[temp_idx]
        temp_labels = zero_indexed_labels[temp_idx]
        temp_groups = groups[temp_idx]

        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
        test_idx, val_idx = next(splitter_2.split(temp_data, temp_labels, groups=temp_groups))

        test_data = temp_data[test_idx]
        test_labels = temp_labels[test_idx]
        val_data = temp_data[val_idx]
        val_labels = temp_labels[val_idx]

    print(f"   训练集: {len(train_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}")

    # Step 5: 创建 DataLoader
    train_dataset = CSIDataset(
        train_data,
        train_labels,
        dataset_name=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )
    val_dataset = CSIDataset(
        val_data,
        val_labels,
        dataset_name=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )
    test_dataset = CSIDataset(
        test_data,
        test_labels,
        dataset_name=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 6: 创建模型
    print("\n🧠 Step 5: 创建模型")
    print(f"   当前模型: {MODEL_NAME}")
    print("   💡 更换模型：修改上面的 MODEL_NAME 变量")
    print("   可用模型示例:")
    for name, cat in sorted(list_models().items())[:5]:
        print(f"      [{cat}] {name}")
    print("      ... (更多模型请运行 list_models() 查看)")

    input_shape = tuple(train_dataset.data_list.shape[1:])
    model = create_model(MODEL_NAME, num_classes=num_classes, input_shape=input_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"   训练设备: {device}")

    # Step 7: 训练
    if num_classes < 2:
        print(f"\n⚠️  当前数据集只有 {num_classes} 个类别，训练结果仅作演示用途。")
        print("    提示：使用包含多个 action 的完整数据集可获得有意义的分类评估。")

    print(f"\n🚀 Step 6: 开始训练 ({NUM_EPOCHS} epochs)")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    checkpoint_path = "output/best_checkpoint_demo.pth"
    os.makedirs("output", exist_ok=True)

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        checkpoint_path=checkpoint_path,
        padding_length=PADDING_LENGTH,
    )
    print(f"   训练完成，最佳模型保存至: {checkpoint_path}")

    # Step 8: 评估
    print("\n📊 Step 7: 测试集评估")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"   测试集准确率: {accuracy:.4f}")

    print("\n" + "=" * 60)
    print("✅ Pipeline 演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

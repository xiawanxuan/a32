import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from typing import Dict, List, Tuple


def compute_overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算总体精度（Overall Accuracy）

    Args:
        y_true: 真实标签 (H, W) 或 (N,)
        y_pred: 预测标签 (H, W) 或 (N,)

    Returns:
        总体精度值
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return accuracy_score(y_true_flat, y_pred_flat)


def compute_kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算Kappa系数

    Args:
        y_true: 真实标签 (H, W) 或 (N,)
        y_pred: 预测标签 (H, W) 或 (N,)

    Returns:
        Kappa系数值
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return cohen_kappa_score(y_true_flat, y_pred_flat)


def compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> Dict:
    """
    计算各类别的评价指标（精确率、召回率、F1分数）

    Args:
        y_true: 真实标签 (H, W) 或 (N,)
        y_pred: 预测标签 (H, W) 或 (N,)
        class_names: 类别名称列表

    Returns:
        包含各类别指标的字典
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    target_names = class_names if class_names else None
    report = classification_report(
        y_true_flat, y_pred_flat,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    return report


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    计算交并比（IoU, Intersection over Union）

    Args:
        y_true: 真实标签 (H, W) 或 (N,)
        y_pred: 预测标签 (H, W) 或 (N,)
        num_classes: 类别数量

    Returns:
        各类别的IoU数组
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    iou_list = []
    for cls in range(num_classes):
        intersection = np.sum((y_true_flat == cls) & (y_pred_flat == cls))
        union = np.sum((y_true_flat == cls) | (y_pred_flat == cls))
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        iou_list.append(iou)

    return np.array(iou_list)


def compute_dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    计算Dice系数

    Args:
        y_true: 真实标签 (H, W) 或 (N,)
        y_pred: 预测标签 (H, W) 或 (N,)
        num_classes: 类别数量

    Returns:
        各类别的Dice系数数组
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    dice_list = []
    for cls in range(num_classes):
        intersection = np.sum((y_true_flat == cls) & (y_pred_flat == cls))
        y_true_sum = np.sum(y_true_flat == cls)
        y_pred_sum = np.sum(y_pred_flat == cls)
        if y_true_sum + y_pred_sum == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / (y_true_sum + y_pred_sum)
        dice_list.append(dice)

    return np.array(dice_list)


def compute_pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算像素级精度和每类精度

    Args:
        y_true: 真实标签 (H, W)
        y_pred: 预测标签 (H, W)

    Returns:
        总体精度和每类精度数组
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    overall_acc = accuracy_score(y_true_flat, y_pred_flat)

    classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    class_acc = []
    for cls in classes:
        mask = y_true_flat == cls
        if np.sum(mask) == 0:
            class_acc.append(1.0)
        else:
            cls_acc = accuracy_score(y_true_flat[mask], y_pred_flat[mask])
            class_acc.append(cls_acc)

    return overall_acc, np.array(class_acc)


def compute_change_detection_metrics(
    change_true: np.ndarray,
    change_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算变化检测的评价指标

    Args:
        change_true: 真实变化图 (H, W)，1表示变化，0表示未变化
        change_pred: 预测变化图 (H, W)，1表示变化，0表示未变化

    Returns:
        包含各项指标的字典
    """
    y_true = change_true.flatten().astype(np.int32)
    y_pred = change_pred.flatten().astype(np.int32)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'iou': iou,
        'kappa': kappa,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

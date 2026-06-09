import os
import numpy as np
from PIL import Image
import json
import csv
from typing import List, Tuple, Dict, Optional, Union

try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def read_image(file_path: str) -> np.ndarray:
    """
    读取遥感影像，支持多种格式（TIFF, PNG, JPEG等）

    Args:
        file_path: 影像文件路径

    Returns:
        numpy数组形式的影像数据 (H, W, C)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.tif', '.tiff', '.geotiff'] and RASTERIO_AVAILABLE:
        return read_geotiff(file_path)
    else:
        img = Image.open(file_path)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = img_array[:, :, np.newaxis]
        return img_array


def write_image(file_path: str, image: np.ndarray) -> None:
    """
    写入影像文件

    Args:
        file_path: 输出文件路径
        image: 影像数据 (H, W, C) 或 (H, W)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.tif', '.tiff', '.geotiff'] and RASTERIO_AVAILABLE:
        write_geotiff(file_path, image)
    else:
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        img = Image.fromarray(image.astype(np.uint8))
        img.save(file_path)


def read_label(file_path: str) -> np.ndarray:
    """
    读取标签影像

    Args:
        file_path: 标签文件路径

    Returns:
        标签数组 (H, W)，值为类别索引
    """
    label = read_image(file_path)
    if label.ndim == 3:
        if label.shape[2] == 1:
            label = label[:, :, 0]
        else:
            label = np.argmax(label, axis=2)
    return label.astype(np.int32)


def write_label(file_path: str, label: np.ndarray, colormap: Optional[Dict[int, Tuple[int, int, int]]] = None) -> None:
    """
    写入标签影像，可选择带颜色映射

    Args:
        file_path: 输出文件路径
        label: 标签数组 (H, W)
        colormap: 颜色映射字典 {class_idx: (R, G, B)}
    """
    if colormap is not None:
        h, w = label.shape
        color_label = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in colormap.items():
            mask = label == class_idx
            color_label[mask] = color
        write_image(file_path, color_label)
    else:
        write_image(file_path, label.astype(np.uint8))


def read_geotiff(file_path: str) -> np.ndarray:
    """
    读取GeoTIFF格式影像

    Args:
        file_path: GeoTIFF文件路径

    Returns:
        影像数据 (H, W, C)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for GeoTIFF support")

    with rasterio.open(file_path) as src:
        data = src.read()
        data = np.transpose(data, (1, 2, 0))
    return data


def write_geotiff(file_path: str, image: np.ndarray, transform=None, crs=None) -> None:
    """
    写入GeoTIFF格式影像

    Args:
        file_path: 输出文件路径
        image: 影像数据 (H, W, C) 或 (H, W)
        transform: 地理变换参数
        crs: 坐标参考系统
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for GeoTIFF support")

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, c = image.shape
    data = np.transpose(image, (2, 0, 1))

    if transform is None:
        transform = from_origin(0, 0, 1, 1)

    with rasterio.open(
        file_path, 'w',
        driver='GTiff',
        height=h, width=w, count=c,
        dtype=data.dtype,
        transform=transform,
        crs=crs
    ) as dst:
        dst.write(data)


def load_dataset_paths(image_dir: str, label_dir: str, image_ext: str = '.tif') -> Tuple[List[str], List[str]]:
    """
    加载数据集路径列表

    Args:
        image_dir: 影像目录路径
        label_dir: 标签目录路径
        image_ext: 影像文件扩展名

    Returns:
        影像路径列表和标签路径列表
    """
    image_paths = []
    label_paths = []

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(image_ext):
            image_path = os.path.join(image_dir, filename)
            label_name = os.path.splitext(filename)[0] + '_label' + image_ext
            label_path = os.path.join(label_dir, label_name)

            if not os.path.exists(label_path):
                label_name = os.path.splitext(filename)[0] + image_ext
                label_path = os.path.join(label_dir, label_name)

            if os.path.exists(label_path):
                image_paths.append(image_path)
                label_paths.append(label_path)

    return image_paths, label_paths


def save_model_history(history: Dict, file_path: str) -> None:
    """
    保存模型训练历史

    Args:
        history: 训练历史字典
        file_path: 保存路径
    """
    history_serializable = {}
    for key, value in history.items():
        history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)


def load_model_history(file_path: str) -> Dict:
    """
    加载模型训练历史

    Args:
        file_path: 历史文件路径

    Returns:
        训练历史字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history


def export_classification_report(report_dict: Dict, file_path: str) -> None:
    """
    导出分类报告为CSV文件

    Args:
        report_dict: 分类报告字典
        file_path: 输出CSV文件路径
    """
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['类别', '精确率', '召回率', 'F1分数', '样本数'])

        for class_name, metrics in report_dict.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            writer.writerow([
                class_name,
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('f1-score', 0):.4f}",
                metrics.get('support', 0)
            ])

        if 'macro avg' in report_dict:
            writer.writerow([])
            writer.writerow([
                '宏平均',
                f"{report_dict['macro avg']['precision']:.4f}",
                f"{report_dict['macro avg']['recall']:.4f}",
                f"{report_dict['macro avg']['f1-score']:.4f}",
                report_dict['macro avg']['support']
            ])

        if 'weighted avg' in report_dict:
            writer.writerow([
                '加权平均',
                f"{report_dict['weighted avg']['precision']:.4f}",
                f"{report_dict['weighted avg']['recall']:.4f}",
                f"{report_dict['weighted avg']['f1-score']:.4f}",
                report_dict['weighted avg']['support']
            ])

        if 'accuracy' in report_dict:
            writer.writerow([])
            writer.writerow(['总体准确率', f"{report_dict['accuracy']:.4f}"])


def export_confusion_matrix(confusion_mat: np.ndarray, class_names: List[str], file_path: str) -> None:
    """
    导出混淆矩阵为CSV文件

    Args:
        confusion_mat: 混淆矩阵 (n_classes, n_classes)
        class_names: 类别名称列表
        file_path: 输出CSV文件路径
    """
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ['真实\\预测'] + class_names
        writer.writerow(header)

        for i, class_name in enumerate(class_names):
            row = [class_name] + [str(x) for x in confusion_mat[i]]
            writer.writerow(row)


def export_prediction_results(results: List[Dict], file_path: str) -> None:
    """
    导出批量推理结果为CSV文件

    Args:
        results: 结果列表，每个元素是包含文件名和统计信息的字典
        file_path: 输出CSV文件路径
    """
    if not results:
        return

    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        keys = list(results[0].keys())
        writer.writerow(keys)

        for result in results:
            row = [result.get(key, '') for key in keys]
            writer.writerow(row)

import os
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Optional, Generator, Callable
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

from configs.config import Config
from utils.io_utils import read_image, read_label, load_dataset_paths


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    影像预处理：调整大小、归一化

    Args:
        image: 输入影像 (H, W, C)
        target_size: 目标尺寸 (height, width)
        normalize: 是否归一化
        mean: 均值列表
        std: 标准差列表

    Returns:
        预处理后的影像
    """
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if normalize:
        image = image.astype(np.float32) / 255.0
        if mean is not None and std is not None:
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            image = (image - mean) / std

    return image.astype(np.float32)


def preprocess_label(
    label: np.ndarray,
    target_size: Tuple[int, int],
    num_classes: int,
    one_hot: bool = True
) -> np.ndarray:
    """
    标签预处理：调整大小、one-hot编码

    Args:
        label: 输入标签 (H, W)
        target_size: 目标尺寸 (height, width)
        num_classes: 类别数量
        one_hot: 是否进行one-hot编码

    Returns:
        预处理后的标签
    """
    if label.shape[:2] != target_size:
        label = cv2.resize(label, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    label = label.astype(np.int32)

    if one_hot:
        label_one_hot = np.zeros((target_size[0], target_size[1], num_classes), dtype=np.float32)
        for c in range(num_classes):
            label_one_hot[:, :, c] = (label == c).astype(np.float32)
        return label_one_hot

    return label


def denormalize_image(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    反归一化影像

    Args:
        image: 归一化后的影像
        mean: 均值列表
        std: 标准差列表

    Returns:
        反归一化后的影像 (0-255)
    """
    img = image.copy()
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img = img * std + mean

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def split_dataset(
    image_paths: List[str],
    label_paths: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    划分数据集为训练集、验证集、测试集

    Args:
        image_paths: 影像路径列表
        label_paths: 标签路径列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据集比例之和必须为1"

    train_img, temp_img, train_lbl, temp_lbl = train_test_split(
        image_paths, label_paths,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True
    )

    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_img, test_img, val_lbl, test_lbl = train_test_split(
        temp_img, temp_lbl,
        test_size=(1 - val_test_ratio),
        random_state=random_seed,
        shuffle=True
    )

    return train_img, train_lbl, val_img, val_lbl, test_img, test_lbl


def get_augmentation_pipeline():
    """
    获取数据增强流水线

    Returns:
        数据增强函数
    """
    try:
        import albumentations as A

        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ])

        def augment(image, label):
            augmented = transform(image=image, mask=label)
            return augmented['image'], augmented['mask']

        return augment
    except ImportError:
        def augment(image, label):
            if np.random.random() > 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)
            if np.random.random() > 0.5:
                image = np.flipud(image)
                label = np.flipud(label)
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            return image.copy(), label.copy()

        return augment


class DataGenerator(tf.keras.utils.Sequence):
    """
    数据生成器，用于批量加载和预处理数据
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        config: Config,
        batch_size: int = 8,
        shuffle: bool = True,
        augment: bool = False
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augment_fn = get_augmentation_pipeline() if augment else None
        self.indexes = np.arange(len(self.image_paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for idx in batch_indexes:
            img_path = self.image_paths[idx]
            lbl_path = self.label_paths[idx]

            image = read_image(img_path)
            label = read_label(lbl_path)

            image = preprocess_image(
                image,
                self.config.data.image_size,
                normalize=False
            )

            label = preprocess_label(
                label,
                self.config.data.image_size,
                self.config.data.num_classes,
                one_hot=False
            )

            if self.augment and self.augment_fn:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                image, label = self.augment_fn(image, label)
                image = image.astype(np.float32)

            if self.config.data.normalize:
                image = image.astype(np.float32) / 255.0
                if self.config.data.image_mean and self.config.data.image_std:
                    mean = np.array(self.config.data.image_mean, dtype=np.float32)
                    std = np.array(self.config.data.image_std, dtype=np.float32)
                    image = (image - mean) / std

            label_one_hot = np.zeros(
                (self.config.data.image_size[0], self.config.data.image_size[1], self.config.data.num_classes),
                dtype=np.float32
            )
            for c in range(self.config.data.num_classes):
                label_one_hot[:, :, c] = (label == c).astype(np.float32)

            batch_images.append(image)
            batch_labels.append(label_one_hot)

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class RemoteSensingDataPreprocessor:
    """
    遥感影像数据预处理器
    """

    def __init__(self, config: Config):
        self.config = config

    def load_and_preprocess_dataset(self) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
        """
        加载并预处理数据集，返回数据生成器

        Returns:
            (train_generator, val_generator, test_generator)
        """
        image_paths, label_paths = load_dataset_paths(
            self.config.data.image_dir,
            self.config.data.label_dir
        )

        if len(image_paths) == 0:
            raise ValueError(f"在 {self.config.data.image_dir} 和 {self.config.data.label_dir} 中未找到匹配的数据")

        print(f"共找到 {len(image_paths)} 对影像-标签数据")

        train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = split_dataset(
            image_paths, label_paths,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            test_ratio=self.config.data.test_ratio,
            random_seed=self.config.data.random_seed
        )

        print(f"训练集: {len(train_img)} 张")
        print(f"验证集: {len(val_img)} 张")
        print(f"测试集: {len(test_img)} 张")

        train_generator = DataGenerator(
            train_img, train_lbl,
            self.config,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            augment=self.config.train.use_augmentation
        )

        val_generator = DataGenerator(
            val_img, val_lbl,
            self.config,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            augment=False
        )

        test_generator = DataGenerator(
            test_img, test_lbl,
            self.config,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            augment=False
        )

        return train_generator, val_generator, test_generator

    def preprocess_single_image(self, image_path: str) -> np.ndarray:
        """
        预处理单张影像用于推理

        Args:
            image_path: 影像文件路径

        Returns:
            预处理后的影像 (1, H, W, C)
        """
        image = read_image(image_path)
        image = preprocess_image(
            image,
            self.config.data.image_size,
            normalize=self.config.data.normalize,
            mean=self.config.data.image_mean if self.config.data.normalize else None,
            std=self.config.data.image_std if self.config.data.normalize else None
        )
        return np.expand_dims(image, axis=0)

    def preprocess_directory(self, input_dir: str, output_dir: str) -> List[str]:
        """
        批量预处理目录中的所有影像

        Args:
            input_dir: 输入目录
            output_dir: 输出目录

        Returns:
            处理后的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []

        for filename in tqdm(os.listdir(input_dir), desc="预处理影像"):
            if filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                image = read_image(input_path)
                processed = preprocess_image(
                    image,
                    self.config.data.image_size,
                    normalize=False
                )

                from utils.io_utils import write_image
                write_image(output_path, processed.astype(np.uint8))
                processed_paths.append(output_path)

        return processed_paths


def sliding_window_inference(
    image: np.ndarray,
    model: tf.keras.Model,
    window_size: Tuple[int, int],
    overlap: int = 32,
    num_classes: int = 6
) -> np.ndarray:
    """
    滑动窗口推理，用于处理大尺寸影像

    Args:
        image: 输入大影像 (H, W, C)
        model: 训练好的模型
        window_size: 窗口大小 (height, width)
        overlap: 重叠像素数
        num_classes: 类别数量

    Returns:
        预测结果 (H, W, num_classes)
    """
    h, w, c = image.shape
    win_h, win_w = window_size
    step_h = win_h - overlap
    step_w = win_w - overlap

    prediction = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            y_end = min(y + win_h, h)
            x_end = min(x + win_w, w)
            y_start = y_end - win_h
            x_start = x_end - win_w

            if y_start < 0:
                y_start = 0
            if x_start < 0:
                x_start = 0

            patch = image[y_start:y_end, x_start:x_end, :]

            if patch.shape[0] != win_h or patch.shape[1] != win_w:
                patch_padded = np.zeros((win_h, win_w, c), dtype=image.dtype)
                patch_padded[:patch.shape[0], :patch.shape[1], :] = patch
                patch = patch_padded

            patch_input = np.expand_dims(patch, axis=0)
            pred_patch = model.predict(patch_input, verbose=0)[0]

            actual_h = y_end - y_start
            actual_w = x_end - x_start
            pred_crop = pred_patch[:actual_h, :actual_w, :]

            prediction[y_start:y_end, x_start:x_end, :] += pred_crop
            count_map[y_start:y_end, x_start:x_end] += 1

    count_map[count_map == 0] = 1
    prediction /= count_map[:, :, np.newaxis]

    return prediction


class ChangeDetectionDataPreprocessor:
    """
    变化检测数据预处理器
    """

    def __init__(self, config: Config):
        self.config = config

    def load_change_detection_pair(self, t1_path: str, t2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载时相1和时相2的影像对

        Args:
            t1_path: 时相1影像路径
            t2_path: 时相2影像路径

        Returns:
            (t1_image, t2_image) 预处理后的影像
        """
        t1_image = read_image(t1_path)
        t2_image = read_image(t2_path)

        t1_image = preprocess_image(
            t1_image,
            self.config.data.image_size,
            normalize=self.config.data.normalize,
            mean=self.config.data.image_mean if self.config.data.normalize else None,
            std=self.config.data.image_std if self.config.data.normalize else None
        )

        t2_image = preprocess_image(
            t2_image,
            self.config.data.image_size,
            normalize=self.config.data.normalize,
            mean=self.config.data.image_mean if self.config.data.normalize else None,
            std=self.config.data.image_std if self.config.data.normalize else None
        )

        return t1_image, t2_image

    def compute_difference_image(self, t1_image: np.ndarray, t2_image: np.ndarray) -> np.ndarray:
        """
        计算差值影像

        Args:
            t1_image: 时相1影像
            t2_image: 时相2影像

        Returns:
            差值影像
        """
        diff = np.abs(t2_image - t1_image)
        return diff

    def create_concatenated_input(self, t1_image: np.ndarray, t2_image: np.ndarray) -> np.ndarray:
        """
        创建拼接输入（用于双输入模型）

        Args:
            t1_image: 时相1影像 (H, W, C)
            t2_image: 时相2影像 (H, W, C)

        Returns:
            拼接后的影像 (H, W, 2*C)
        """
        return np.concatenate([t1_image, t2_image], axis=-1)

    def load_change_detection_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """
        加载变化检测数据集

        Returns:
            (t1_paths, t2_paths, label_paths)
        """
        t1_dir = self.config.change_detection.image_t1_dir
        t2_dir = self.config.change_detection.image_t2_dir
        label_dir = self.config.data.label_dir

        t1_paths = []
        t2_paths = []
        label_paths = []

        if not os.path.exists(t1_dir) or not os.path.exists(t2_dir):
            return t1_paths, t2_paths, label_paths

        for filename in sorted(os.listdir(t1_dir)):
            t1_path = os.path.join(t1_dir, filename)
            t2_path = os.path.join(t2_dir, filename)
            label_path = os.path.join(label_dir, filename)

            if os.path.exists(t1_path) and os.path.exists(t2_path):
                t1_paths.append(t1_path)
                t2_paths.append(t2_path)
                if os.path.exists(label_path):
                    label_paths.append(label_path)

        return t1_paths, t2_paths, label_paths

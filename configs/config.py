import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import json


@dataclass
class DataConfig:
    """数据配置"""
    image_dir: str = 'data/raw/images'
    label_dir: str = 'data/raw/labels'
    processed_dir: str = 'data/processed'
    image_size: Tuple[int, int] = (256, 256)
    num_classes: int = 6
    in_channels: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42
    normalize: bool = True
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    class_names: List[str] = field(default_factory=lambda: [
        '建筑物', '植被', '水体', '道路', '裸地', '其他'
    ])
    class_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (128, 128, 128),
        5: (0, 0, 0)
    })


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = 'unet'
    backbone: str = 'resnet50'
    pretrained: bool = True
    dropout_rate: float = 0.2
    use_attention: bool = False
    filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss_function: str = 'categorical_crossentropy'
    use_augmentation: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 10
    reduce_lr: bool = True
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    model_save_dir: str = 'outputs/models'
    model_name: str = 'remote_sensing_unet'
    save_best_only: bool = True
    monitor_metric: str = 'val_loss'


@dataclass
class ChangeDetectionConfig:
    """变化检测配置"""
    image_t1_dir: str = 'data/raw/time1'
    image_t2_dir: str = 'data/raw/time2'
    change_threshold: float = 0.5
    use_difference: bool = True
    use_concatenation: bool = False
    min_change_area: int = 10


@dataclass
class OutputConfig:
    """输出配置"""
    result_dir: str = 'outputs/results'
    visualization_dir: str = 'outputs/visualizations'
    save_predictions: bool = True
    save_visualizations: bool = True
    save_metrics: bool = True
    export_format: str = 'png'
    colormap_visualization: bool = True
    overlay_visualization: bool = True
    overlay_alpha: float = 0.5


@dataclass
class Config:
    """整体配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    change_detection: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        self._make_dirs()

    def _make_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.data.processed_dir,
            self.train.model_save_dir,
            self.output.result_dir,
            self.output.visualization_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'data': {
                'image_dir': self.data.image_dir,
                'label_dir': self.data.label_dir,
                'processed_dir': self.data.processed_dir,
                'image_size': list(self.data.image_size),
                'num_classes': self.data.num_classes,
                'in_channels': self.data.in_channels,
                'train_ratio': self.data.train_ratio,
                'val_ratio': self.data.val_ratio,
                'test_ratio': self.data.test_ratio,
                'random_seed': self.data.random_seed,
                'normalize': self.data.normalize,
                'image_mean': self.data.image_mean,
                'image_std': self.data.image_std,
                'class_names': self.data.class_names,
                'class_colors': {str(k): list(v) for k, v in self.data.class_colors.items()},
            },
            'model': {
                'model_type': self.model.model_type,
                'backbone': self.model.backbone,
                'pretrained': self.model.pretrained,
                'dropout_rate': self.model.dropout_rate,
                'use_attention': self.model.use_attention,
                'filters': self.model.filters,
            },
            'train': {
                'batch_size': self.train.batch_size,
                'epochs': self.train.epochs,
                'learning_rate': self.train.learning_rate,
                'optimizer': self.train.optimizer,
                'loss_function': self.train.loss_function,
                'use_augmentation': self.train.use_augmentation,
                'early_stopping': self.train.early_stopping,
                'early_stopping_patience': self.train.early_stopping_patience,
                'reduce_lr': self.train.reduce_lr,
                'reduce_lr_patience': self.train.reduce_lr_patience,
                'reduce_lr_factor': self.train.reduce_lr_factor,
                'model_save_dir': self.train.model_save_dir,
                'model_name': self.train.model_name,
                'save_best_only': self.train.save_best_only,
                'monitor_metric': self.train.monitor_metric,
            },
            'change_detection': {
                'image_t1_dir': self.change_detection.image_t1_dir,
                'image_t2_dir': self.change_detection.image_t2_dir,
                'change_threshold': self.change_detection.change_threshold,
                'use_difference': self.change_detection.use_difference,
                'use_concatenation': self.change_detection.use_concatenation,
                'min_change_area': self.change_detection.min_change_area,
            },
            'output': {
                'result_dir': self.output.result_dir,
                'visualization_dir': self.output.visualization_dir,
                'save_predictions': self.output.save_predictions,
                'save_visualizations': self.output.save_visualizations,
                'save_metrics': self.output.save_metrics,
                'export_format': self.output.export_format,
                'colormap_visualization': self.output.colormap_visualization,
                'overlay_visualization': self.output.overlay_visualization,
                'overlay_alpha': self.output.overlay_alpha,
            }
        }

    def save(self, file_path: str):
        """保存配置到JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> 'Config':
        """从JSON文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = cls()

        if 'data' in config_dict:
            d = config_dict['data']
            config.data.image_dir = d.get('image_dir', config.data.image_dir)
            config.data.label_dir = d.get('label_dir', config.data.label_dir)
            config.data.processed_dir = d.get('processed_dir', config.data.processed_dir)
            config.data.image_size = tuple(d.get('image_size', list(config.data.image_size)))
            config.data.num_classes = d.get('num_classes', config.data.num_classes)
            config.data.in_channels = d.get('in_channels', config.data.in_channels)
            config.data.train_ratio = d.get('train_ratio', config.data.train_ratio)
            config.data.val_ratio = d.get('val_ratio', config.data.val_ratio)
            config.data.test_ratio = d.get('test_ratio', config.data.test_ratio)
            config.data.random_seed = d.get('random_seed', config.data.random_seed)
            config.data.normalize = d.get('normalize', config.data.normalize)
            config.data.image_mean = d.get('image_mean', config.data.image_mean)
            config.data.image_std = d.get('image_std', config.data.image_std)
            config.data.class_names = d.get('class_names', config.data.class_names)
            if 'class_colors' in d:
                config.data.class_colors = {int(k): tuple(v) for k, v in d['class_colors'].items()}

        if 'model' in config_dict:
            m = config_dict['model']
            config.model.model_type = m.get('model_type', config.model.model_type)
            config.model.backbone = m.get('backbone', config.model.backbone)
            config.model.pretrained = m.get('pretrained', config.model.pretrained)
            config.model.dropout_rate = m.get('dropout_rate', config.model.dropout_rate)
            config.model.use_attention = m.get('use_attention', config.model.use_attention)
            config.model.filters = m.get('filters', config.model.filters)

        if 'train' in config_dict:
            t = config_dict['train']
            config.train.batch_size = t.get('batch_size', config.train.batch_size)
            config.train.epochs = t.get('epochs', config.train.epochs)
            config.train.learning_rate = t.get('learning_rate', config.train.learning_rate)
            config.train.optimizer = t.get('optimizer', config.train.optimizer)
            config.train.loss_function = t.get('loss_function', config.train.loss_function)
            config.train.use_augmentation = t.get('use_augmentation', config.train.use_augmentation)
            config.train.early_stopping = t.get('early_stopping', config.train.early_stopping)
            config.train.early_stopping_patience = t.get('early_stopping_patience', config.train.early_stopping_patience)
            config.train.reduce_lr = t.get('reduce_lr', config.train.reduce_lr)
            config.train.reduce_lr_patience = t.get('reduce_lr_patience', config.train.reduce_lr_patience)
            config.train.reduce_lr_factor = t.get('reduce_lr_factor', config.train.reduce_lr_factor)
            config.train.model_save_dir = t.get('model_save_dir', config.train.model_save_dir)
            config.train.model_name = t.get('model_name', config.train.model_name)
            config.train.save_best_only = t.get('save_best_only', config.train.save_best_only)
            config.train.monitor_metric = t.get('monitor_metric', config.train.monitor_metric)

        if 'change_detection' in config_dict:
            cd = config_dict['change_detection']
            config.change_detection.image_t1_dir = cd.get('image_t1_dir', config.change_detection.image_t1_dir)
            config.change_detection.image_t2_dir = cd.get('image_t2_dir', config.change_detection.image_t2_dir)
            config.change_detection.change_threshold = cd.get('change_threshold', config.change_detection.change_threshold)
            config.change_detection.use_difference = cd.get('use_difference', config.change_detection.use_difference)
            config.change_detection.use_concatenation = cd.get('use_concatenation', config.change_detection.use_concatenation)
            config.change_detection.min_change_area = cd.get('min_change_area', config.change_detection.min_change_area)

        if 'output' in config_dict:
            o = config_dict['output']
            config.output.result_dir = o.get('result_dir', config.output.result_dir)
            config.output.visualization_dir = o.get('visualization_dir', config.output.visualization_dir)
            config.output.save_predictions = o.get('save_predictions', config.output.save_predictions)
            config.output.save_visualizations = o.get('save_visualizations', config.output.save_visualizations)
            config.output.save_metrics = o.get('save_metrics', config.output.save_metrics)
            config.output.export_format = o.get('export_format', config.output.export_format)
            config.output.colormap_visualization = o.get('colormap_visualization', config.output.colormap_visualization)
            config.output.overlay_visualization = o.get('overlay_visualization', config.output.overlay_visualization)
            config.output.overlay_alpha = o.get('overlay_alpha', config.output.overlay_alpha)

        return config

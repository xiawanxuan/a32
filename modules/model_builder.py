import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional, List

from configs.config import Config


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = True
) -> tf.Tensor:
    """
    卷积块：Conv2D + BatchNorm + ReLU + Conv2D + BatchNorm + ReLU

    Args:
        inputs: 输入张量
        filters: 卷积核数量
        kernel_size: 卷积核大小
        dropout_rate: dropout比率
        use_batchnorm: 是否使用批归一化

    Returns:
        输出张量
    """
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(inputs)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def encoder_block(
    inputs: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    编码器块：卷积块 + 最大池化

    Args:
        inputs: 输入张量
        filters: 卷积核数量
        dropout_rate: dropout比率
        use_batchnorm: 是否使用批归一化

    Returns:
        (卷积输出, 池化输出)
    """
    conv = conv_block(inputs, filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def decoder_block(
    inputs: tf.Tensor,
    skip_features: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = True
) -> tf.Tensor:
    """
    解码器块：上采样 + 跳跃连接 + 卷积块

    Args:
        inputs: 输入张量
        skip_features: 跳跃连接的特征图
        filters: 卷积核数量
        dropout_rate: dropout比率
        use_batchnorm: 是否使用批归一化

    Returns:
        输出张量
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = layers.concatenate([x, skip_features], axis=-1)
    x = conv_block(x, filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    return x


def build_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    filters: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    use_batchnorm: bool = True
) -> tf.keras.Model:
    """
    构建U-Net模型

    Args:
        input_shape: 输入形状 (height, width, channels)
        num_classes: 类别数量
        filters: 各层卷积核数量列表
        dropout_rate: dropout比率
        use_batchnorm: 是否使用批归一化

    Returns:
        U-Net模型
    """
    if filters is None:
        filters = [64, 128, 256, 512, 1024]

    inputs = layers.Input(shape=input_shape)

    c1, p1 = encoder_block(inputs, filters[0], dropout_rate=dropout_rate * 0.5, use_batchnorm=use_batchnorm)
    c2, p2 = encoder_block(p1, filters[1], dropout_rate=dropout_rate * 0.5, use_batchnorm=use_batchnorm)
    c3, p3 = encoder_block(p2, filters[2], dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    c4, p4 = encoder_block(p3, filters[3], dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)

    bn = conv_block(p4, filters[4], dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)

    d1 = decoder_block(bn, c4, filters[3], dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    d2 = decoder_block(d1, c3, filters[2], dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    d3 = decoder_block(d2, c2, filters[1], dropout_rate=dropout_rate * 0.5, use_batchnorm=use_batchnorm)
    d4 = decoder_block(d3, c1, filters[0], dropout_rate=dropout_rate * 0.5, use_batchnorm=use_batchnorm)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = models.Model(inputs=[inputs], outputs=[outputs], name='UNet')
    return model


def build_resnet_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    构建基于ResNet骨干的U-Net模型

    Args:
        input_shape: 输入形状
        num_classes: 类别数量
        backbone: 骨干网络名称 ('resnet50' 或 'resnet34')
        pretrained: 是否使用预训练权重
        dropout_rate: dropout比率

    Returns:
        ResNet-UNet模型
    """
    weights = 'imagenet' if pretrained else None

    inputs = layers.Input(shape=input_shape)

    if input_shape[2] == 1:
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
    else:
        x = inputs

    if backbone == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=x,
            input_shape=input_shape if input_shape[2] != 1 else (input_shape[0], input_shape[1], 3)
        )
        layer_names = [
            'input_2' if input_shape[2] == 1 else 'input_1',
            'conv1_relu',
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block6_out',
        ]
        bottleneck_name = 'conv5_block3_out'
    elif backbone == 'resnet34':
        base_model = tf.keras.applications.ResNet101(
            include_top=False,
            weights=weights,
            input_tensor=x,
            input_shape=input_shape if input_shape[2] != 1 else (input_shape[0], input_shape[1], 3)
        )
        layer_names = [
            'input_2' if input_shape[2] == 1 else 'input_1',
            'conv1_relu',
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block23_out',
        ]
        bottleneck_name = 'conv5_block3_out'
    else:
        raise ValueError(f"不支持的骨干网络: {backbone}")

    if pretrained:
        for layer in base_model.layers:
            layer.trainable = True

    skip_features = [base_model.get_layer(name).output for name in layer_names]
    bottleneck = base_model.get_layer(bottleneck_name).output

    filters = [512, 256, 128, 64, 32]

    x = bottleneck
    for i in range(len(skip_features) - 1, -1, -1):
        skip = skip_features[i]
        f = filters[len(filters) - 1 - i] if i < len(filters) else 32
        x = decoder_block(x, skip, f, dropout_rate=dropout_rate)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs], name=f'{backbone}_UNet')
    return model


def build_simple_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    构建简单的CNN分类模型（用于场景分类）

    Args:
        input_shape: 输入形状
        num_classes: 类别数量
        dropout_rate: dropout比率

    Returns:
        简单CNN模型
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='SimpleCNN')

    return model


def build_fcn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    构建FCN（全卷积网络）模型

    Args:
        input_shape: 输入形状
        num_classes: 类别数量
        dropout_rate: dropout比率

    Returns:
        FCN模型
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    pool1 = x

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    pool2 = x

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    pool3 = x

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(num_classes, (1, 1), activation='linear')(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1))(pool3)])

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1))(pool2)])

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.add([x, layers.Conv2D(num_classes, (1, 1))(pool1)])

    x = layers.UpSampling2D(size=(2, 2))(x)
    outputs = layers.Activation('softmax')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs], name='FCN')
    return model


def build_siamese_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int = 2,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    构建孪生U-Net模型（用于变化检测）

    Args:
        input_shape: 输入形状 (单个时相)
        num_classes: 类别数量（变化检测通常为2类：变化/未变化）
        dropout_rate: dropout比率

    Returns:
        孪生U-Net模型
    """
    input_t1 = layers.Input(shape=input_shape, name='input_t1')
    input_t2 = layers.Input(shape=input_shape, name='input_t2')

    def shared_encoder(inputs):
        c1, p1 = encoder_block(inputs, 64, dropout_rate=dropout_rate * 0.5)
        c2, p2 = encoder_block(p1, 128, dropout_rate=dropout_rate * 0.5)
        c3, p3 = encoder_block(p2, 256, dropout_rate=dropout_rate)
        c4, p4 = encoder_block(p3, 512, dropout_rate=dropout_rate)
        bn = conv_block(p4, 1024, dropout_rate=dropout_rate)
        return [c1, c2, c3, c4, bn]

    feat_t1 = shared_encoder(input_t1)
    feat_t2 = shared_encoder(input_t2)

    diff_features = []
    for f1, f2 in zip(feat_t1, feat_t2):
        diff = layers.subtract([f1, f2])
        diff_features.append(diff)

    x = diff_features[-1]
    skip_features = diff_features[:-1][::-1]

    filters = [512, 256, 128, 64]
    for i, skip in enumerate(skip_features):
        x = decoder_block(x, skip, filters[i], dropout_rate=dropout_rate)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=[input_t1, input_t2], outputs=[outputs], name='Siamese_UNet')
    return model


def build_change_detection_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int = 2,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    构建变化检测U-Net模型（输入为两时相拼接影像）

    Args:
        input_shape: 输入形状 (H, W, 2*C)
        num_classes: 类别数量
        dropout_rate: dropout比率

    Returns:
        变化检测U-Net模型
    """
    return build_unet(input_shape, num_classes, dropout_rate=dropout_rate)


class ModelBuilder:
    """
    模型构建器
    """

    def __init__(self, config: Config):
        self.config = config

    def build_model(self) -> tf.keras.Model:
        """
        根据配置构建模型

        Returns:
            构建好的Keras模型
        """
        input_shape = (
            self.config.data.image_size[0],
            self.config.data.image_size[1],
            self.config.data.in_channels
        )
        num_classes = self.config.data.num_classes

        model_type = self.config.model.model_type.lower()

        if model_type == 'unet':
            model = build_unet(
                input_shape=input_shape,
                num_classes=num_classes,
                filters=self.config.model.filters,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type in ['resnet_unet', 'resnet50_unet', 'resnet34_unet']:
            backbone = 'resnet50' if '50' in model_type else 'resnet34'
            model = build_resnet_unet(
                input_shape=input_shape,
                num_classes=num_classes,
                backbone=backbone,
                pretrained=self.config.model.pretrained,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'fcn':
            model = build_fcn(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'simple_cnn':
            model = build_simple_cnn(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'siamese_unet':
            model = build_siamese_unet(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        elif model_type == 'change_detection_unet':
            cd_input_shape = (
                self.config.data.image_size[0],
                self.config.data.image_size[1],
                self.config.data.in_channels * 2
            )
            model = build_change_detection_unet(
                input_shape=cd_input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        return model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        编译模型

        Args:
            model: 待编译的模型

        Returns:
            编译后的模型
        """
        optimizer_type = self.config.train.optimizer.lower()

        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.train.learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.config.train.learning_rate,
                momentum=0.9
            )
        elif optimizer_type == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.train.learning_rate)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")

        loss_type = self.config.train.loss_function.lower()

        if loss_type in ['categorical_crossentropy', 'ce']:
            loss = 'categorical_crossentropy'
        elif loss_type in ['sparse_categorical_crossentropy', 'sparse_ce']:
            loss = 'sparse_categorical_crossentropy'
        elif loss_type in ['dice', 'dice_loss']:
            loss = dice_loss
        elif loss_type in ['jaccard', 'iou_loss']:
            loss = jaccard_loss
        elif loss_type in ['cce_dice', 'combined']:
            loss = combined_loss
        else:
            loss = 'categorical_crossentropy'

        metrics = [
            'accuracy',
            tf.keras.metrics.MeanIoU(
                num_classes=self.config.data.num_classes,
                name='mean_iou'
            )
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return model

    def build_and_compile(self) -> tf.keras.Model:
        """
        构建并编译模型

        Returns:
            编译好的模型
        """
        model = self.build_model()
        model = self.compile_model(model)
        return model


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Dice损失函数

    Args:
        y_true: 真实标签
        y_pred: 预测结果
        smooth: 平滑因子

    Returns:
        Dice损失值
    """
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    return 1 - (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) + smooth
    )


def jaccard_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Jaccard（IoU）损失函数

    Args:
        y_true: 真实标签
        y_pred: 预测结果
        smooth: 平滑因子

    Returns:
        Jaccard损失值
    """
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    union = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) - intersection
    return 1 - (intersection + smooth) / (union + smooth)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    组合损失：交叉熵 + Dice损失

    Args:
        y_true: 真实标签
        y_pred: 预测结果

    Returns:
        组合损失值
    """
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return ce_loss + dice

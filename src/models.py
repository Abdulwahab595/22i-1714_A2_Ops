"""Small scratch CNN and a lightweight ImageNet transfer model."""
import tensorflow as tf
from tensorflow import keras


def augmentation(seed=42):
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
        keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=seed + 2),
        keras.layers.RandomContrast(0.1, seed=seed + 3),
    ], name="training_augmentation")


def build_cnn(image_size=128, lr=1e-3, dropout=0.3, augment=True, seed=42):
    inputs = keras.Input((image_size, image_size, 3), name="rgb_pixels_0_255")
    x = augmentation(seed)(inputs) if augment else inputs
    x = keras.layers.Rescaling(1.0 / 255, name="cnn_normalization")(x)
    for index, filters in enumerate([16, 32, 64], 1):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                name=f"conv_{index}")(x)
        x = keras.layers.BatchNormalization(name=f"normalization_{index}")(x)
        x = keras.layers.Activation("relu", name=f"relu_{index}")(x)
        x = keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
    x = keras.layers.GlobalAveragePooling2D(name="spatial_average")(x)
    x = keras.layers.Dropout(dropout, seed=seed, name="regularization")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = keras.Model(inputs, outputs, name="scratch_cnn")
    compile_model(model, lr)
    return model


def build_transfer(image_size=128, lr=1e-3, dropout=0.3, augment=True, seed=42,
                   weights="imagenet"):
    inputs = keras.Input((image_size, image_size, 3), name="rgb_pixels_0_255")
    x = augmentation(seed)(inputs) if augment else inputs
    # This is exactly MobileNetV2 preprocess_input: [0,255] -> [-1,1].
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)
    backbone = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                             input_shape=(image_size, image_size, 3))
    backbone.trainable = False
    # Keep BN moving statistics fixed even when selected convolution layers are unfrozen.
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="spatial_average")(x)
    x = keras.layers.Dropout(dropout, seed=seed, name="regularization")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = keras.Model(inputs, outputs, name="mobilenet_v2")
    compile_model(model, lr)
    return model


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.AUC(name="auc")])


def unfreeze(model, lr=1e-5, last_layers=None):
    # Augmentation is also a Keras Model and appears before the feature extractor.
    backbone = next(layer for layer in model.layers if isinstance(layer, keras.Model)
                    and any(sub.name.startswith("block_16_") for sub in layer.layers))
    backbone.trainable = True
    candidates = backbone.layers[-last_layers:] if last_layers else [
        layer for layer in backbone.layers if layer.name.startswith("block_16_")
        or layer.name in {"Conv_1", "Conv_1_bn", "out_relu"}
    ]
    selected = {layer.name for layer in candidates}
    for layer in backbone.layers:
        layer.trainable = layer.name in selected and not isinstance(
            layer, keras.layers.BatchNormalization)
    if not any(layer.trainable_weights for layer in backbone.layers
               if isinstance(layer, keras.layers.Conv2D)):
        raise ValueError("Fine-tuning did not select any backbone convolution weights")
    compile_model(model, lr)
    return model


def trainable_layer_names(model):
    names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            names.extend(f"{layer.name}/{sub.name}" for sub in layer.layers
                         if sub.trainable and sub.trainable_weights)
        elif layer.trainable and layer.trainable_weights:
            names.append(layer.name)
    return names


def set_seed(seed=42):
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

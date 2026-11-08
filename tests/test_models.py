from tensorflow import keras

from src.models import build_transfer, trainable_layer_names, unfreeze


def test_fine_tuning_selects_backbone_instead_of_augmentation():
    model = build_transfer(image_size=32, weights=None)
    assert trainable_layer_names(model) == ["ai_probability"]
    unfreeze(model)
    names = trainable_layer_names(model)
    assert any("/block_16_expand" in name for name in names)
    assert any("/Conv_1" in name for name in names)
    assert not any("/block_15_" in name for name in names)
    backbone = next(layer for layer in model.layers if isinstance(layer, keras.Model)
                    and any(sub.name.startswith("block_16_") for sub in layer.layers))
    assert all(not layer.trainable for layer in backbone.layers
               if isinstance(layer, keras.layers.BatchNormalization))

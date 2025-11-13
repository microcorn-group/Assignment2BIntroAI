from typing import Tuple, Optional, Dict, Any
import os
import numpy as np
try:
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras import callbacks  # type: ignore
except Exception:  # TensorFlow not installed or import error
    keras = None
    layers = None
    callbacks = None

CLASS_NAMES = ["01-minor", "02-moderate", "03-severe"]
LABEL_MAP = {"01-minor": "minor", "02-moderate": "moderate", "03-severe": "severe"}


def _augmenter():
    if layers is None:
        raise RuntimeError("TensorFlow is not installed.")
    # Lightweight on-the-fly augmentation to improve generalization
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="augment")


def make_model_simple(input_shape=(128, 128, 3), num_classes=3):
    if keras is None or layers is None:
        raise RuntimeError("TensorFlow is not installed. Install it or run without --incident-image.")
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        _augmenter(),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def make_model_deep(input_shape=(128, 128, 3), num_classes=3):
    """A deeper CNN (inspired by your second CNN notebook)"""
    if keras is None or layers is None:
        raise RuntimeError("TensorFlow is not installed. Install it or run without --incident-image.")
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        _augmenter(),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def make_model_bn(input_shape=(128, 128, 3), num_classes=3):
    """A CNN with BatchNorm + Dropout for better regularization (still from scratch)."""
    if keras is None or layers is None:
        raise RuntimeError("TensorFlow is not installed. Install it or run without --incident-image.")
    inputs = keras.Input(shape=input_shape)
    x = _augmenter()(inputs)
    x = layers.Rescaling(1./255)(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_or_load_model(
    data_dir: str,
    model_path: str,
    arch: str = 'simple',
    epochs: int = 5,
    force: bool = False,
    use_class_weights: bool = True,
):
    """Train a small CNN on images located under data_dir or load existing model.

    Expects directory structure like:
    data_dir/
      training/
        01-minor/ ...images...
        02-moderate/ ...
        03-severe/ ...
      validation/
        01-minor/ ...
        02-moderate/ ...
        03-severe/ ...
    """
    if keras is None:
        raise RuntimeError("TensorFlow not available. Cannot train or load ML model.")
    if os.path.exists(model_path) and not force:
        return keras.models.load_model(model_path)

    img_size = (128, 128)
    batch = 16

    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'training'),
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch,
        shuffle=False,
    )

    # basic caching/prefetch if available
    try:
        train_ds = train_ds.cache().prefetch(buffer_size=16)
        val_ds = val_ds.cache().prefetch(buffer_size=16)
    except Exception:
        pass

    if arch.lower() == 'deep':
        model = make_model_deep(input_shape=img_size + (3,), num_classes=3)
    elif arch.lower() in ('bn', 'batchnorm'):
        model = make_model_bn(input_shape=img_size + (3,), num_classes=3)
    else:
        model = make_model_simple(input_shape=img_size + (3,), num_classes=3)

    # Training callbacks to improve stability and accuracy
    cbs = []
    if callbacks is not None:
        cbs = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=max(2, epochs//3), restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, epochs//4), verbose=1),
            callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ]

    # Optional class weighting for imbalanced data
    class_weight = None
    if use_class_weights:
        # Count files per class in training directory
        counts = []
        total = 0
        for cname in CLASS_NAMES:
            cdir = os.path.join(data_dir, 'training', cname)
            n = sum(1 for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))) if os.path.isdir(cdir) else 0
            counts.append(n)
            total += n
        # Avoid div-by-zero
        if total > 0 and all(c > 0 for c in counts):
            num_classes = len(CLASS_NAMES)
            class_weight = {i: total / (num_classes * counts[i]) for i in range(num_classes)}

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs or None,
        class_weight=class_weight,
    )

    try:
        model.save(model_path)
    except Exception:
        pass
    return model


def predict_severity(model, image_path: str) -> Tuple[str, float]:
    """Return severity label ('minor'|'moderate'|'severe') and confidence."""
    from PIL import Image
    img_size = model.input_shape[1:3]
    img = Image.open(image_path).convert('RGB').resize(img_size)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = LABEL_MAP[CLASS_NAMES[idx]]
    return label, float(probs[idx])


def evaluate_model(model, dataset_dir: str) -> Dict[str, Any]:
    """Evaluate model on validation set, return metrics and confusion matrix."""
    if keras is None:
        raise RuntimeError("TensorFlow not available for evaluation.")
    from sklearn.metrics import confusion_matrix, classification_report
    img_size = model.input_shape[1:3]
    ds = keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, 'validation'),
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=32,
        shuffle=False,
    )
    y_true, y_pred = [], []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(preds.tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(3)))
    report = classification_report(y_true, y_pred, target_names=[LABEL_MAP[c] for c in CLASS_NAMES], output_dict=True)
    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0.0
    return {"accuracy": float(acc), "confusion_matrix": cm.tolist(), "report": report}

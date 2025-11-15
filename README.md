# -------------------------------
# LAND USE LAND COVER CLASSIFICATION USING U-NET
# FULL PYTHON EXECUTABLE CODE
# -------------------------------

import os
import cv2
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# ----------------------------------------------------
# 1. LOAD SENTINEL BANDS & CREATE 4-BAND COMPOSITE
# ----------------------------------------------------
def load_sentinel_image(path_B02, path_B03, path_B04, path_B08):
    with rasterio.open(path_B02) as b2: blue = b2.read(1)
    with rasterio.open(path_B03) as b3: green = b3.read(1)
    with rasterio.open(path_B04) as b4: red = b4.read(1)
    with rasterio.open(path_B08) as b8: nir = b8.read(1)

    image = np.stack([blue, green, red, nir], axis=-1)
    image = (image - image.min()) / (image.max() - image.min())
    return image

# ----------------------------------------------------
# 2. NDVI MASK GENERATION (vegetation)
# ----------------------------------------------------
def generate_ndvi_mask(image):
    red = image[:, :, 2]
    nir = image[:, :, 3]
    ndvi = (nir - red) / (nir + red + 1e-10)
    mask = (ndvi > 0.3).astype(np.uint8)
    return mask

# ----------------------------------------------------
# 3. NDBI MASK GENERATION (buildings)
# ----------------------------------------------------
def generate_ndbi_mask(image):
    swir = image[:, :, 3]   # fallback: use NIR if SWIR unavailable
    green = image[:, :, 1]
    ndbi = (swir - green) / (swir + green + 1e-10)
    mask = (ndbi > 0.2).astype(np.uint8)
    return mask

# ----------------------------------------------------
# 4. PATCH GENERATION FOR TRAINING (256x256)
# ----------------------------------------------------
def create_patches(image, mask, size=256):
    h, w, _ = image.shape
    X, Y = [], []

    for i in range(0, h - size, size):
        for j in range(0, w - size, size):
            img_patch = image[i:i+size, j:j+size]
            mask_patch = mask[i:i+size, j:j+size]

            if img_patch.shape[0]==size and img_patch.shape[1]==size:
                X.append(img_patch)
                Y.append(mask_patch[..., np.newaxis])

    return np.array(X), np.array(Y)

# ----------------------------------------------------
# 5. U-NET MODEL
# ----------------------------------------------------
def unet_model(input_size=(256, 256, 4)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    b = Conv2D(512, 3, activation='relu', padding='same')(p3)
    b = Conv2D(512, 3, activation='relu', padding='same')(b)

    # Decoder
    u3 = UpSampling2D()(b)
    u3 = Concatenate()([u3, c3])
    c4 = Conv2D(256, 3, activation='relu', padding='same')(u3)

    u2 = UpSampling2D()(c4)
    u2 = Concatenate()([u2, c2])
    c5 = Conv2D(128, 3, activation='relu', padding='same')(u2)

    u1 = UpSampling2D()(c5)
    u1 = Concatenate()([u1, c1])
    c6 = Conv2D(64, 3, activation='relu', padding='same')(u1)

    outputs = Conv2D(1, 1, activation='sigmoid')(c6)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------------------------------
# 6. LOAD IMAGES (UPDATE WITH YOUR FILE PATHS)
# ----------------------------------------------------
path_B02 = "B02.jp2"
path_B03 = "B03.jp2"
path_B04 = "B04.jp2"
path_B08 = "B08.jp2"

image = load_sentinel_image(path_B02, path_B03, path_B04, path_B08)

# ----------------------------------------------------
# 7. CREATE MASKS
# ----------------------------------------------------
veg_mask = generate_ndvi_mask(image)
bld_mask = generate_ndbi_mask(image)

# Choose one: vegetation OR buildings
mask = veg_mask        # or bld_mask

# ----------------------------------------------------
# 8. CREATE TRAINING PATCHES
# ----------------------------------------------------
X, Y = create_patches(image, mask, size=256)

print("Total patches:", len(X))

# ----------------------------------------------------
# 9. TRAIN-TEST SPLIT
# ----------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 10. TRAIN U-NET
# ----------------------------------------------------
model = unet_model()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)

# ----------------------------------------------------
# 11. EVALUATION
# ----------------------------------------------------
loss, acc = model.evaluate(X_val, y_val)
print("Validation Accuracy:", acc)

# ----------------------------------------------------
# 12. PREDICTION & VISUALIZATION
# ----------------------------------------------------
pred = model.predict(X_val[:1])[0]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Image"); plt.imshow(X_val[0][:,:,:3])
plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(y_val[0].squeeze(), cmap='gray')
plt.subplot(1,3,3); plt.title("Predicted"); plt.imshow(pred.squeeze(), cmap='gray')
plt.show()

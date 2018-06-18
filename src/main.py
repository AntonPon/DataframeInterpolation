from keras.optimizers import RMSprop
import os
import numpy as np
from src.model.feature_extractor import extract_features, get_image_patch
from src.model.generator import build_model, patches_correlation

model = build_model()

sgd = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=sgd)


def get_images():
    x_train = np.zeros((10, 1, 1024, 1), dtype=np.ndarray)
    y_train = np.zeros((10, 360, 482, 3), dtype=np.ndarray)
    path = '../../dataset'
    files = os.listdir(path)
    for idx, file in enumerate(files[:10]):
        file_path = os.path.join(path, file)
        patch = get_image_patch(file_path)
        #print(get_patch(patch).shape)
        x_train[idx, :, :, :] = get_patch(patch).reshape(1, -1, 1)
        y_train[idx, :, :, :] = patch[1]
    return x_train, y_train


def get_patch(patch):
    left_features = extract_features(patch[0])
    right_features = extract_features(patch[-1])
    return patches_correlation(left_features, right_features)




x_train, y_train = get_images()
print(y_train.shape)
epochs = 3
batch_size = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

print(x_train[0])


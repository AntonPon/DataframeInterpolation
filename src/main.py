from keras.optimizers import RMSprop

from src.model.generator import build_model, patches_correlation

model = build_model()

sgd = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='me', metrics=['accuracy'], optimizer=sgd)
# spots1 = np_utils.to_categorical(train_labels, num_classes=None)

# model.fit(train_images, spots1, batch_size=128, epochs=20)
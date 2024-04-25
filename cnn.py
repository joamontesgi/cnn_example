import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Carga de datos
(train_images, train_labels),(test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar las imágenes - rango entre [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

modelo = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

modelo.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
)

historico = modelo.fit(train_images, train_labels, epochs=2,
                       validation_data=(test_images, test_labels))

test_loss, test_acc = modelo.evaluate(test_images, test_labels, verbose=2)
print('\n Precisión en prueba: ', test_acc)

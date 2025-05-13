import tensorflow as tf
import numpy as np
from PIL import Image

class WoundClassifier:
    def __init__(self, model_path=None, class_labels=['ok', 'suggest', 'urgent'], input_size=(224, 224, 3), num_classes=3):
        self.class_labels = class_labels
        self.input_size = input_size[:2]
        self.num_classes = num_classes

        if model_path:
            print(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Building a new model")
            self.model = self.build_model(input_shape=input_size)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),

            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-5,
            decay_steps=1200,
            decay_rate=0.86
        )
       
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        return model
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size)
        img = np.array(img) / 255.0
        img = tf.cast(tf.expand_dims(img, 0), tf.float32)
        return img
    
    def predict(self, image_path):
        img_tensor = self.preprocess_image(image_path)
        predictions = self.model.predict(img_tensor)
        idx = np.argmax(predictions[0])
        label = self.class_labels[idx]
        confidence = predictions[0][idx]
        return label, confidence
    
    def take_action(self, img_path):
        label, confidence = self.predict(img_path)
        message = {
            'ok': f'No action needed. Confidence: {confidence:.2f}',
            'suggest': f'Please monitor the wound closely. Confidence: {confidence:.2f}',
            'urgent': f'Immediate medical attention required! Confidence: {confidence:.2f}'
        }
        return message[label]
    
    def train(self, dir, save_dir, batch_size=32, epochs=40, validation_split=0.2):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(dir + '/train'),
            validation_split=validation_split,
            subset='training',
            seed=456,
            label_mode='int',
            image_size=self.input_size,
            batch_size=batch_size,
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(dir + '/train'),
            validation_split=validation_split,
            subset='validation',
            seed=456,
            label_mode='int',
            image_size=self.input_size,
            batch_size=batch_size
        )
        
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(dir + '/test'),
            image_size=self.input_size,
            label_mode='int',
            batch_size=batch_size
        )
        
        class_names = train_ds.class_names
        print(class_names)

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1)
        ])

        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.model.fit(train_ds, validation_data=val_ds,epochs=epochs, verbose=1, callbacks=[early_stopping])
        print("Training completed.")
        loss, accuracy = self.model.evaluate(test_ds, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        self.model.save(save_dir)
        print("Model saved.")
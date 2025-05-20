import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import classification_report

@tf.keras.utils.register_keras_serializable()
class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)

@tf.keras.utils.register_keras_serializable()
class ReduceMaxLayer(tf.keras.layers.Layer):
    def __init__(self, axis, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)
class WoundClassifier:
    def __init__(self, model_path=None, class_labels=['ok', 'suggest', 'urgent'], input_size=(224, 224, 3), num_classes=3, learning_rate=1e-4, decay_steps=1200,decay_rate=0.86, dropout_rate=0.2):
        self.class_labels = class_labels
        self.input_size = input_size[:2]
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.dropout_rate = dropout_rate
        
        if model_path:
            print(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss':self.focal_loss(), 'ReduceMeanLayer': ReduceMeanLayer, 'ReduceMaxLayer': ReduceMaxLayer})
        else:
            print("Building a new model")
            self.model = self.build_model(input_shape=input_size)

    def cbam_block(self, input_tensor, ratio=8):
        channel = input_tensor.shape[-1]

        # Channel attention
        channel_avg_pool = ReduceMeanLayer(axis=[1, 2], keepdims=True)(input_tensor)
        channel_max_pool = ReduceMaxLayer(axis=[1, 2], keepdims=True)(input_tensor)

        shared_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(channel // ratio, activation='relu'),
            tf.keras.layers.Dense(channel)
        ])

        channel_avg_out = shared_dense(channel_avg_pool)
        channel_max_out = shared_dense(channel_max_pool)
        channel_attention = tf.keras.layers.Activation('sigmoid')(channel_avg_out + channel_max_out)
        channel_refined = tf.keras.layers.Multiply()([input_tensor, channel_attention])

        # Spatial attention
        spatial_avg_pool = ReduceMeanLayer(axis=-1, keepdims=True)(channel_refined)
        spatial_max_pool = ReduceMaxLayer(axis=-1, keepdims=True)(channel_refined)
        concat = tf.keras.layers.Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])
        spatial_refined = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
        
        return tf.keras.layers.Multiply()([channel_refined, spatial_refined])
    
    
    def focal_loss(self, gamma=2.0, alpha=0.2):
        @tf.keras.utils.register_keras_serializable()
        def loss_fn(y_true, y_pred):
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            pt = tf.reduce_sum(y_true * y_pred, axis=-1)
            return alpha * tf.pow(1. - pt, gamma) * cross_entropy
        return loss_fn
    
    def build_model(self, input_shape):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        inputs = tf.keras.Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = self.cbam_block(x)

        x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = self.cbam_block(x)

        x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        # x = self.cbam_block(x)

        # x = tf.keras.layers.Conv2D(256, 3, padding='same')(inputs)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)

        model = tf.keras.Model(inputs, outputs)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate
        )
       
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(optimizer=optimizer, loss=self.focal_loss(), metrics=['accuracy'])
        return model
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        img = np.array(img) / 255.0
        img = tf.cast(tf.expand_dims(img, axis=0), tf.float32)
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
    
    def train(self, dir, save_dir, batch_size=32, epochs=30, validation_split=0.2):
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
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
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
        
        val_early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,         
            patience=3,         
            min_lr=1e-6
        )

        self.model.fit(train_ds, validation_data=val_ds,epochs=epochs, verbose=1, callbacks=[val_early_stopping, lr_scheduler])
        print("Training completed.")
        loss, accuracy = self.model.evaluate(test_ds, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

        y_true, y_pred = [], []
        for images, labels in test_ds:
            y_true.extend(labels.numpy())
            predictions = self.model.predict(images)
            y_pred.extend(np.argmax(predictions, axis=1))

        report = classification_report(y_true, y_pred, target_names=self.class_labels)
        print(report)
        self.model.save(save_dir)
        print("Model saved.")
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Main directory containing multiple plant folders
main_directory = '.\Plant Leaf Disease Dataset'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)  # 20% of the data will be used for validation

# Custom data generator to iterate through all subdirectories
train_generator = train_datagen.flow_from_directory(
    main_directory,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    subset='training')  # Subset for training data

# Validation generator
valid_generator = train_datagen.flow_from_directory(
    main_directory,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    subset='validation')  # Subset for validation data

# Load ResNet50 pre-trained model without top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze some layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for model training
checkpoint = ModelCheckpoint('new_best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with callbacks
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    callbacks=[checkpoint, early_stopping])

# Save the trained model
model.save('leaf_disease_model.h5')

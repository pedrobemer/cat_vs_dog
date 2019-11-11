from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import backend
# from tensorflow import ConfigProto
# from tensorflow import Session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil

def read_dataset(base_dir, dataset_type, image_size, data_augmentation):
    type_dir = os.path.join(base_dir, dataset_type)
    if (dataset_type == 'train' and data_augmentation == True):
        datagen = data_augmentation()
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(type_dir,
                                            target_size=(image_size[0], 
                                                         image_size[1]), 
                                            batch_size=32,
                                            class_mode='binary') 
    return generator

def create_conv_net():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',\
            input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def data_augmentation():
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True,)
    return datagen

def train_conv_net(model, train_generator, val_generator, lr_rate):
    file_save = 'repelente_vs_copo' + str(lr_rate) +'.h5'
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=lr_rate),
                metrics=['acc'])
    history = model.fit_generator(train_generator, steps_per_epoch=200,
                                  epochs=40, validation_data=val_generator,
                                  validation_steps=80,
                                  callbacks=[
                                             callbacks.EarlyStopping(
                                                monitor="val_loss",
                                                min_delta=1e-7, patience=5,
                                                restore_best_weights=True
                                             ),
                                             callbacks.ModelCheckpoint(
                                                filepath=file_save,
                                                monitor="val_loss", verbose=1,
                                                save_best_only=True
                                             ),
                                            ]
                                 )
    model.save(file_save)


base_dir = 'myDataset2'

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# backend.set_session(Session(config=config))

lr_rate = [0.00001, 0.0001, 0.0005, 0.0010, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
train_gen = read_dataset(base_dir,'train',[256,256], False)
val_gen = read_dataset(base_dir,'validation',[256,256], False)
test_gen = read_dataset(base_dir,'test',[256,256], False)
for i in lr_rate:
    print(i)
    model = create_conv_net()
    train_conv_net(model, train_gen,val_gen, i)

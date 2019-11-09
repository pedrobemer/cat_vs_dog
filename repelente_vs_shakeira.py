from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import backend
from tensorflow import ConfigProto
from tensorflow import Session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil



def create_folders(original_dataset_dir,base_dir):
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir, 'copo')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'repelente')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'copo')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'repelente')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'copo')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'repelente')
    os.mkdir(test_dogs_dir)


    fnames = ['copo.{}.jpg'.format(i) for i in range(1,70)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
        fnames = ['repelente.{}.jpg'.format(i) for i in range(1,70)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['copo.{}.jpg'.format(i) for i in range(71, 95)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['repelente.{}.jpg'.format(i) for i in range(71, 95)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['copo.{}.jpg'.format(i) for i in range(96, 120)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['repelente.{}.jpg'.format(i) for i in range(96, 120)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

def read_train_dataset(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    train_datagen = data_augmentation()
    # train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(256, 256), 
                                                        batch_size=32,
                                                        class_mode='binary') 
    return train_generator

def read_val_dataset(base_dir):
    val_dir = os.path.join(base_dir, 'validation')
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(val_dir,
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='binary')  
    return val_generator

def read_test_dataset(base_dir):
    test_dir = os.path.join(base_dir, 'test')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='binary')
    return test_generator


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

config = ConfigProto()
config.gpu_options.allow_growth = True
backend.set_session(Session(config=config))
base_dir = 'myDataset2'
original_dataset_dir = 'pedro-dataset'
# create_folders(original_dataset_dir,base_dir)
lr_rate = [0.00001, 0.0001, 0.0005, 0.0010, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
train_gen = read_train_dataset(base_dir)
val_gen = read_val_dataset(base_dir)
test_gen = read_test_dataset(base_dir)
for i in lr_rate:
    print(i)
    model = create_conv_net()
    train_conv_net(model, train_gen,val_gen, i)

import os, shutil


def create_folders(original_dataset_dir,base_dir, binary_classes, 
                   train_samples, val_samples, test_samples):
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir, binary_classes[0])
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, binary_classes[1])
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, binary_classes[0])
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, binary_classes[1])
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, binary_classes[0])
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, binary_classes[1])
    os.mkdir(test_dogs_dir)

    filename_class_one = binary_classes[0]+'.{}.jpg'
    filename_class_two = binary_classes[1]+'.{}.jpg'
    print (filename_class_one)
    fnames = [filename_class_one.format(i) for i in range(1,train_samples + 1)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
        fnames = [filename_class_two.format(i) for i in range(1,train_samples + 1)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [filename_class_one.format(i) for i in range(train_samples + 1,
              train_samples + val_samples + 1)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = [filename_class_two.format(i) for i in range(train_samples + 1,
              train_samples + val_samples + 1)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [filename_class_one.format(i) for i in range(train_samples +
              val_samples + 1, train_samples + val_samples + test_samples + 1)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = [filename_class_two.format(i) for i in range(train_samples +
              val_samples + 1, train_samples + val_samples + test_samples + 1)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)



training_samples = 40
validation_samples = 40
test_samples = 40
base_dir = 'myDataset2'
original_dataset_dir = 'pedro-dataset'
create_folders(original_dataset_dir,base_dir,['copo','repelente'],
               training_samples, validation_samples, test_samples)
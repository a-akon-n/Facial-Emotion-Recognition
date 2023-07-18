import os
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator


def split_train_validation(X_train, y_train, validation_size):
    '''
        Method to split the training data into validation and training data, based on a percentage
    '''
    train_data = list(zip(X_train, y_train))
    np.random.shuffle(train_data)
    val_size = int(len(train_data) * validation_size)
    val_data = train_data[:val_size + 1]
    train_data = train_data[val_size:]

    X_train, y_train = zip(*train_data)
    X_val, y_val = zip(*val_data)
    return X_train, y_train, X_val, y_val


def classing_labels(y_data):
    '''
        Transforming the value given to an array where the position corresponding to
        the input will be 1 and the rest will be 0.
    '''
    class_list = []
    for y in y_data:
        x = [0.] * 7
        x[int(y) - 1] = 1.
        class_list.append(x)
    return np.array(class_list)


def read_data():
    image_folder = 'data/aligned'
    annotation_file = 'data/EmoLabel/list_patition_label.txt'

    with open(annotation_file) as f:
        annotations = f.readlines()

    images_train = []
    images_test = []
    labels_train = []
    labels_test = []

    for annotation in annotations:
        image_path, label = annotation.strip().split()
        image_path = image_path.replace('.jpg', '_aligned.jpg')
        image = cv2.imread(os.path.join(
            image_folder, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')/255.0

        if 'test' in image_path:
            images_test.append(image)
            labels_test.append(label)
        else:
            images_train.append(image)
            labels_train.append(label)

    x_train, y_train, x_val, y_val = split_train_validation(
        images_train, labels_train, 0.1)
    x_test = np.array(images_test)
    y_test = np.array(labels_test)

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    return np.array(x_train), classing_labels(y_train), np.array(x_val), classing_labels(y_val), np.array(x_test), classing_labels(y_test)

def load_example():
    predict_dir = "predict"
    files = os.listdir(predict_dir)
    image = cv2.imread(os.path.join(predict_dir, files[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32')/255.0
    return np.array([image])
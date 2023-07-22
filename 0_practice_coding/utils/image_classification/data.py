import os
import random

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


KEYS = [
    'train_img_paths',
    'val_img_paths',
    'train_labels',
    'val_labels',
    'train_labels_idx',
    'val_labels_idx'
]

def train_test_split_by_list(data_dict, test_size):
    returned_data_dict = {}
    class_idx = 0
    for label, img_paths in data_dict.items():
        labels = [label] * len(img_paths)
        train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
            img_paths, labels, test_size=test_size, random_state=42
        )
        train_labels_idx = [class_idx] * len(train_labels)
        val_labels_idx = [class_idx] * len(val_labels)

        returned_data_dict[label] = {}
        for key in KEYS:
            returned_data_dict[label][key] = eval(key)

        class_idx += 1
    return returned_data_dict


def create_train_val_df(data_dict):
    grouped_data_dict = {}
    for key in KEYS:
        grouped_data_dict[key] = []
        for class_data_dict in data_dict.values():
            grouped_data_dict[key] += class_data_dict[key]
    
    train_df = pd.DataFrame(data={
        'image_path': grouped_data_dict['train_img_paths'],
        'label': grouped_data_dict['train_labels'],
        'labels_idx': grouped_data_dict['train_labels_idx']
    })
    val_df = pd.DataFrame(data={
        'image_path': grouped_data_dict['val_img_paths'],
        'label': grouped_data_dict['val_labels'],
        'labels_idx': grouped_data_dict['val_labels_idx']
    })
    return train_df, val_df


def show_random_images(image_paths):
    image_paths = random.choices(image_paths, k=16)
    
    subplot_n_cols, subplot_n_rows = 4, 4

    plt.figure(figsize=(15,12))
    for idx, image_path in enumerate(image_paths):
        plt.subplot(subplot_n_rows, subplot_n_cols, idx+1)
        image = cv2.imread(os.path.join(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)

    plt.tight_layout()
    plt.show()


def create_img_data_generator(df, img_size, batch_size):
    img_data_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='label',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    return img_data_generator

from torch.utils.data.dataset import Dataset
from torchvision.utils import save_image
from prepare_labels import extract_class_labels_list_from_csv, extract_train_labels_distribution
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np  # Numpy like wrapper
from torch import from_numpy
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Parameters:
# address_train_csv_file ='./input_csv/training_3clss.csv'


class MultiLabelDataset(Dataset):
    """Dataset wrapping images and target labels for Multi-Label Image Tagging.
    Arguments:
        A CSV file
        Path to image folder
        Extension of images
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        # read csv file:
        self.df = pd.read_csv(csv_path)

        try:
            assert self.df['image_name'].apply(lambda x: os.path.isfile(img_path + x)).all(), \
                "Some images were not found" + img_path

            # storing the class labels list & its labels distributions
            self.listClassLabels = extract_class_labels_list_from_csv(csv_path)
            self.listLabelsDistributions = extract_train_labels_distribution(csv_path, self.listClassLabels)

            self.mlb = MultiLabelBinarizer(classes=self.listClassLabels)
            self.img_path = img_path
            self.transform = transform

            self.X = self.df['image_name']
            self.y = self.mlb.fit_transform(self.df['tags'].str.split()).astype(np.float32)
        except:
            print("ERROR in reading CSV file!")

    def X(self):
        return self.X

    def __getitem__(self, index):
        try:
            img = Image.open(self.img_path + self.X[index])

            img_name = self.X[index].split('/')[-1]
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

        except Exception as e:
            print(self.img_path + self.X[index])

        label = from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.df.index)

    def get_label_encoder(self):
        return self.mlb

    def get_df(self):
        return self.df

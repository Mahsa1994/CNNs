

from torch.utils.data.dataset import Dataset
# from torchvision import transforms
from torchvision.utils import save_image

from PrepareTrainListLabels import extractClassLabelsListFromCSVFile, extractTrainLabelsDistribution
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
# from torch import np, from_numpy  # Numpy like wrapper
import numpy as np  # Numpy like wrapper
from torch import from_numpy
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Parameters:
# address_train_csv_file ='./input_csv/training_new_clean++_10k_20200113_3clss.csv'


class MultiLabelDataset(Dataset):
    """Dataset wrapping images and target labels for Multi-Label Image Tagging.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        # read csv file:
        self.df = pd.read_csv(csv_path)

        # assert self.df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
        #     "Some images referenced in the CSV file were not found"
        try:

            assert self.df['image_name'].apply(lambda x: os.path.isfile(img_path + x)).all(), \
                "Some images referenced in the CSV file were not found" + img_path

            # storing the class labels list & its labels distributions
            #    self.listClassLabels = extractClassLabelsListFromCSVFile(address_train_csv_file)
            self.listClassLabels = extractClassLabelsListFromCSVFile(csv_path)
            self.listLabelsDistributions = extractTrainLabelsDistribution(csv_path, self.listClassLabels)

            self.mlb = MultiLabelBinarizer(
                classes = self.listClassLabels
            )
            self.img_path = img_path
            # self.img_ext = img_ext
            self.transform = transform

            self.X = self.df['image_name']
            self.y = self.mlb.fit_transform(self.df['tags'].str.split()).astype(np.float32)

        except:
            print("An error Occured")

    def X(self):
        return self.X

    def __getitem__(self, index):

        # img = Image.open(self.img_path + self.X[index] + self.img_ext) ## with extension
        try:
            img = Image.open(self.img_path + self.X[index])

            img_name = self.X[index].split('/')[-1]
            img = img.convert('RGB') # maybe comment later
            if self.transform is not None:
                img = self.transform(img)
                # save_image(img, '/home/nojan/Nojan_project/Codes_V2/pytorch/'+(self.X[index]).split('/')[-1])


        except Exception as e:
            print(self.img_path + self.X[index])

        label = from_numpy(self.y[index])


        return img, label

    def __len__(self):

        return len(self.df.index)

    def getLabelEncoder(self):
        return self.mlb

    def getDF(self):
        return self.df

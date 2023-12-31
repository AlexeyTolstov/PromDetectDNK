import albumentations as A
from os import listdir
from PIL import Image
import numpy as np
from shutil import copyfile

folder_ds = "Datasets/Dataset/"

transform = A.Compose([
    A.RandomBrightness(limit=.3, p=.5),
    A.RGBShift(p=.2),
    A.RandomContrast(limit=0.1, p=.6),
    A.JpegCompression(p=.8),
    A.RandomRain(blur_value=1, p=.6)
])


for file_name in listdir(folder_ds):
    if file_name.lower().endswith((".jpg", ".png")):
        img = np.asarray(Image.open(folder_ds + file_name))
        copyfile(folder_ds + "".join(file_name.split(".")[:-1]) + ".txt", "Datasets/EditedDataset/" + "_" + "".join(file_name.split(".")[:-1]) + ".txt")

        transformed_img = transform(image=img)["image"]
        Image.fromarray(transformed_img).save("Datasets/EditedDataset/" + "_" + file_name)

print("End")
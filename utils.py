import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
def load_meta_data_from_dir(dir_name, file_name ='meta_data'):
    """
    Loads meta data of dataset in given dir_name. It is assumed that all data is in the
    data folder. File name defaults to meta_df.
    :param dir_name:
    :return: df, img_path, meta_path: dataframe containing metadata, path to directory holding the images
    and path to meta data file
    """
        
        
    sample_data_folder = os.path.join('data', dir_name)
    cleaned_meta_data_file = os.path.join(sample_data_folder, file_name)
    images_folder = os.path.join(sample_data_folder, 'images')
    image_captions = pd.read_csv(cleaned_meta_data_file)

    return image_captions, images_folder, cleaned_meta_data_file

#load a few sample images with captions
def display_samples(meta_df, num_samples=5, seed = 0):
    #sample num_sample rows from the dataframe
    samples = meta_df.sample(n=num_samples,random_state = 0)
    for idx,row in samples.iterrows():
        file_name = row['image_path']
        caption = row['caption']
        #get objects and confidence scores
        objects = row['objects'].split(',')
        confidences = row['object_confidences'].split(',')
        
        obj_conf = [str((obj,conf[0:4])) for obj,conf in zip(objects,confidences)]
        num_obj = len(obj_conf)
        obj_str = "\n".join(obj_conf)
        image = Image.open(file_name)
        fig = plt.figure(figsize=(10,(0.3*num_obj)))
        ax = fig.add_subplot(121)
        
                         
        plt.xticks([])
        plt.yticks([])
        ax.imshow(image)        
        ax = fig.add_subplot(122)
        ax.text(0.1, 0.5, obj_str, horizontalalignment='left',verticalalignment='center')
        plt.suptitle(caption)
        plt.xticks([])
        plt.yticks([])
        plt.show()
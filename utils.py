import os
import pandas as pd
def load_meta_data_from_dir(dir_name):
    """
    Loads meta data of dataset in given dir_name. It is assumed that all data is in the
    data folder. Changes column names to meaningful names.
    :param dir_name:
    :return: df, path: dataframe containing metadata and path to directory holding the images
    """
    sample_data_folder = os.path.join('data', dir_name)
    cleaned_meta_data_file = os.path.join(sample_data_folder, 'cleaned_meta_data.csv')
    images_folder = os.path.join(sample_data_folder, 'images')
    image_captions = pd.read_csv(cleaned_meta_data_file, index_col='index')
#     image_captions.rename(
#         columns={'0': 'caption', '1': 'link', '2': 'objects', '3': 'mid', '4': 'object_confidence'},
#         inplace=True)

    return image_captions, images_folder


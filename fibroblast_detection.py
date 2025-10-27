import os
import Pandas as pd
from ultralytics import SAM
import cv2
from fibroblast_images import ImageFibroblast,Cell
from typing import List


def load_image(image:str)->List[ImageFibroblast]:
    img = ImageFibroblast(image)
    return [img]


def load_model()->SAM:
    model = SAM("sam2.1_b.pt")
    return model


def load_folder_images(filder_path:str)->List[ImageFibroblast]:
    file_names = get_filenames(filder_path)
    images = [ImageFibroblast(file_path) for file_path in file_names]
    return images


def start_detect_in_images(images:List[ImageFibroblast],model:SAM):
    for image in images:
        image.find_cell_in_image(model)
    for image in images:
        if image.cell is not None:
            filename = os.path.basename(image.file_path)
            print(f"{filename} no cell found")


def get_picture_of_cells(images:List[ImageFibroblast],save_dir:str)->None:
    for image in images:
        if image.cell is not None:
            image.cell.show_different_cell_areas(save_dir=save_dir)


def get_pixels_area(images:List[ImageFibroblast])->pd.DataFrame:
    pixel_area_df = pd.DataFrame(["file","cell_area","body_area","arms_area"])
    for i,image in enumerate(images):
        if image.cell is not None:
            filename = os.path.basename(image.file_path)
            cell_area = image.cell.get_cell_area()
            body_area = image.cell.get_body_area()
            arms_area = image.cell.get_arms_area()
            pixel_area_df.loc[i] = [filename,cell_area,body_area,arms_area]
    pixel_area_df.set_index("file",inplace=True)
    return pixel_area_df


def get_ratios(images:List[ImageFibroblast])->pd.DataFrame:
    ratio = pd.DataFrame(columns=["file","Invasion_ratio","arms_to_body_ratio","arms_to_cell_ratio"])
    for i,image in enumerate(images):
        if image.cell is not None:
            file = os.path.basename(image.file_path)
            invasion_ratio = image.cell.get_cell_to_body_ratio()
            arms_to_body_ratio = image.cell.get_arms_to_body_ratio()
            arms_to_cell_ratio = image.cell.get_arms_to_cell_ratio()
            ratio.loc[i] = [file,invasion_ratio,arms_to_body_ratio,arms_to_cell_ratio]
    ratio.set_index("file",inplace=True)
    return ratio


def get_filenames(dir_path:str=None)->list:
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            files.append(os.path.join(dir_path,file))
    return files


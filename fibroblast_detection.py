import os
import pandas as pd
from ultralytics import SAM
import cv2
from fibroblast_images import ImageFibroblast,Cell
from typing import List
from pathlib import Path

def get_filenames(in_dir:Path):
    extensions = [".jpg",".jpeg",".png"]
    return sorted([p for p in in_dir.iterdir() if p.suffix.lower() in extensions])


def load_image(image:str,sobel_tresh:float)->List[ImageFibroblast]:
    img = ImageFibroblast(image,sobel_threshold=sobel_tresh)
    return [img]


def load_model(device:str)->SAM:
    model = SAM("sam2.1_b.pt",device=device)
    return model


def load_folder_images(folder_path:str,sobel_thresh:float)->List[ImageFibroblast]:
    file_names = get_filenames(folder_path)
    images = [ImageFibroblast(file_path,sobel_threshold=sobel_thresh) for file_path in file_names]
    return images


def delete_scale(images:List[ImageFibroblast]):
    for image in images:
        image.delete_scale()


def preprocess_images(images:List[ImageFibroblast]):
    for image in images:
        image.preprocess_images()

    
def start_detect_in_images(images:List[ImageFibroblast],model:SAM):
    for image in images:
        image.find_cell_in_image(model)
    for image in images:
        if image.cell is not None:
            filename = os.path.basename(image.file_path)
            print(f"{filename} no cell found")


def get_pixels_area(images:List[ImageFibroblast],out_dir:Path)->pd.DataFrame:
    pixel_area_df = pd.DataFrame(["file","cell_area","body_area","arms_area"])
    for i,image in enumerate(images):
        if image.cell is not None:
            filename = os.path.basename(image.file_path)
            cell_area = image.cell.get_cell_area()
            body_area = image.cell.get_body_area()
            arms_area = image.cell.get_arms_area()
            pixel_area_df.loc[i] = [filename,cell_area,body_area,arms_area]
    pixel_area_df.set_index("file",inplace=True)
    if out_dir is not None:
        pixel_area_df.to_csv(out_dir / "_stats.csv",sep=";")
    return pixel_area_df


def get_ratios(images:List[ImageFibroblast],out_dir:Path)->pd.DataFrame:
    ratio = pd.DataFrame(columns=["file","Invasion_ratio","arms_to_body_ratio","arms_to_cell_ratio"])
    for i,image in enumerate(images):
        if image.cell is not None:
            file = os.path.basename(image.file_path)
            invasion_ratio = image.cell.get_cell_to_body_ratio()
            arms_to_body_ratio = image.cell.get_arms_to_body_ratio()
            arms_to_cell_ratio = image.cell.get_arms_to_cell_ratio()
            ratio.loc[i] = [file,invasion_ratio,arms_to_body_ratio,arms_to_cell_ratio]
    ratio.set_index("file",inplace=True)
    if out_dir is not None:
        ratio.to_csv(out_dir / "_stats.csv",sep=";")
    return ratio


def combine_ratios_areas(ratio:pd.DataFrame,area:pd.DataFrame,out_dir:Path):
    combined = pd.concat([ratio, area], axis=1)
    if out_dir is not None:
        combined.to_csv(out_dir / "_stats.csv",sep=";")
    return combined
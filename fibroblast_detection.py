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


def load_image(image:Path,**params)->ImageFibroblast:
    img = ImageFibroblast(image, **params)
    return img


def load_model()->SAM:
    model = SAM("sam2.1_b.pt")
    return model


def preprocess_image(image:ImageFibroblast):
    image.preprocess_image()

    
def start_detect_in_image(image:ImageFibroblast,model:SAM):
    image.find_cell_in_image(model)
    if image.cell is None:
        print(f"{image.filename} no cell found")


def create_stats_file(out_dir:Path,in_dir:Path):
        header = "file;cell_area;body_area;arms_area;invasion_ratio;arms_to_body_ratio;arms_to_cell_ratio\n"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir/f"{in_dir.name}_stats.csv","w") as f:
            f.write(header)

def write_to_stats_file(image:ImageFibroblast,out_dir:Path,in_dir:Path)->pd.DataFrame:
    
    if image.cell is not None:
        file = os.path.basename(image.file_path)
        cell_area = image.cell.get_cell_area()
        body_area = image.cell.get_body_area()
        arms_area = image.cell.get_arms_area()
        invasion_ratio = image.cell.get_cell_to_body_ratio()
        arms_to_body_ratio = image.cell.get_arms_to_body_ratio()
        arms_to_cell_ratio = image.cell.get_arms_to_cell_ratio()

        with open(out_dir/f"{in_dir.name}_stats.csv","a") as f:
            f.write(f"{file};{cell_area};{body_area};{arms_area};{invasion_ratio};{arms_to_body_ratio};{arms_to_cell_ratio}\n"
            )
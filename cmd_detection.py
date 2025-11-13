
import argparse
from pathlib import Path
import fibroblast_detection as fd

def main():
    parser = argparse.ArgumentParser(description="command-line tool for fibroblast detection")

    # Define arguments
    #Folders
    parser.add_argument("-i","--input_folder", help="Path to the input folder",default=None)
    parser.add_argument("-ii","--input_image", help="Input image (with path)",default=None)
    parser.add_argument("-o", "--output", help="output dir name", default="out")
    
    #Parameters
    parser.add_argument("-ts","--treshold_sobel", help="Threshold for sobel",default=0.1)
    parser.add_argument("-tmin","--threshold_min_radius", help="Minimal fibroblast radius",default=130)
    parser.add_argument("-tmax","--threshold_max_size", help="Maximal relative size fibroblast",default=0.65)
    parser.add_argument("-ti","--threshold_irregular", help="Threshhold to not be perfect ellipse", type=float, default=0.7)
    
    parser.add_argument("-fi","--filtering_mean_interference", help="Filtering: interference mean dark value find more if higher", type=int, default=120)
    parser.add_argument("-fc","--filtering_min_n_cluster", help="Filtering: remove small noise cluster of this size", type=int, default=100)
    parser.add_argument("-ff","--filtering_iterations_filling", help="Filtering: Iterations filling holes in noise mask", type=int, default=10)

    #Plotting
    parser.add_argument("-pa","--plot_areas", help="Plot areas of detected cells", action="store_true",default=False)
    parser.add_argument("-ps","--plot_segmentation_masks", help="Plot areas found segmentation masks", action="store_true",default=False)
    parser.add_argument("-pe","--plot_cell_and_edge_masks", help="Plot areas of detected cell and edges", action="store_true",default=False)
    parser.add_argument("-pt","--plot_threshold", help="Plot threshhold", action="store_true",default=False)
    parser.add_argument("-pp","--plot_preprocessing", help="Plot preprocessing found artefacts", action="store_true",default=False)

    #general   
    parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-d", "--device", help="Do you have a GPU for the SAM model", default="cpu",choices=["cpu","cuda"])
    
    # Parse args
    args = parser.parse_args()
    
    #logging minifunction
    def vprint(x:str):
        if args.verbose:
            print(x)
    
    #insert logic for tool
    print("Started fibroblast detection")
    out_dir = Path(args.output)
    in_dir = Path(args.input_folder)
    input_image = Path(args.input_image)

    out_dir.mkdir(parents=True, exist_ok=True)

    if in_dir is None and input_image is None:
        raise Exception("Need input either image or folder")
    if in_dir is not None and input_image is not None:
        raise Exception("Need only one input either image or folder")

    vprint("Created stats file")
    fd.create_stats_file(out_dir,in_dir)

    if input_image is not None:
        file_names = [input_image]
    if in_dir is not None:
        file_names = fd.get_filenames(in_dir)

    vprint(f"Found: {len(file_names)} files")

    vprint("Load SAM Model")    
    model = fd.load_model(args.device)

    vprint("Start processing pictures")
    for i,image_path in enumerate(file_names):
        
        image = fd.load_image(image_path,args.sobel_threshold)
        vprint(f"{i}/{len(file_names)}: Loaded image {image.name} ")
        vprint("Start Preprocessing")
        fd.preprocess_image(image,args.filtering_mean_interference,args.min_n_cluster,args.iterations_filling)
        vprint("Start image segmentation")
        fd.start_detect_in_image(image,model)
        vprint(f"Wrote stats to {out_dir/in_dir.name}_stats.csv")
        fd.write_to_stats_file(image,out_dir,in_dir)

        vprint("Start plotting")
        if args.plot_preprocessing:
            image.plot_preprocessing(save_dir=out_dir)
        if args.plot_threshold:
            image.plot_threshold_histogram(save_dir=out_dir)
        if args.plot_segmentation_masks:
            image.plot_segmentation_masks(save_dir=out_dir)
        if args.plot_cell_and_edge_masks:
            image.plot_cell_with_edges(save_dir=out_dir)
        if args.plot_areas:
            image.plot_cell_areas(saver_dir=out_dir)
        vprint(f"Finished {image.name}")

    print("Finished all pictures")
    

if __name__ == "__main__":
    main()
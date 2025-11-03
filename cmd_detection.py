
import argparse
from pathlib import Path
import fibroblast_detection as fd

def main():
    parser = argparse.ArgumentParser(description="command-line tool for fibroblast detection")

    # Define arguments
    parser.add_argument("-i","--input_folder", help="Path to the input folder",default=None)
    parser.add_argument("-ii","--input_image", help="Input image (with path)",default=None)

    parser.add_argument("-ts","--treshold_sobel", help="Threshold for sobel",default=0.1)
    parser.add_argument("-tmin","--threshold_min_radius", help="Minimal fibroblast radius",default=130)
    parser.add_argument("-tmax","--threshold_max_size", help="Maximal relative size fibroblast",default=0.65)
    parser.add_argument("-ti","--threshold_irregular", help="Threshhold to not be perfect ellipse", type=float, default=0.7)
    
    parser.add_argument("-pa","--plot_areas", help="Plot areas of detected cells", action="store_true",default=False)
    parser.add_argument("-ps","--plot_segmentation_masks", help="Plot areas found segmentation masks", action="store_true",default=False)
    parser.add_argument("-pe","--plot_cell_and_edge_masks", help="Plot areas of detected cell and edges", action="store_true",default=False)
    parser.add_argument("-pt","--plot_threshold", help="Plot threshhold", action="store_true",default=False)
    parser.add_argument("-pp","--plot_preprocessing", help="Plot preprocessing found artefacts", action="store_true",default=False)


    parser.add_argument("-r","--ratios", help="Return ratio csv", action="store_true",default=True)
    parser.add_argument("-a", "--areas", help="Return areas csv", action="store_true", default=True)
    
    parser.add_argument("-o", "--output", help="output dir name", default="out")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-d", "--device", help="Do you have a GPU for the SAM model", default="cpu")
    
    # Parse args
    args = parser.parse_args()
    
    #logging minifunction
    def verbose_print(x:str):
        if args.verbose:
            print(x)
    
    #insert logic for tool
    print("Started fibroblast detection")
    out_dir = Path(args.output)
    in_dir = Path(args.input_folder)
    input_image = Path(args.input_image)


    if in_dir is None and input_image is None:
        raise Exception("Need input either image or folder")
    if in_dir is not None and input_image is not None:
        raise Exception("Need only one input either image or folder")

    #load images
    if input_image is not None:
        images = fd.load_image(input_image,args.threshold_sobel)
    if in_dir is not None:
        images = fd.load_folder_images(in_dir)
    verbose_print("Loaded Images")


    fd.preprocess_images(images)
    verbose_print("Preprocessed Images removed bubbles and meshes")

    #load model
    model = fd.load_model(device=args.device)
    verbose_print("Loaded Model")

    #start detection
    verbose_print("Start cellbody detection")

    fd.start_detect_in_images(images,model)
    verbose_print("Detected cellbodies in images")

    #plot
    if args.plot_preprocessing:
        [image.plot_preprocessing(save_dir=out_dir) for image in images]

    if args.plot_threshold:
        [image.plot_threshold_histogram(save_dir=out_dir) for image in images]

    if args.plot_masks:
        [image.plot_segmentation_masks for image in images]

    if args.plot_cell_and_edge_masks:
        [image.plot_cell_with_edges(save_dir=out_dir) for image in images]

    if args.plot_areas:
        [image.plot_cell_areas(saver_dir=out_dir) for image in images]
    verbose_print("Plotting finished")

    #return ratios and area
    combine =  args.ratios and args.area
    if args.ratios:
        ratios = fd.get_ratios(images,out_dir=out_dir)
    if args.area:
        areas  = fd.get_pixels_area(images,out_dir=out_dir)
    if combine:
        fd.combine_ratios_areas(ratios,areas,out_dir=out_dir)
    verbose_print("Calcualted Stats")

    print("Finished fibroblast detection successfully")


    
    
    



    

if __name__ == "__main__":
    main()
import argparse

def main():
    parser = argparse.ArgumentParser(description="Example command-line tool")

    # Define arguments
    parser.add_argument("-i","input_folder", help="Path to the input folder",default=None)
    parser.add_argument("-ii","input_image", help="Input image (with path)",default=None)

    parser.add_argument("-ts","treshhold_sobel", help="Threshold for sobel",default=0.1)
    parser.add_argument("-tmin","threshold_min_radius", help="Minimal fibroblast radius",default=130)
    parser.add_argument("-tmax","threshold_max_size", help="Maximal relative size fibroblast",default=0.65)
    parser.add_argument("-ti","threshold_irregular", help="Threshhold to not be perfect ellipse", type=float, default=0.7)
    
    parser.add_argument("-pa","plot_areas", help="Plot areas of detected cells", action="store_true",default=False)
    parser.add_argument("-pm","plot_masks", help="Plot areas of generated masks", action="store_true",default=False)
    parser.add_argument("-pt","plot_threshold", help="Plot threshhold", action="store_true",default=False)

    parser.add_argument("-r","ratios", help="Return ratio csv", action="store_true",default=True)
    parser.add_argument("-a", "--areas", help="Return areas csv", action="store_true", default=True)
    
    parser.add_argument("-o", "--output", help="output dir name", default="out")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Parse args
    args = parser.parse_args()
    #args_input 
    
    #insert logic for tool


    

if __name__ == "__main__":
    main()
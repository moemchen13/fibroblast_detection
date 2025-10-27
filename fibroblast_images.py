import  os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel


class ImageFibroblast:
    def __init__(self, file_path,sobel_threshold:float=0.1,multiple_cells_possible:bool=False):
        self.file_path = file_path
        self.sobel_threshold = sobel_threshold
        self.image = cv2.imread(file_path)
        self.cell = None
        self.multiple_cells_possible = multiple_cells_possible
    

    def detect_possible_arms(self):
        sobel_img = sobel(self.img)
        counts,bins = np.histogram(sobel_img.ravel(),bins=300)
        mask = (sobel_img>self.sobel_threshold)
        return mask, counts,bins

    def plot_threshold_histogram(self,threshold,save_dir:str=None):
        mask, counts, bins = self.detect_possible_arms(threshold)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Distribution of edges deteted by sobel filter")
        ax1.bar(bins[:-1], counts, width=np.diff(bins), align='edge', color='gray', edgecolor='black')
        ax1.vlines(threshold,ymin=0,ymax=counts.max(),color="r",label="threshold")
        ax1.legend()
        ax2.imshow(self.img,cmap="gray")
        ax2.imshow(self.mask, cmap='Reds', alpha=0.5)
        ax2.set_title("Mask of Edges")
        if save_dir is not None:
            filename = os.path.basename(self.file_path)
            name, ext = os.path.splitext(filename)
            plt.savefig(os.path.join(save_dir,f"{name}_Areas{ext}"))
        plt.show()
    

    def fuse_cell_with_arms(self,cell_mask,possible_arms_mask):
        cell = (cell_mask > 0).astype(np.uint8)
        edges = (possible_arms_mask > 0).astype(np.uint8)
        # dilate cell to catch touching edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        cell_d = cv2.dilate(cell, kernel)
        # connected components on edges
        num, labels = cv2.connectedComponents(edges, connectivity=8)
        # labels touching cell
        touch_ids = np.unique(labels[cell_d.astype(bool)])
        keep = np.isin(labels, touch_ids[touch_ids != 0]).astype(np.uint8)
        # optional thicken
        keep = cv2.dilate(keep, kernel)
        fused_cell = np.clip(cell + keep, 0, 1).astype(np.uint8)
        return fused_cell
    

    def detect_arms_for_cell(self,cell_bodies,possible_arms):
        whole_cells = []
        for cell in cell_bodies:
            cell_with_arms = self.fuse_cell_with_arms(cell,possible_arms)
            whole_cells.append(cell_with_arms)
        return whole_cells


    def remove_overlapping_arms(self,arms,body):
        arms = arms-body
        arms[arms < 0] = 0
        return arms
    

    def segment_cell_in_image(self,segmentation_model,threshold_low=130,threshold_high=0.65,tolerance_irregular_shape=0.7):
        results = segmentation_model(self.file_path)  # image
        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0]<=1:
            print(f"Didn't found cellbody for image: {self.file_path}")
        else:
            #select biggest mask for that is not whole image
            mask_sums = masks.sum(axis=(1,2))
            w,b = cv2.imread(self.file_path,cv2.IMREAD_GRAYSCALE).shape
            thresholds_max_mask = mask_sums< (w*b*threshold_high) #nicht größer als 2/3
            if not np.any(thresholds_max_mask):
                print(f"only found one big mask no cell")
            else:
                idx = np.argmax(mask_sums * thresholds_max_mask)
            if mask_sums[idx]<(threshold_low**2 * np.pi):
                print(f"found mask to small beneath {threshold_low} pixels")

            #ensure the mask is not just an artefact by checking if it is a spheroid
            ys,xs = np.where(masks[idx,:,:]==1)
            height = ys.max() -ys.min()
            width = xs.max() - xs.min()
            area_elipse = int((height/2)*(width/2)*np.pi)
            area_elipse *= tolerance_irregular_shape  #tolerance for irregular shapes
            area_mask = masks[idx,:,:].sum()
            if not area_mask < area_elipse:
                #check spheroid to not match artefacts
                print(f"found mask is not ellipsoide therefore no cell")
            else:
                return masks[idx,:,:]
            

    def find_cell_in_image(self,model):
        body_mask = self.segment_cell_in_image(model)
        if body_mask is not None:
            edges_mask,_,_ = self.detect_possible_arms()
            cell_mask = self.fuse_cell_with_arms(cell_mask,edges_mask)
            arms_mask = self.remove_overlapping_arms(cell_mask,body_mask)
            self.cell = Cell(self.file_path,cell=cell_mask,body=body_mask,arms=arms_mask,edges=edges_mask)


class Cell:
    def __init__(self, file_path:str,cell:np.ndarray=None, body:np.ndarray=None, 
                arms:np.ndarray=None,edges:np.ndarray=None, threshold_edges:float=0.1,
                threshold_low:float=130, threshold_high:float=0.65, tolerance_irregular_shape:float=0.7):
        self.file_path = file_path
        self.cell = cell
        self.body = body
        self.arms = arms
        self.edges = edges
        self.threshold_edges = threshold_edges
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.tolerance_irregular_shape = tolerance_irregular_shape


    def get_area(self,mask):
        if mask is not None:
            return np.sum(mask > 0)
        else:
            return 0
        

    def get_arms_area(self):
        return self.get_area(self.arms)
    

    def get_body_area(self):
        return self.get_area(self.body)
    

    def get_cell_area(self):
        return self.get_area(self.cell)
    

    @staticmethod
    def show_area(file_path,mask):
        plt.imshow(cv2.imread(file_path), cmap='gray')
        plt.imshow(mask, cmap='jet',alpha=0.5)
        plt.show()


    def show_cell(self):
        if self.cell is not None:
            self.show_area(self.cell)
        else:
            print("No body mask available.")
    

    def show_arms(self):
        if self.arms is not None:
            self.show_area(self.arms)
        else:
            print("No arms mask available.")    


    def show_body(self):
        if self.body is not None:
            return np.sum(self.body > 0)
        else:
            print("No body mask available.")
            return 0
        

    def show_different_cell_areas(self,save_dir:str=None):
        plt.imshow(cv2.imread(self.file_path), cmap='gray')
        combined_mask = np.zeros_like(self.cell)
        if self.body is not None:
            combined_mask[self.body > 0] = 1
        if self.arms is not None:
            combined_mask[self.arms > 0] = 2
        plt.imshow(combined_mask, cmap='jet', alpha=0.5)
        if save_dir is not None:
            filename = os.path.basename(self.file_path)
            name, ext = os.path.splitext(filename)
            plt.savefig(os.path.join(save_dir,f"{name}_Areas{ext}"))
        plt.show()


    def show_areas_with_detected_edges(self,save_dir:str=None):
        img = cv2.imread(self.file_path,cv2.IMREAD_GRAYSCALE)
        fig,ax = plt.subplots(1, 4,figsize=(12, 4))
        for axis in ax:
            axis.imshow(img,"grey")
            axis.axis("off")

        ax[0].set_title("Original Image")

        ax[1].imshow(self.body, cmap='jet', alpha=0.5)
        ax[1].set_title("Cell Body")

        ax[2].imshow(self.edges, cmap='jet', alpha=0.5)
        ax[2].set_title(f"Edges (t:{self.threshold:.2f})")

        ax[3].imshow(self.cell, cmap='jet', alpha=0.5)
        ax[3].set_title("Final Cell")
        if save_dir is not None:
            filename = os.path.basename(self.file_path)
            name, ext = os.path.splitext(filename)
            plt.savefig(os.path.join(save_dir,f"{name}_Edges{ext}"))
        plt.show()
    

    @staticmethod
    def get_ratio(maskA:np.ndarray,maskB:np.ndarray):
        areaA = maskA.sum()
        areaB = maskB.sum()
        if areaB > 0:
            return areaA / areaB
        else:
            return float('inf')
        

    def get_arms_to_body_ratio(self):
        return self.get_ratio(self.arms, self.body)
    
    def get_arms_to_cell_ratio(self):
        return self.get_ratio(self.arms, self.cell)
    
    def get_cell_to_body_ratio(self):
        return self.get_ratio(self.cell, self.body)
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from sklearn.cluster import MiniBatchKMeans
from skimage.morphology import remove_small_objects, convex_hull_image, remove_small_holes, opening, closing, disk
from skimage import filters, measure
from scipy import ndimage 
from PIL import Image


class ImageFibroblast:
    def __init__(self, file_path:Path,k:int=3,
                 sobel_cell_threshold:float=1,interference_pixel_value:int=120,
                 disturbing_mesh:int=40000,min_regions_mash:int=4,
                 min_mash_hole_size:int=625,min_crop_ratio:float=0.85,
                 sobel_arms_threshold:float= 1,background_ratio:float=0.7,
                 scale_at_bottom:bool=True,r:int=50,smooth_radius:int=5,
                 round_radius:int=5):
        self.file_path = file_path
        self.filename = file_path.name
        self.name = self.filename.split(".")[0]
        self.original_image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        self.image = self.original_image
        self.cell = None
        self.segmentation_masks = None
        self.kmeans_labels = None
        self.k=k
        self.background_ratio=background_ratio
        #find Cell in image params
        self.sobel_cell_threshold=sobel_cell_threshold
        self.interference_pixel_value=interference_pixel_value
        self.distrubing_mesh = disturbing_mesh
        self.min_regions_mash = min_regions_mash
        self.min_mash_hole_size = min_mash_hole_size #bigger as than 25**2 quader
        self.sobel_arms_threshold = sobel_arms_threshold
        self.min_crop_ratio = min_crop_ratio
        self.scale_at_bottom =scale_at_bottom
        #classical cell body creation params
        self.r = r
        self.smooth_radius=smooth_radius
        self.round_radius=round_radius
    

    def kmeans(self,image:np.ndarray=None,k:int=None):
        if k is None:
            k=self.k
        if image is None:
            image = self.original_image

        channels = 1
        if len(image.shape)>2:
            channels = image.shape[2]
        pixels = image.reshape(-1, channels)
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, max_iter=100, random_state=0)
        labels_kmeans = kmeans.fit_predict(pixels).reshape(image.shape[:2])
        centers = kmeans.cluster_centers_.ravel() 
        order = np.argsort(centers)  
        mapping = np.empty_like(order)
        mapping[order] = np.arange(k*channels)
        labels_ranked = mapping[labels_kmeans]
        return labels_ranked
        
    
    def is_interference(self)->int|None:
        return (self.original_image[self.kmeans_labels==0].mean())<self.interference_pixel_value
        

    @staticmethod
    def mash_has_enclosed_regions(mash,n_regions,cluster_size):
        inv_mash = ~mash
        mask,n_labels = ndimage.label(inv_mash)
        sizes = ndimage.sum(inv_mash,mask,range(n_labels+1))
        #check if recurring enclosed regions
        return (sizes>cluster_size).sum()>(n_regions +1)
    


    def get_mash_from_interference(self):
        interference = self.kmeans_labels==0
        labeled_mask, n_labels = ndimage.label(interference)
        sizes = ndimage.sum(interference, labeled_mask, range(n_labels + 1))
        largest_label = np.argmax(sizes)
        largest_component = labeled_mask == largest_label
        if self.mash_has_enclosed_regions(largest_component,self.min_regions_mash,self.min_mash_hole_size):
            if largest_component.sum() > self.distrubing_mesh:
                return largest_component
        else: 
            return None
        

    def crop_horizontal(self,hull):
        horizontal = hull.sum(axis=0)/hull.shape[0]
        mask = horizontal>self.min_crop_ratio
        left = np.argmax(~mask)
        right = np.argmax(mask)
        if left > 0 or right >0:
            if left>right:
                self.image = self.image[:,left:]
                hull = hull[:,left:]
            else:
                self.image = self.image[:,:-right]
                hull = hull[:,:-right]
        return hull


    def crop_vertical(self,hull):
        vertical = hull.sum(axis=1)/hull.shape[1]
        mask = vertical > self.min_crop_ratio
        top = np.argmax(~mask)
        bottom = np.argmax(~mask[::-1])
        if bottom > 0 or top >0:
            if top>bottom:
                self.image = self.image[top:,:]
            else:
                bottom = hull.shape[1]-bottom
                self.image = self.image[:-bottom,:]


    def remove_mash(self,mash):
        #first removes as much as possible from one side then till threshold
        convex_hull = convex_hull_image(mash)
        H,W = convex_hull.shape
        cropped_convex_hull = self.crop_horizontal(convex_hull)
        self.crop_vertical(cropped_convex_hull)


    def preprocess_image(self):
        if self.scale_at_bottom:
            self.original_image = self.original_image[:-24,:]

        self.image = self.original_image
        self. kmeans_labels = self.kmeans()
        if self.is_interference():
            mash = self.get_mash_from_interference()
            if mash is not None:
                self.remove_mash(mash)
        else:
            self.image = self.original_image


    def detect_possible_arms(self):
        sobel_img = sobel(self.image)
        counts,bins = np.histogram(sobel_img.ravel(),bins=300)
        sobel_threshold = self.sobel_arms_threshold*sobel_img.std() + sobel_img.mean()
        mask = (sobel_img>sobel_threshold)
        return mask, counts,bins


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
    

    def segment_image(self,segmentation_model,tmp_dir=Path("./tmp")):
        #tmp_dir.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(self.image)
        image_path = tmp_dir/(self.name + ".jpeg")
        im.save(image_path)
        results = segmentation_model(image_path)  # image
        self.segmentation_masks = results[0].masks.data.cpu().numpy()
        image_path.unlink()
        return self.segmentation_masks
    

    def is_background(self, mask):
        H,W = mask.shape
        return mask.sum()/(H*W)> self.background_ratio
    

    def find_point_in_cell_by_sobel(self):
        #Finds point cell by weighted centroid in sobel image
        sobel_image = filters.sobel(self.image)
        #normalize image
        sobel_image = (sobel_image-sobel_image.min())/(sobel_image.max()-sobel_image.min() +1e-12)
        threshold = self.sobel_cell_threshold* sobel_image.std() + sobel_image.mean()
        filtered_image = (sobel_image>=threshold)

        filtered_image = remove_small_objects(filtered_image,min_size=100)
        lab = measure.label(filtered_image)
        props = measure.regionprops(lab,intensity_image=self.image)
        if not props:
            raise ValueError("No structure detected after threshold.")
        best = max(props,key=lambda r: (r.intensity_image[r.filled_image].sum() 
                                        if r.intensity_image is not None else r.area))
        yc,xc = best.weighted_centroid
        return int(yc),int(xc)


    @staticmethod
    def circular_struct(radius):
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        return (X**2 + Y**2) <= radius**2


    def connect_nearby_components(self,mask, radius=3):
        selem = self.circular_struct(radius)
        # closing = dilation followed by erosion
        closed = ndimage.binary_closing(mask, structure=selem)
        return closed


    def remove_small_and_fill(self,mask):
        # 1. Remove thin structures
        thin_removed = ndimage.binary_opening(mask, structure=np.ones((3, 3), bool))

        # 2. Remove tiny components
        labels = measure.label(thin_removed, connectivity=2)
        counts = np.bincount(labels.ravel())
        big = counts >= 20   # choose your min_size
        big[0] = False
        cleaned = big[labels]

        # 3. Connect nearby blobs
        connected = self.connect_nearby_components(cleaned, radius=5)
        return connected
    
    @staticmethod
    def get_connected_components(mask):
        labels = measure.label(mask, connectivity=2) 
        regions = measure.regionprops(labels)
        return regions


    def create_mask_for_cell_pixel(self,cell_point):
        shape = self.image.shape
        mask = np.zeros(shape)
        H,W = shape
        bbox = (max(cell_point[0]-self.r,0),min(cell_point[0]+self.r,H),
                max(cell_point[1]-self.r,0),min(cell_point[1]+self.r,W))
        mask[bbox[0]:bbox[1],bbox[2]:bbox[3]] = 1
        return mask.astype(bool)


    @staticmethod
    def create_region_mask(region,shape):
        mask = np.zeros(shape)
        cor = region.coords
        mask[cor[:,0],cor[:,1]] = 1
        return mask.astype(bool)

    
    def select_biggest_overlap(self,mask_cell_pixel,regions):
        contained_region = None
        n_pixels_contained = 0

        for region in regions:
            region_mask = self.create_region_mask(region,mask_cell_pixel.shape)
            overlap = region_mask & mask_cell_pixel
            if overlap.sum()>n_pixels_contained:
                n_pixels_contained=overlap.sum()
                contained_region = region_mask
        return contained_region


    @staticmethod
    def fill_internal_holes(mask):
        return ndimage.binary_fill_holes(mask)


    def smooth_and_round(self,mask):
        """
        smooth_radius: small smoother edges, removed tiny bumps
        round_radius:  large round corners, filled small concavities
        necessary to make region more cell like
        """
        selem_smooth = disk(self.smooth_radius)
        selem_round  = disk(self.round_radius)

        # Remove tiny spikes / jaggies
        removed_spikes_mask = opening(mask, selem_smooth)
        # Round the object and fill small gaps
        clean_mask = closing(removed_spikes_mask, selem_round)
        return clean_mask



    def SAM_found_cell_mask(self,cell_point):
        masks = self.segmentation_masks
        if masks is not None or len(masks)!=0:
            for mask in masks:
                if not self.is_background(mask):
                    if mask[cell_point[0],cell_point[1]]:
                        return mask
        print(f"SAM model didn't found cellbody, image: {self.filename}")


    def create_classical_mask(self,cell_point):
        #1. kmeans on sobel and grey image to detect foreground and background
        img = self.image
        sobel_img = sobel(img)
        sobel_grey = np.stack((img[:,:],sobel_img[:,:]),axis=-1)
        kmeans_clusters = self.kmeans(image=sobel_grey,k=2)
        foreground = kmeans_clusters==0

        #2. Prepare foreground
        foreground = self.remove_small_and_fill(foreground)
        regions = self.get_connected_components(foreground)
        #3. Biggest overlap between quadrat around cell_pixel and region
        mask_cell_pixel = self.create_mask_for_cell_pixel(cell_point)
        best_region_mask = self.select_biggest_overlap(mask_cell_pixel,regions)

        if best_region_mask is not None:
            #4. prepare mask to make it more cell shaped and whole
            best_region_mask = self.fill_internal_holes(best_region_mask)
            best_region_mask = self.smooth_and_round(best_region_mask)
            #pieces can be broken of while cleaning
            connected_components_best_region = self.get_connected_components(best_region_mask)
            biggest_region = max(connected_components_best_region, key=lambda x:x.area)
            body_mask = self.create_region_mask(biggest_region,best_region_mask.shape)
            return body_mask
        
        print(f"Classical method didn't found body either, image: {self.filename}")
        return None



    def find_cell_in_image(self,model,masks=None)->None:
        if masks is None:
            self.segment_image(model)
        else:
            self.segmentation_masks=masks
        
        cell_point = self.find_point_in_cell_by_sobel()
        body_mask = self.SAM_found_cell_mask(cell_point)
        if body_mask is None:
            body_mask = self.create_classical_mask(cell_point)
        if body_mask is not None:
            edges_mask,_,_ = self.detect_possible_arms()
            cell_mask = self.fuse_cell_with_arms(body_mask,edges_mask)
            arms_mask = self.remove_overlapping_arms(cell_mask,body_mask)
            self.cell = Cell(self.file_path,cell=cell_mask,body=body_mask,arms=arms_mask,edges=edges_mask)

    ###Plotting funcs###
    def plot_segmentation_masks(self,save_dir:Path)->None:
        if self.segmentation_masks is not None:
            masks = self.segmentation_masks
            n_masks = len(masks)
            if n_masks==1:
                plt.imshow(masks[0],"grey")
            else:
                fig,axs = plt.subplots(nrows=1,ncols=n_masks)
                for ax,mask in zip(axs,masks):
                    ax.imshow(mask,"grey")
                    ax.axis("off")
            if save_dir is not None:
                plt.savefig(save_dir/ f"{self.name}_segmentation_masks.png")
            plt.show()

    
    def plot_threshold_histogram(self,save_dir:Path=None):
        mask, counts, bins = self.detect_possible_arms()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Distribution of edges deteted by sobel filter")
        ax1.bar(bins[:-1], counts, width=np.diff(bins), align='edge', color='gray', edgecolor='black')
        threshold = sobel(self.image).mean()+ sobel(self.image).std() * self.sobel_arms_threshold
        ax1.vlines(threshold,ymin=0,ymax=counts.max(),color="r",label="threshold")
        ax1.legend()
        ax2.imshow(self.image,cmap="gray")
        ax2.imshow(mask, cmap='Reds', alpha=0.5)
        ax2.set_title("Mask of Edges")
        if save_dir is not None:
            plt.savefig(save_dir / f"{self.name}_threshold.png")
        plt.show()


    def plot_cell_with_edges(self,save_dir:Path):
        if self.cell is not None:
            self.cell.show_areas_with_detected_edges(self.image,self.name,save_dir)


    def plot_cell_areas(self,save_dir:Path):
        if self.cell is not None:
            self.cell.show_different_cell_areas(self.image,self.name,save_dir)

    
    def plot_preprocessing(self,save_dir:Path):
        fig, axs = plt.subplots(1, 3)
        axs[0].set_title("Original Image")
        axs[0].imshow(self.original_image,cmap="gray")

        axs[1].set_title(f"Kmeans k={self.k}")
        axs[1].imshow(self.kmeans_labels, cmap='jet', alpha=0.5)


        axs[2].set_title("Processed Image")
        axs[2].imshow(self.image,cmap="gray")

        for ax in axs:
            ax.axis("off")
        if save_dir is not None:
            plt.savefig(save_dir / f"{self.name}_preprocessing.png")
        plt.show()

    

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
        

    def show_different_cell_areas(self,img,name:str,save_dir:Path=None):
        plt.imshow(img, cmap='gray')
        combined_mask = np.zeros_like(self.cell)
        if self.body is not None:
            combined_mask[self.body > 0] = 1
        if self.arms is not None:
            combined_mask[self.arms > 0] = 2
        plt.imshow(combined_mask, cmap='jet', alpha=0.5)
        if save_dir is not None:
            plt.savefig(save_dir/ f"{name}_Areas.png")
        plt.show()


    def show_areas_with_detected_edges(self,img,name:str,save_dir:Path=None):
        
        fig,ax = plt.subplots(1, 4,figsize=(12, 4))
        for axis in ax:
            axis.imshow(img,"grey")
            axis.axis("off")

        ax[0].set_title("Original Image")

        ax[1].imshow(self.body, cmap='jet', alpha=0.5)
        ax[1].set_title("Cell Body")

        ax[2].imshow(self.edges, cmap='jet', alpha=0.5)
        ax[2].set_title(f"Edges")

        ax[3].imshow(self.cell, cmap='jet', alpha=0.5)
        ax[3].set_title("Final Cell")
        if save_dir is not None:
            plt.savefig(save_dir/f"{name}_Edges.png")
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
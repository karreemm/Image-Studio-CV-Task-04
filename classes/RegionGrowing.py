import numpy as np

class RegionGrowingSegmentation:
    def segment(self, image, seed_points, threshold):
        """
        Perform region growing segmentation on the image using the given seed points.
        
        Args:
            image: Input image (grayscale)
            seed_points: List of seed points (x, y) to start the region growing
            threshold: Intensity threshold for region growing
            
        Returns:
            List of segmented regions as sets of pixel coordinates
        """
        self.threshold = threshold
        self.segmented_regions = np.zeros_like(image)
        self.visited = np.zeros_like(image)
        
        for seed in seed_points:
            self.visited[seed[0], seed[1]] = 1
            region = self.grow(image, seed)
            self.segmented_regions += region
            
        return self.segmented_regions
    
    def grow(self, image, seed_point):
        """
        Perform region growing on the image starting from the seed point.
        """
        region_pixels = [seed_point]
        segmented_region = np.zeros_like(image)
        segmented_region[seed_point[0], seed_point[1]] = 255
        
        for pixel in region_pixels:
            x, y = pixel
            
            # Check the 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (x + dx, y + dy)
                    
                    if (0 <= neighbor[0] < image.shape[0] and
                        0 <= neighbor[1] < image.shape[1] and self.visited[neighbor[0], neighbor[1]] == 0):
                        self.visited[neighbor[0], neighbor[1]] = 1
                        if abs(int(image[x, y]) - int(image[neighbor[0], neighbor[1]])) <= self.threshold:
                            segmented_region[neighbor[0], neighbor[1]] = 255
                            region_pixels.append(neighbor)
            
        
        return segmented_region
    
    
    
    
    
import cv2
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt

# Global list to store seed points
seed_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selected point: ({y}, {x})")
        seed_points.append((y, x))  # Notice (row, col) => (y, x)

def main():
    # Load the image
    image_path = 'D:\\Computer Vision\\Image-Studio-CV-Task-04\\Final Test Data\\cameraman.bmp'  # <-- Change this to your image path
    image_bgr = cv2.imread(image_path)  # Read normally (in BGR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Luv)
    L, U, V = cv2.split(image)
    image = L  
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Show image and set mouse callback
    cv2.namedWindow('Select Seed Points')
    cv2.setMouseCallback('Select Seed Points', mouse_callback)

    print("Click to select seed points. Press 'q' to finish.")

    while True:
        temp_image = image.copy()
        # Draw selected points
        for point in seed_points:
            cv2.circle(temp_image, (point[1], point[0]), 3, 200, -1)
        
        cv2.imshow('Select Seed Points', temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to break
            break

    cv2.destroyAllWindows()

    # Set a threshold value
    threshold = 25

    # Perform segmentation
    segmenter = RegionGrowingSegmentation()
    segmented_image = segmenter.segment(image, seed_points, threshold)

    # Show the original and segmented images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()


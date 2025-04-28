import numpy as np
from scipy.spatial import distance_matrix
import cv2

class AgglomerativeClustering:

    def agglomerative_clustering_fast(self,image, n_clusters):
        resized_image = cv2.resize(image, (20, 20)) 
        height, width = resized_image.shape
        pixels = []

        for i in range(height):
            for j in range(width):
                pixels.append([resized_image[i, j], i, j])  # intensity + position

        pixels = np.array(pixels)
        n_samples = len(pixels)
        clusters = [[i] for i in range(n_samples)]  # Each point is its own cluster initially
        
        # Initial distance matrix
        D = distance_matrix(pixels, pixels)
        np.fill_diagonal(D, np.inf)  # Don't merge a point with itself
        
        while len(clusters) > n_clusters:
            # Find the closest two clusters
            i, j = np.unravel_index(np.argmin(D), D.shape)
            
            # Merge clusters
            clusters[i] = clusters[i] + clusters[j]
            clusters.pop(j)
            
            # Update the distance matrix
            # Rule: single linkage -> min distance between any two points of the two clusters
            
            # Remove the j-th row and column
            D = np.delete(D, j, axis=0)
            D = np.delete(D, j, axis=1)
            
            # Update distances of the new merged cluster (row i)
            for k in range(len(D)):
                if k != i:
                    # Find minimum distance between cluster i and cluster k
                    dists = [np.linalg.norm(pixels[p1] - pixels[p2]) for p1 in clusters[i] for p2 in clusters[k]]
                    D[i, k] = D[k, i] = np.min(dists)
            
            D[i, i] = np.inf  # No self-merge
            

        segmented_img = np.zeros((height, width), dtype=np.uint8)

        for cluster in clusters:
            intensity = np.mean([pixels[idx][0] for idx in cluster])  # Average intensity
            for idx in cluster:
                i, j = int(pixels[idx][1]), int(pixels[idx][2])
                segmented_img[i, j] = intensity
        
        return segmented_img


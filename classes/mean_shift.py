import numpy as np
import matplotlib.pyplot as plt

def mean_shift_segmentation(data, bandwidth=1.0, threshold=0.01):
    """
    Mean Shift Segmentation Algorithm
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data points of shape (n_samples, n_features)
    bandwidth : float
        Bandwidth parameter for the window
    threshold : float
        Convergence threshold
        
    Returns:
    --------
    cluster_centers : numpy.ndarray
        Centers of clusters
    labels : numpy.ndarray
        Cluster labels for each point
    """
    # Initialize variables
    n_points = data.shape[0]
    visited = np.zeros(n_points, dtype=bool)
    labels = -1 * np.ones(n_points, dtype=int)
    cluster_centers = []
    
    # Main algorithm loop
    cluster_idx = 0
    
    while np.sum(visited) < n_points:
        # Select a random unvisited point
        unvisited_indices = np.where(~visited)[0]
        if len(unvisited_indices) == 0:
            break
            
        # Choose the first unvisited point
        current_idx = unvisited_indices[0]
        current_mean = data[current_idx].copy()
        tracked_points = []
        
        # Mean Shift procedure
        while True:
            # Find points within bandwidth
            distances = np.sqrt(np.sum((data - current_mean) ** 2, axis=1))
            in_bandwidth_indices = np.where(distances <= bandwidth)[0]
            tracked_points = list(set(tracked_points + list(in_bandwidth_indices)))
            
            # Calculate new mean
            if len(in_bandwidth_indices) > 0:
                new_mean = np.mean(data[in_bandwidth_indices], axis=0)
            else:
                break
            
            # Check convergence
            mean_shift = np.sqrt(np.sum((new_mean - current_mean) ** 2))
            if mean_shift < threshold:
                # Check for merge with existing clusters
                merged = False
                for i, center in enumerate(cluster_centers):
                    if np.sqrt(np.sum((center - new_mean) ** 2)) < 0.5 * bandwidth:
                        # Merge clusters
                        merged_center = 0.5 * (center + new_mean)
                        cluster_centers[i] = merged_center
                        # Assign all tracked points to the existing cluster
                        for tracked_idx in tracked_points:
                            if not visited[tracked_idx]:
                                labels[tracked_idx] = i
                                visited[tracked_idx] = True
                        merged = True
                        break
                
                if not merged:
                    # Create new cluster
                    cluster_centers.append(new_mean)
                    # Assign all tracked points to the new cluster
                    for tracked_idx in tracked_points:
                        if not visited[tracked_idx]:
                            labels[tracked_idx] = cluster_idx
                            visited[tracked_idx] = True
                    cluster_idx += 1
                break
            
            current_mean = new_mean
    
    return np.array(cluster_centers), labels

def apply_mean_shift_segmentation_to_image(image, bandwidth=20, threshold=1):
    """
    Apply mean shift segmentation to an image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image of shape (height, width, channels)
    bandwidth : float
        Bandwidth parameter for mean shift
    threshold : float
        Convergence threshold
        
    Returns:
    --------
    segmented_image : numpy.ndarray
        Segmented image where each pixel is replaced by its cluster center
    """
    # Reshape the image to a 2D array of pixels
    height, width, channels = image.shape
    pixels = image.reshape((-1, channels)).astype(np.float32)
    
    # Apply mean shift segmentation
    centers, labels = mean_shift_segmentation(pixels, bandwidth=bandwidth, threshold=threshold)
    
    # Create segmented image
    segmented_pixels = centers[labels]
    segmented_image = segmented_pixels.reshape((height, width, channels)).astype(np.uint8)
    
    # unique_colors = [
    #         (255, 0, 0),     
    #         (0, 255, 0),     
    #         (0, 0, 255),     
    #         (255, 255, 0),   
    #         (255, 0, 255),   
    #         (0, 255, 255),   
    #         (255, 128, 0),   
    #         (128, 0, 255),   
    #         (0, 128, 0),     
    #         (128, 128, 128)  
    #     ]
    
    # # Create an image where each segment has a unique color
    # colored_labels = np.zeros((height * width, 3), dtype=np.uint8)
    # for i, color in enumerate(unique_colors):
    #     colored_labels[labels == i] = color
    
    # colored_segmented_image = colored_labels.reshape((height, width, 3))

    return segmented_image


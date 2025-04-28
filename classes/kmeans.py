import numpy as np
import random


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def initialize_centroids(data, k):
    """Initialize k centroids randomly from the data points."""
    # Get the number of data points
    n_samples = data.shape[0]
    
    # Generate k random indices
    random_indices = random.sample(range(n_samples), k)
    
    # Return the data points at these indices as initial centroids
    return data[random_indices]


def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    # Array to store cluster assignments for each point
    labels = np.zeros(data.shape[0], dtype=np.int32)
    
    # For each data point
    for i, point in enumerate(data):
        # Calculate distance to each centroid
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        
        # Find the index of the closest centroid
        closest_centroid = np.argmin(distances)
        
        # Assign this point to the closest centroid
        labels[i] = closest_centroid
    
    return labels


def update_centroids(data, labels, k):
    """Update centroids based on current cluster assignments."""
    # Initialize new centroids
    new_centroids = np.zeros((k, data.shape[1]))
    counts = np.zeros(k)
    
    # For each point
    for i, point in enumerate(data):
        # Add this point to its assigned centroid's sum
        cluster_idx = labels[i]
        new_centroids[cluster_idx] += point
        counts[cluster_idx] += 1
    
    # Calculate mean for each cluster
    for i in range(k):
        if counts[i] > 0:
            new_centroids[i] = new_centroids[i] / counts[i]
        else:
            # If a cluster is empty, initialize with a random point
            new_centroids[i] = data[random.randint(0, data.shape[0]-1)]
    
    return new_centroids



def apply_specific_colors(height , width, labels, specific_colors):
    """
    Apply specific intensity values to each cluster (for 1-channel image).
    
    Parameters:
    - image: original single-channel image (height, width)
    - labels: cluster labels for each pixel (1D array)
    - specific_colors: list of intensities to apply to each cluster (0-255)
    
    Returns:
    - colored_image: image with specific intensities applied to each cluster
    """
    k = len(specific_colors)
    
    # Convert specific_colors to numpy array if it's not already
    specific_colors = np.array(specific_colors)

    # Apply specific intensities based on labels
    colored_pixels = np.array([specific_colors[label % k] for label in labels])
    # Reshape back to original image dimensions
    colored_image = colored_pixels.reshape(height, width,3)

    # Ensure the values are in valid image range [0, 255]
    colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)

    return colored_image


def kmeans_image(image, k, seed_points,specific_colors=None, max_iterations=20, tolerance=0.001):
    """
    Apply K-means clustering to an image and color each cluster with specific colors.
    
    Parameters:
    - image: numpy array of shape (height, width, channels)
    - k: number of clusters (colors)
    - specific_colors: list of specific colors to apply to each cluster
                      if None, the centroid colors will be used
    - max_iterations: maximum number of iterations to run
    - tolerance: convergence threshold for centroid movement
    
    Returns:
    - colored_image: image with each cluster colored with a specific color
    - centroids: final cluster centroids
    - iterations: number of iterations run
    """
    #Generate Random Colors
    specific_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255]   # Magenta
        ]    
    height, width = image.shape
    # Reshape L channel to a 1D array of pixels
    pixels = image.reshape(-1, 1) 
    
    # Initialize centroids
    if (len(seed_points) == 0):
        centroids = initialize_centroids(pixels, k)
    else:
        centroids = pixels[seed_points]
    
    # Keep track of previous centroids for convergence check
    prev_centroids = np.zeros_like(centroids)
    
    # Initialize variables
    iterations = 0
    converged = False
    
    while not converged and iterations < max_iterations:
        # Assign points to clusters
        labels = assign_clusters(pixels, centroids)
        
        # Store current centroids for convergence check
        prev_centroids = centroids.copy()
        
        # Update centroids based on current assignments
        centroids = update_centroids(pixels, labels, k)
        
        # Check for convergence
        centroid_movement = np.sum([euclidean_distance(prev_centroids[i], centroids[i]) for i in range(k)])
        converged = centroid_movement < tolerance
        
        # Increment iteration counter
        iterations += 1
        
        print(f"Iteration {iterations}, centroid movement: {centroid_movement:.6f}")
    
    # If specific colors are provided, use them, otherwise use centroids
    if specific_colors is not None:
        segmented_image = apply_specific_colors(height ,width, labels, specific_colors)
    else:
        # Replace each pixel with its cluster's centroid L value
        clustered_pixels = np.array([centroids[label][0] for label in labels])
        segmented_image = clustered_pixels.reshape(height, width)
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)
    
    return segmented_image


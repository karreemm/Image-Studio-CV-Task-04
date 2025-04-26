import numpy as np
import cv2
from scipy import ndimage

class SpectralThresholding:
    def __init__(self, mode):
        self.mode = mode.lower()
        if self.mode not in ['global', 'local']:
            raise ValueError("Mode must be 'global' or 'local'")
        
    def global_otsu_multithreshold( image, 
                                    num_classes = 3,
                                    smoothing_sigma = 1.0):
        
        if len(image.shape) > 2:
            raise ValueError("Input image must be grayscale")
        
        # Compute global histogram
        hist_global = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_global = hist_global / np.sum(hist_global)  # Normalize histogram
        
        # Apply Gaussian smoothing to histogram
        smoothed_hist = ndimage.gaussian_filter1d(hist_global, smoothing_sigma)
        pixel_range = np.arange(256)
        
        # Global mean
        mg = np.sum(pixel_range * smoothed_hist)
        
        # For 2 classes
        if num_classes == 2:
            max_variance = 0
            optimal_threshold = 0
            
            for t in range(256):
                p1 = np.sum(smoothed_hist[:t+1])
                p2 = np.sum(smoothed_hist[t+1:])
                
                if p1 == 0 or p2 == 0:
                    continue
                
                m1 = np.sum(pixel_range[:t+1] * smoothed_hist[:t+1]) / p1
                m2 = np.sum(pixel_range[t+1:] * smoothed_hist[t+1:]) / p2
                
                variance = p1 * p2 * ((m1 - m2) ** 2)
                
                if variance > max_variance:
                    max_variance = variance
                    optimal_threshold = t
            
            return [optimal_threshold], max_variance
        
        # For multiple classes
        max_variance = 0
        thresholds = []
        
        def calculate_variance(t_list):
            t_list = sorted(t_list)
            variance = 0
            
            ranges = [(0, t_list[0])] + \
                    [(t_list[i], t_list[i+1]) for i in range(len(t_list)-1)] + \
                    [(t_list[-1], 256)]
                    
            for start, end in ranges:
                p_k = np.sum(smoothed_hist[start:end])
                if p_k == 0:
                    continue
                m_k = np.sum(pixel_range[start:end] * smoothed_hist[start:end]) / p_k
                variance += p_k * ((m_k - mg) ** 2)
                
            return variance
        
        # Initialize with peaks from smoothed histogram
        peaks = []
        for i in range(1, 255):
            if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
                prominence = min(
                    smoothed_hist[i] - np.min(smoothed_hist[max(0, i-10):i]),
                    smoothed_hist[i] - np.min(smoothed_hist[i+1:min(256, i+11)])
                )
                if prominence > 0.01 * np.max(smoothed_hist):
                    peaks.append(i)
        
        # Use top num_classes-1 peaks as initial thresholds
        if len(peaks) >= num_classes-1:
            peak_heights = [smoothed_hist[p] for p in peaks]
            indices = np.argsort(peak_heights)[-(num_classes-1):]
            thresholds = sorted([peaks[i] for i in indices])
        else:
            # Fallback to evenly spaced thresholds
            step = 256 // num_classes
            thresholds = [i * step for i in range(1, num_classes)]
        
        # Refine thresholds
        for _ in range(5):
            for i in range(len(thresholds)):
                best_t = thresholds[i]
                for t in range(max(0, best_t-10), min(255, best_t+10)):
                    temp_thresholds = thresholds.copy()
                    temp_thresholds[i] = t
                    variance = calculate_variance(temp_thresholds)
                    if variance > max_variance:
                        max_variance = variance
                        best_t = t
                thresholds[i] = best_t
        
        return sorted(thresholds), max_variance
    

    def local_otsu_multithreshold(  image, 
                                    num_classes = 3,
                                    window_size= 3,  
                                    smoothing_sigma = 1.0,
                                    min_prominence = 0.1):

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        if window_size < 3:
            raise ValueError("Window size must be at least 3")
            
        result = np.zeros_like(image, dtype=np.float32)
        pad_size = window_size // 2
        
        # Pad image for window analysis
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                cv2.BORDER_REFLECT)
        
        # Process each pixel with its local window
        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                # Extract local window
                window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
                
                # Compute histogram for window
                hist = cv2.calcHist([window], [0], None, [256], [0, 256]).flatten()
                smoothed_hist = ndimage.gaussian_filter1d(hist, smoothing_sigma)
                
                # Find peaks
                peaks = []
                peak_heights = []
                
                for k in range(1, 255):
                    if smoothed_hist[k] > smoothed_hist[k-1] and smoothed_hist[k] > smoothed_hist[k+1]:
                        prominence = min(
                            smoothed_hist[k] - np.min(smoothed_hist[max(0, k-10):k]),
                            smoothed_hist[k] - np.min(smoothed_hist[k+1:min(256, k+11)])
                        )
                        if prominence > min_prominence * np.max(smoothed_hist):
                            peaks.append(k)
                            peak_heights.append(smoothed_hist[k])
                
                # Select strongest peaks and find thresholds
                if len(peaks) >= num_classes:
                    indices = np.argsort(peak_heights)[-num_classes:]
                    peaks = sorted([peaks[idx] for idx in indices])
                    
                    # Find valleys between peaks
                    thresholds = []
                    for k in range(len(peaks)-1):
                        start, end = peaks[k], peaks[k+1]
                        valley_idx = start + np.argmin(smoothed_hist[start:end+1])
                        thresholds.append(valley_idx)
                    
                    # Assign class label based on thresholds
                    pixel_value = image[i-pad_size, j-pad_size]
                    label = 0
                    for k, t in enumerate(thresholds):
                        if pixel_value > t:
                            label = k + 1
                    result[i-pad_size, j-pad_size] = label
                else:
                    # If not enough peaks, use mean value
                    result[i-pad_size, j-pad_size] = np.mean(window)
        
        return result

    def segment_image(image, thresholds):

        segmented = np.zeros_like(image)
        thresholds = sorted(thresholds)
        
        for i, t in enumerate(thresholds):
            if i == 0:
                segmented[image <= t] = 0
            else:
                segmented[(image > thresholds[i-1]) & (image <= t)] = i
        
        segmented[image > thresholds[-1]] = len(thresholds)
        return segmented
    
    def apply_thresholding(self, image, num_classes=3, window_size=3, smoothing_sigma=1.0, min_prominence=0.1):
        if self.mode == 'global':
            thresholds, _ = self.global_otsu_multithreshold(image, num_classes, smoothing_sigma)
            result = self.segment_image(image, thresholds)
        elif self.mode == 'local':
            result = self.local_otsu_multithreshold(image, num_classes, window_size, smoothing_sigma, min_prominence)
        
        return result


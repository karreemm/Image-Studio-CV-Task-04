import cv2
import numpy as np
from image import Image
from PyQt5.QtWidgets import QApplication
import sys

class OptimalThresholding:
    """
    A class to perform optimal thresholding on grayscale images using either global or local methods.

    Optimal thresholding iteratively computes a threshold to segment an image into background
    and foreground by minimizing the difference between their mean intensities. This class supports:
    - Global mode: Applies thresholding to the entire image.
    - Local mode: Divides the image into blocks, computes thresholds, and interpolates them.

    Attributes:
        thresholding_mode (str): The mode of thresholding, either "global" or "local".
        corner_size (int): Size of the corner patches (in pixels) used for initial background estimation.
        block_size (int): Size of blocks (in pixels) for local thresholding.
        step_size (int): Step size for overlapping blocks in local thresholding (default is block_size / 2).
        convergence_threshold (float): Tolerance for convergence in the iterative thresholding process.
    """
    corner_size = 1  # Using 1x1 to represent single corner pixels
    block_size = 64  # Increased to reduce sensitivity to local variations
    step_size = 32   # 50% overlap
    convergence_threshold = 1e-3

    def __init__(self, thresholding_mode):
        """
        Initialize the OptimalThresholding class.

        Args:
            thresholding_mode (str): Either "global" or "local" to specify the thresholding method.
        
        Raises:
            ValueError: If thresholding_mode is not "global" or "local".
        """
        mode = thresholding_mode.lower()
        if mode not in ["global", "local"]:
            raise ValueError("thresholding_mode must be 'global' or 'local'")
        self.thresholding_mode = mode

    def _estimate_initial_background(self, image):
        """
        Estimate initial background pixels by sampling the four corner pixels of the image.

        Args:
            image (np.ndarray): Grayscale image as a 2D NumPy array.

        Returns:
            np.ndarray: Flattened array of the four corner pixel values.
        """
        rows, cols = image.shape
        corners = np.array([
            image[0, 0],              # Top-left corner pixel
            image[0, cols-1],         # Top-right corner pixel
            image[rows-1, 0],         # Bottom-left corner pixel
            image[rows-1, cols-1]     # Bottom-right corner pixel
        ])
        return corners

    def _compute_optimal_threshold(self, image):
        """
        Compute the optimal threshold for a grayscale image using iterative mean updates.

        This implements the algorithm from the lecture slide:
        1. Start with an initial guess (four corners as background, rest as object).
        2. Compute mean gray-levels of background and object.
        3. Update the threshold as the average of the means.
        4. Repeat until the threshold converges.

        Args:
            image (np.ndarray): Grayscale image as a 2D NumPy array.

        Returns:
            float: The optimal threshold value.
        """
        # Step 1: Initial guess for background and object pixels
        background_pixels = self._estimate_initial_background(image)
        # Object pixels are all pixels not in the corners
        all_pixels = image.flatten()
        object_pixels = all_pixels[~np.isin(all_pixels, background_pixels)]

        # Step 2-4: Iterative threshold computation
        previous_threshold = 0.0
        while True:
            # Compute mean gray-levels of background and object
            background_mean = (np.mean(background_pixels) if len(background_pixels) > 0 else 0)
            object_mean = (np.mean(object_pixels) if len(object_pixels) > 0 else 0)

            # Update threshold as the average of the means
            current_threshold = (background_mean + object_mean) / 2

            # Check for convergence
            if abs(current_threshold - previous_threshold) < self.convergence_threshold:
                break

            # Update pixel classifications based on the new threshold
            background_pixels = image[image <= current_threshold].flatten()
            object_pixels = image[image > current_threshold].flatten()
            previous_threshold = current_threshold

        return current_threshold

    def _apply_global_thresholding(self, image):
        """
        Apply global optimal thresholding to the entire image.

        Args:
            image (np.ndarray): Grayscale image as a 2D NumPy array.

        Returns:
            np.ndarray: Binary image (0 for background, 255 for foreground).

        Raises:
            ValueError: If the image is not a 2D grayscale array.
        """
        # Ensure the image is 2D (grayscale)
        if len(image.shape) != 2:
            raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

        # Compute the optimal threshold for the entire image
        threshold = self._compute_optimal_threshold(image)

        # Create binary image: pixels above threshold are foreground (255), others are background (0)
        binary_image = np.zeros_like(image, dtype=np.uint8)
        binary_image[image > threshold] = 255
        return binary_image

    def _apply_local_thresholding(self, image):
        """
        Apply local optimal thresholding by computing thresholds for overlapping blocks and interpolating them.

        This method:
        1. Computes thresholds for overlapping blocks.
        2. Interpolates the thresholds across the image using weighted averaging.
        3. Applies the interpolated thresholds to the image.
        4. Uses morphological opening to remove small speckles.

        Args:
            image (np.ndarray): Grayscale image as a 2D NumPy array.

        Returns:
            np.ndarray: Binary image (0 for background, 255 for foreground).

        Raises:
            ValueError: If the image is not a 2D grayscale array.
        """
        # Ensure the image is 2D (grayscale)
        if len(image.shape) != 2:
            raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

        rows, cols = image.shape
        # Arrays to accumulate thresholds and weights for each pixel
        accumulated_thresholds = np.zeros_like(image, dtype=np.float32)
        weight_sum = np.zeros_like(image, dtype=np.float32)

        # Process overlapping blocks to compute thresholds
        for row in range(0, rows - self.block_size + 1, self.step_size):
            for col in range(0, cols - self.block_size + 1, self.step_size):
                # Extract the block
                row_end = min(row + self.block_size, rows)
                col_end = min(col + self.block_size, cols)
                block = image[row:row_end, col:col_end]

                if block.size == 0:  # Skip empty blocks
                    continue

                # Compute the optimal threshold for the block
                threshold = self._compute_optimal_threshold(block)

                # Compute weights based on distance from block center
                block_rows, block_cols = block.shape
                center_row, center_col = block_rows / 2, block_cols / 2
                row_indices, col_indices = np.indices((block_rows, block_cols))
                distances = np.sqrt((row_indices - center_row) ** 2 + (col_indices - center_col) ** 2)
                max_distance = np.sqrt(center_row ** 2 + center_col ** 2)
                weights = 1 - (distances / max_distance)  # Linearly decreasing weight from center

                # Accumulate weighted thresholds and weights
                accumulated_thresholds[row:row_end, col:col_end] += threshold * weights
                weight_sum[row:row_end, col:col_end] += weights

        # Handle edge regions (where blocks don't fully overlap)
        for row in range(0, rows, self.step_size):
            for col in range(0, cols, self.step_size):
                row_start = max(0, row - self.block_size // 2)
                col_start = max(0, col - self.block_size // 2)
                row_end = min(rows, row_start + self.block_size)
                col_end = min(cols, col_start + self.block_size)
                if row_end - row_start < self.block_size and col_end - col_start < self.block_size:
                    block = image[row_start:row_end, col_start:col_end]
                    if block.size == 0:
                        continue
                    threshold = self._compute_optimal_threshold(block)

                    block_rows, block_cols = block.shape
                    center_row, center_col = block_rows / 2, block_cols / 2
                    row_indices, col_indices = np.indices((block_rows, block_cols))
                    distances = np.sqrt((row_indices - center_row) ** 2 + (col_indices - center_col) ** 2)
                    max_distance = np.sqrt(center_row ** 2 + center_col ** 2)
                    weights = 1 - (distances / max_distance)

                    accumulated_thresholds[row_start:row_end, col_start:col_end] += threshold * weights
                    weight_sum[row_start:row_end, col_start:col_end] += weights

        # Normalize by the sum of weights (avoid division by zero)
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero in regions with no overlap
        interpolated_thresholds = accumulated_thresholds / weight_sum

        # Apply the interpolated thresholds to the image
        binary_image = np.zeros_like(image, dtype=np.uint8)
        binary_image[image > interpolated_thresholds] = 255

        # Apply morphological opening to remove small speckles
        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for morphological operation
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        return binary_image

    def apply_thresholding(self, image):
        """
        Apply optimal thresholding to an Image object in either global or local mode.

        Args:
            image (Image): Image object containing the input image to be thresholded.

        Returns:
            Image: The updated Image object with the binary result in output_image.

        Raises:
            ValueError: If the input image is not loaded.
        """
        if image.input_image is None:
            raise ValueError("Input image is not loaded. Call select_image() first.")

        # Work on a copy of the input image and ensure it's grayscale
        img = image.input_image.copy()
        # Always convert to grayscale, regardless of image_type, to ensure consistency
        if len(img.shape) == 3:  # If the image is 3D (color)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) != 2:  # If the image is neither 2D nor 3D
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # Apply thresholding based on the specified mode
        if self.thresholding_mode == "global":
            binary_image = self._apply_global_thresholding(img)
        else:  # Local mode
            binary_image = self._apply_local_thresholding(img)

        # Update the Image object with the result
        image.output_image = binary_image
        image.image_type = 'grey'  # Output is always grayscale
        return image

# Test the OptimalThresholding class
if __name__ == "__main__":
    # Initialize the QApplication (required for PyQt5 GUI operations)
    app = QApplication(sys.argv)

    # Create an Image object and load an image
    test_image = Image()
    test_image.select_image()  # Opens a file dialog to select an image

    if test_image.input_image is not None:
        # Test global thresholding
        global_thresholding = OptimalThresholding("global")
        global_result = global_thresholding.apply_thresholding(test_image)
        global_output = global_result.output_image

        # Test local thresholding
        local_thresholding = OptimalThresholding("local")
        local_result = local_thresholding.apply_thresholding(test_image)
        local_output = local_result.output_image

        # Display the original and thresholded images
        original_display = test_image.input_image
        if len(original_display.shape) == 3:  # If the original is color
            original_display = cv2.cvtColor(original_display, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Original Image", original_display)
        cv2.imshow("Global Thresholding", global_output)
        cv2.imshow("Local Thresholding", local_output)

        # Save the results for reference
        cv2.imwrite("global_thresholded.png", global_output)
        cv2.imwrite("local_thresholded.png", local_output)

        # Wait for a key press and then close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image was selected.")

    # Exit the application
    sys.exit(app.exec_())
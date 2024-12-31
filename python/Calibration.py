
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_last_thresholds(file_path='thresholds_log.txt'):
    """Reads the last logged thresholds from the file."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # If the file is not empty, get the last line
            if lines:
                last_line = lines[-1].strip()
                # Extract the values of lower and upper thresholds
                lower_threshold = int(last_line.split('Final Lower Threshold: ')[1].split(',')[0])
                upper_threshold = int(last_line.split('Final Upper Threshold: ')[1].split(',')[0])
                return lower_threshold, upper_threshold
            else:
                print("No thresholds found in the file.")
                return None, None
    except FileNotFoundError:
        print("Thresholds log file not found.")
        return None, None
    
def preprocess_image(image_path):
    '''
    Preprocess the input image by reading, converting to grayscale, 
    applying Gaussian blur, and detecting edges using Canny.

    Parameters:
        image_path (str): Path to the input image file.

    Returns:
        tuple: Edges image and original image.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
        ValueError: If the provided path is not a valid image file.
    '''
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from path: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        lower_threshold, upper_threshold = read_last_thresholds()
        if lower_threshold is not None and upper_threshold is not None:
            edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
        else:
            edges = cv2.Canny(blurred, 50, 150)
        return edges, image
    except Exception as e:
        raise ValueError(f"Error in preprocessing image: {e}")

def find_contour(edges):
    '''
    Identify contours in the edge-detected image and select the largest contour.

    Parameters:
        edges (numpy.ndarray): Edge-detected image.

    Returns:
        tuple: All contours and the largest contour.

    Raises:
        ValueError: If no contours are found.
    '''
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the edge-detected image.")
    
    # Assuming the largest contour is the hanging drop
    contour = max(contours, key=cv2.contourArea)
    return contours, contour

def measure_diameters(contour):
    '''
    Calculate the maximum diameter (d_e) and secondary diameter (d_s) of a drop.

    Parameters:
        contour (numpy.ndarray): Largest contour points.

    Returns:
        tuple: Maximum diameter (d_e), secondary diameter (d_s), 
               left and right points at y-level for d_s.

    Raises:
        ValueError: If contour data is not in expected format.
    '''
    try:
        # Get the points in the contour
        points = np.squeeze(contour)
        if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("Contour points are not in the expected format.")
        # Split the x and y coordinates
        x = points[:, 0]
        y = points[:, 1]

        # Perform interpolation (linear interpolation)
        f_x = interp1d(np.arange(len(x)), x, kind='linear', fill_value="extrapolate")
        f_y = interp1d(np.arange(len(y)), y, kind='linear', fill_value="extrapolate")

        # Generate new set of points
        new_x = f_x(np.linspace(0, len(x)-1, num=500))  # 500 points for smoother curve
        new_y = f_y(np.linspace(0, len(y)-1, num=500))
        points = np.vstack((new_x, new_y)).T
        # Find the leftmost and rightmost points for d_e
        left_point = points[np.argmin(points[:, 0])]
        right_point = points[np.argmax(points[:, 0])]
        d_e = np.linalg.norm(right_point - left_point)
        
        # Find the lowest point
        bottom_point = points[np.argmax(points[:, 1])]
        
        # Find the y-coordinate d_e units above the bottom point
        y_above = bottom_point[1] - d_e
        
        # Find the points within a small range around y_above
        tolerance = 3  # Adjust this value as needed
        points_at_y_above = points[(points[:, 1] >= y_above - tolerance) & (points[:, 1] <= y_above + tolerance)]

        
        if len(points_at_y_above) == 0:
            print("No points found at y = bottom_y - d_e level.")
            return d_e, None, None, None

        left_point_at_y = points_at_y_above[np.argmin(points_at_y_above[:, 0])]
        right_point_at_y = points_at_y_above[np.argmax(points_at_y_above[:, 0])]
        d_s = np.linalg.norm(right_point_at_y - left_point_at_y)
        return d_e, d_s, left_point_at_y, right_point_at_y
    except Exception as e:
        raise ValueError(f"Error in measuring diameters: {e}")

def calculate_needle_diameter(image, blur_kernel=(5, 5)):
    '''
    Calculate the diameter of the needle using edge detection and contour analysis.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        blur_kernel (tuple): Gaussian blur kernel size.
        canny_thresholds (tuple): Canny edge detection thresholds.

    Returns:
        tuple: Diameter of the needle in pixels and its key points.

    Raises:
        ValueError: If no contours are found or contour format is incorrect.
    '''
    try:
        # Step 1: Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

        # Step 3: Perform edge detection using Canny
        lower_threshold, upper_threshold = read_last_thresholds()
        if lower_threshold is not None and upper_threshold is not None:
            edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
        else:
            edges = cv2.Canny(blurred, 50, 150)

        # Step 4: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found. Check the image quality or adjust parameters.")

        # Step 5: Use the largest contour to determine the needle diameter
        largest_contour = max(contours, key=cv2.contourArea)
        points = np.squeeze(largest_contour)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Contour points are not in the expected format.")

        # Step 6: Find the 10th point from the top based on the y-coordinate
        sorted_points = sorted(points, key=lambda x: x[1])  # Sort by y-coordinate
        tenth_point = sorted_points[0]  # 10th point from the top (index 9)
        y_level = tenth_point[1]  # Use the y-coordinate of this point as the reference
        
        # Step 7: Find leftmost and rightmost points at this y-level
        leftmost_point = None
        rightmost_point = None

        for point in points:
            if point[1] == y_level:  # Check if the point is at the specified height
                if leftmost_point is None or point[0] < leftmost_point[0]:
                    leftmost_point = point
                if rightmost_point is None or point[0] > rightmost_point[0]:
                    rightmost_point = point

        # Step 8: Calculate the needle diameter
        if leftmost_point is None or rightmost_point is None:
            raise ValueError("Could not find leftmost or rightmost points at the specified height.")

        diameter = np.linalg.norm(np.array(rightmost_point) - np.array(leftmost_point))
        return diameter,leftmost_point,rightmost_point
    except Exception as e:
        raise ValueError(f"Error in calculating needle diameter: {e}")


def calculate_corrected_needle_diameter(d_e, d_s, d_n):
    '''
    Calculate corrected needle diameter using measured diameters and known parameters.

    Parameters:
        d_e (float): Maximum diameter of the drop in pixels.
        d_s (float): Secondary diameter of the drop in pixels.
        d_n (float): diameter of the needle in pixels.

    Returns:
        float: Corrected needle diameter in mm.
    '''
    # H = ((d_s/d_e)**2.4898)/(0.35)  # correction coefficient H
    try:
        C_n_d= (((21.292282)*(d_s**2.49)*(d_n**2))/(d_e**4.49))**0.5
        return C_n_d
    except ZeroDivisionError:
        raise ValueError("Maximum diameter (d_e) must not be zero.")
    except Exception as e:
        raise ValueError(f"Error in calibrating: {e}")

def main(image_path):
    '''
    Main function to process an image, measure diameters, 
    calculate surface tension, and visualize results.

    Parameters:
        image_path (str): Path to the input image file.
    '''
    edges, image = preprocess_image(image_path)
    contours, contour = find_contour(edges)
    de, ds, left_point_at_y, right_point_at_y = measure_diameters(contour)
    if ds is None:
        print("No points found at y = bottom_y - d_e level")
        return
    
    needle_diameter,nlp,nrp=calculate_needle_diameter(image)
    corrected_needle_diameter = calculate_corrected_needle_diameter(de,ds,needle_diameter)
    
    # Create the text content
    text_content = (
        f"Corrected diameter of the needle after calibration: {corrected_needle_diameter} mm\n"   
    )

    # File path to save the data
    file_path = "Calibration_result.txt"

    # Write the data to a text file
    with open(file_path, "w") as file:
        file.write(text_content)

# image_path = r"C:\Users\91982\Downloads\pdlab3.jpg"
image_path = r"c:\Users\Prem\OneDrive\Pictures\Screenshots\Screenshot 2024-12-22 051457.png"
main(image_path)


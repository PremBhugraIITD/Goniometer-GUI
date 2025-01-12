import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import defaultdict
# from ADB_extractor import image_retrieval_from_phone
from Thresholds import get_threshes
# Global variables for cropping
ref_point = []
cropping = False

# File path to save the data
file_path = "Static_Contact_Angle.txt"
with open(file_path, "w") as file:
    file.write("Sessile Drop Analysis started...\n")

# Mouse callback function to select the cropping area
def crop_image(event, x, y, flags, param):
    '''
    Handles mouse events for cropping an image. Tracks the region selected by the user
    using left mouse button events and dynamically updates the cropping area.

    Args:
        event: OpenCV mouse event (e.g., button press, movement, release).
        x, y: Coordinates of the mouse pointer during the event.
        flags: OpenCV flags for mouse event.
        param: Clone of the image to allow real-time rectangle drawing.
    '''
    global ref_point, cropping

    clone = param.copy()  # Copy the original image to draw the rectangle dynamically

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            # Draw a rectangle in real-time as the user selects the ROI
            current_point = (x, y)
            top_left = (min(ref_point[0][0], current_point[0]), min(ref_point[0][1], current_point[1]))
            bottom_right = (max(ref_point[0][0], current_point[0]), max(ref_point[0][1], current_point[1]))
            cv2.rectangle(clone, top_left, bottom_right, (0, 0, 255), 2)  # Draw the rectangle in red
            # Resize the window (width, height)
            cv2.resizeWindow("Crop Image", 960, 540)
            # Move the window (x, y position on screen)
            cv2.moveWindow("Crop Image", 0 , 0)
            cv2.imshow("Crop Image", clone)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        if len(ref_point) != 2:
            with open(file_path, "a") as file:
                file.write("Invalid cropping points. Please select a valid ROI.\n")
            raise ValueError("Invalid cropping points. Please select a valid ROI.")
        
        # Draw the final rectangle around the region of interest
        top_left = (min(ref_point[0][0], x), min(ref_point[0][1], y))
        bottom_right = (max(ref_point[0][0], x), max(ref_point[0][1], y))
        cv2.rectangle(param, top_left, bottom_right, (0, 0, 255), 2)  # Draw the rectangle in red
        # Resize the window (width, height)
        cv2.resizeWindow("Crop Image", 960, 540)
        # Move the window (x, y position on screen)
        cv2.moveWindow("Crop Image", 0, 0)
        cv2.imshow("Crop Image", param)

# Function to resize the image to fit within the screen
def resize_to_fit_screen(image, screen_width=800, screen_height=600):
    """
    Resizes the image to fit within the screen dimensions while maintaining the aspect ratio.

    Args:
        image: The input image to resize.
        screen_width: Maximum width of the screen.
        screen_height: Maximum height of the screen.

    Returns:
        Resized image and the scale factor.
    """
    h, w = image.shape[:2]
    scale = min(screen_width / w, screen_height / h)
    if scale < 1:  # Resize only if the image is larger than the screen
        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized_image, scale
    return image, 1.0  # Return the original image if no resizing is needed

# Function to allow the user to crop the image
def get_cropped_image(image):
    '''
    Allows the user to interactively crop the input image using OpenCV.

    Args:
        image: Input image to be cropped.

    Returns:
        Cropped portion of the image or the original image if cropping is skipped.
    '''
    global ref_point
    
    if image is None:
        with open(file_path, "a") as file:
            file.write("Input image is None. Please provide a valid image.\n")
        raise ValueError("Input image is None. Please provide a valid image.")
    
    # Resize the image to fit the screen
    resized_image, scale = resize_to_fit_screen(image)

    clone = resized_image.copy()

    # Set up the window and set the mouse callback function for cropping
    cv2.namedWindow("Crop Image")
    # Resize the window (width, height)
    cv2.resizeWindow("Crop Image", 960, 540)
    # Move the window (x, y position on screen)
    cv2.moveWindow("Crop Image", 0, 0)
    cv2.setMouseCallback("Crop Image", crop_image, param=clone)

    # Display the image and wait for cropping
    while True:
        cv2.imshow("Crop Image", clone)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset cropping
        if key == ord("r"):
            ref_point = []  # Clear previous ROI points
            # clone = image.copy()  # Reset the clone to the original image
            print("Reset cropping selection.")
            with open(file_path, "a") as file:
                file.write("Reset cropping selection.\n")

        # Press 'c' to confirm the crop and break the loop
        elif key == ord("c") and len(ref_point) == 2:
            print("Cropping confirmed.")
            with open(file_path, "a") as file:
                file.write("Cropping confirmed.\n")
            break

        # Press 'n' to skip cropping
        elif key == ord("n"):
            cv2.destroyAllWindows()  # Close the window
            return image  # Return the original image without cropping

        # Check if the window is closed
        if cv2.getWindowProperty("Crop Image", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            with open(file_path, "a") as file:
                file.write("Window closed.\n")
            cv2.destroyAllWindows()
            return image  # Return the original image without cropping

    # Close the cropping window after confirmation
    cv2.destroyAllWindows()

    # Crop the selected area
    if len(ref_point) == 2:
        x0, y0 = int(ref_point[0][0] / scale), int(ref_point[0][1] / scale)
        x1, y1 = int(ref_point[1][0] / scale), int(ref_point[1][1] / scale)
        # Get the coordinates in correct order
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        
        if x0 == x1 or y0 == y1:
            with open(file_path, "a") as file:
                file.write("Cropping region must have a non-zero area.\n")
            raise ValueError("Cropping region must have a non-zero area.")
        
        cropped_image = image[y0:y1, x0:x1]
        return cropped_image
    else:
        return image

def read_last_thresholds(file_path2='thresholds_log.txt'):
    """Reads the last logged thresholds from the file."""
    try:
        with open(file_path2, 'r') as file:
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
                with open(file_path, "a") as file:
                    file.write("No thresholds found in the file.\n")
                return None, None
    except FileNotFoundError:
        print("Thresholds log file not found.")
        with open(file_path, "a") as file:
            file.write("Thresholds log file not found.\n")
        return None, None

def average_y_for_same_x(points):
    '''
    Averages the y-coordinates for all points sharing the same x-coordinate.

    Args:
        points: List or array of [x, y] points.

    Returns:
        Numpy array of unique x-coordinates with averaged y-coordinates.
    '''
    if not points.any():
        with open(file_path, "a") as file:
            file.write("Input points array of points with same x-coordinate is empty.\n")
        raise ValueError("Input points array of points with same x-coordinate is empty.")
    
    # Dictionary to store points grouped by x-coordinates
    x_dict = defaultdict(list)

    # Group points by their x-coordinate
    for point in points:
        x_dict[point[0]].append(point[1])  # Add y-coordinates for the same x-coordinate

    # Create a new list of points with unique x and averaged y
    averaged_points = []
    for x, y_values in x_dict.items():
        avg_y = np.mean(y_values)  # Calculate the average of y-values for the same x-coordinate
        averaged_points.append([x, avg_y])

    return np.array(averaged_points)

def calculate_contact_angle(image, baseline_y):
    '''
    Calculates the static contact angle of a droplet on a surface based on the contour.

    Args:
        image: Input image of the droplet.
        baseline_y: Baseline y-coordinate for contact angle calculation.

    Returns:
        Average contact angle or None if calculation fails.
    '''
    if image is None or baseline_y < 0 or baseline_y >= image.shape[0]:
        with open(file_path, "a") as file:
            file.write("Invalid image or baseline position.\n")
        raise ValueError("Invalid image or baseline position.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform edge detection using Canny
    lower_threshold, upper_threshold = read_last_thresholds()
    if lower_threshold is not None and upper_threshold is not None:
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    else:
        edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found!")
        with open(file_path, "a") as file:
            file.write("No contours found!\n")
        return None

    all_points = np.vstack(contours)  # Stack all contour points into one array
    contour = all_points.reshape(-1, 2)

    # Get the x, y coordinates of the contour points
    contour_points = np.squeeze(contour)

    # Filter contour points above the baseline
    above_baseline_points = contour_points[contour_points[:, 1] < baseline_y]

    if len(above_baseline_points) < 3:
        print("Not enough points found above the baseline for contact angle calculation.")
        with open(file_path, "a") as file:
            file.write("Not enough points found above the baseline for contact angle calculation.\n")
        return None

    # Sort points by x-coordinate
    sorted_points = above_baseline_points[np.argsort(above_baseline_points[:, 0])]

    # Find intersection points
    intersection_points = sorted_points[np.isclose(sorted_points[:, 1], baseline_y, atol=1.0)]
    if len(intersection_points) < 2:
        print("Could not find sufficient intersection points with the baseline.")
        with open(file_path, "a") as file:
            file.write("Could not find sufficient intersection points with the baseline.\n")
        return None

    left_intersection = intersection_points[0]
    right_intersection = intersection_points[-1]

    # Case 1: Intersection points are the first and last elements
    if np.all(sorted_points[0] == left_intersection) and np.all(sorted_points[-1] == right_intersection):
        obtuse = False
        left_points = sorted_points[:201]  # First 11 points
        left_points = average_y_for_same_x(left_points)
        right_points = sorted_points[-201:]  # Last 11 points
        right_points  = average_y_for_same_x(right_points)

    else:
        # Case 2: Intersection points are in between
        obtuse = True
        # Distance threshold from the baseline (y-axis proximity check)
        distance_threshold = 80  # Adjust based on your image scale

        # Calculate distance from baseline for all points
        distances_from_baseline = np.abs(sorted_points[:, 1] - baseline_y)

        # Select only points within the distance threshold near the left intersection
        left_points = sorted_points[(distances_from_baseline < distance_threshold) & 
                                    (sorted_points[:, 0] <= left_intersection[0])]
        left_points = average_y_for_same_x(left_points)
        # Select only points within the distance threshold near the right intersection
        right_points = sorted_points[(distances_from_baseline < distance_threshold) & 
                                    (sorted_points[:, 0] >= right_intersection[0])]
        right_points = average_y_for_same_x(right_points)


    # Polynomial fitting and contact angle calculation
    def calculate_angle(points, intersection_point):
        poly_coeffs = np.polyfit(points[:, 0], points[:, 1], 2)
        poly = np.poly1d(poly_coeffs)
        slope = np.polyder(poly)(intersection_point[0])
        angle = np.arctan(slope) * (180 / np.pi)
        return angle

    # Calculate left and right contact angles
    if obtuse:
        left_angle = 180 - calculate_angle(left_points, left_intersection)
        right_angle = 180 + calculate_angle(right_points, right_intersection)
    else : 
        left_angle = -calculate_angle(left_points, left_intersection)
        right_angle = calculate_angle(right_points, right_intersection)

    # Average contact angle
    avg_contact_angle = (left_angle + right_angle) / 2

    # Create the text content
    text_content = (
        f"Left Contact Angle: {left_angle:.2f} degrees\n"
        f"Right Contact Angle: {right_angle:.2f} degrees\n"
        f"Average Contact Angle: {avg_contact_angle:.2f} degrees\n"
    )

    # Write the data to a text file
    with open(file_path, "a") as file:
        file.write(text_content)
        
    # Plotting for visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot contours and intersection points
    plt.plot(sorted_points[:, 0], sorted_points[:, 1], 'b.', label='Contour Points')
    plt.plot(left_points[:, 0], left_points[:, 1], 'go', label='Left Fit Points')
    plt.plot(right_points[:, 0], right_points[:, 1], 'ro', label='Right Fit Points')
    plt.axhline(y=baseline_y, color='blue', linestyle='--', label='Baseline')
    plt.plot(left_intersection[0], left_intersection[1], 'yx', label='Left Intersection', markersize=10)
    plt.plot(right_intersection[0], right_intersection[1], 'yx', label='Right Intersection', markersize=10)
    
    plt.title(f'Contact Angle: {avg_contact_angle:.2f} degrees')
    plt.legend()
    # Access the current figure's Tkinter window
    canvas = plt.gcf().canvas
    tk_window = canvas.manager.window
    # Set the window size and position using Tkinter's geometry method
    tk_window.geometry("960x540+0+0")
    plt.show()

    return avg_contact_angle

# Function to display image and let user select baseline using a slider
def select_baseline(image):
    '''
    Provides an interactive slider for the user to select the baseline y-coordinate.

    Args:
        image: Input image to display for baseline selection.

    Returns:
        Selected baseline y-coordinate.
    '''
    if image is None:
        with open(file_path, "a") as file:
            file.write("Input image is None. Please provide a valid image.\n")
        raise ValueError("Input image is None. Please provide a valid image.")
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Adjust the baseline using the slider')

    # Initial baseline y position
    init_baseline_y = image.shape[0] // 2

    # Create a slider for baseline adjustment
    ax_baseline = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    baseline_slider = Slider(ax_baseline, 'Baseline Y', 0, image.shape[0], valinit=init_baseline_y)

    # Draw the baseline
    line = ax.axhline(y=init_baseline_y, color='r', linestyle='--')

    # Update function for the slider
    def update(val):
        baseline_y = baseline_slider.val
        line.set_ydata([baseline_y] * len(line.get_xdata()))  # Use len() instead of .size
        fig.canvas.draw_idle()

    baseline_slider.on_changed(update)
    # Access the current figure's Tkinter window
    canvas = plt.gcf().canvas
    tk_window = canvas.manager.window
    # Set the window size and position using Tkinter's geometry method
    tk_window.geometry("960x540+0+0")
    plt.show()

    return int(baseline_slider.val)


# Full function to process the image, select baseline, and calculate contact angle
def process_image(image):
    '''
    Orchestrates the process of cropping, baseline selection, and contact angle calculation.

    Args:
        image: Input image of the droplet.
    '''
    if image is None:
        with open(file_path, "a") as file:
            file.write("Input image is None. Please provide a valid image.\n")
        raise ValueError("Input image is None. Please provide a valid image.")
    global cropped_image
    # Get the cropped image from the user
    cropped_image = get_cropped_image(image)
    # Check if the user has skipped cropping
    if np.array_equal(cropped_image, image):
        # User chose not to crop, proceed with original image
        print("No cropping done. Using the original image.")
        with open(file_path, "a") as file:
            file.write("No cropping done. Using the original image.\n")
    get_threshes(cropped_image)
    # Select baseline using slider
    baseline_y = select_baseline(cropped_image)

    # Calculate contact angle
    avg_contact_angle = calculate_contact_angle(cropped_image, baseline_y)
    
    if avg_contact_angle is None:
        print("Contact angle calculation failed.")
        with open(file_path, "a") as file:
            file.write("Contact angle calculation failed.\n")

def get_latest_file(directory):
    """
    Get the most recent file from the specified directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        str: Path to the latest file, or None if no files are found.
    """
    try:
        # Convert directory path to a Path object
        path = Path(directory)

        # List all files in the directory
        files = [f for f in path.iterdir() if f.is_file()]

        if not files:
            print(f"No files found in {directory}.")
            with open(file_path, "a") as file:
                file.write(f"No files found in {directory}.\n")
            return None

        # Find the most recent file based on modification time
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        with open(file_path, "a") as file:
            file.write(f"An error occurred: {e}\n")
        return None

def load_latest_image_path():
    # Replace with the directory where files are stored on your PC
    DIRECTORY = r"C:\Users\91982\OneDrive\Desktop\SURA\Retrieved_Data_Input"

    latest_file = get_latest_file(DIRECTORY)
    if latest_file:
        return latest_file
    else:
        print("No file could be retrieved.")
        with open(file_path, "a") as file:
            file.write("No file could be retrieved.\n")

def sessile_drop():
    # image_retrieval_from_phone()
    os.chdir(r"C:\Users\91982\OneDrive\Desktop\SURA")
    image_path= str(load_latest_image_path())
    image = cv2.imread(image_path)
    process_image(image)

# sessile_drop()
# # Load the image
# # image_path = r"C:\Users\91982\Downloads\no annotation 2.png" 
# image_path = r"C:\Users\91982\Downloads\drop4.jpg"
# image_path = r"C:\Users\91982\OneDrive\Pictures\Screenshots\Screenshot 2024-08-07 230147.png"
# image_path = r"C:\Users\91982\OneDrive\Pictures\Screenshots\Screenshot 2024-08-07 230147.png"
# image_path = r"C:\Users\91982\Downloads\img19.jpg"
# image_path = r"C:\Users\91982\OneDrive\Pictures\Screenshots\Screenshot 2024-12-20 025005.png"
# # image_path = r"C:\Users\91982\OneDrive\Pictures\Screenshots\Screenshot 2024-08-07 230147.png"

image_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\image_final.png"
# image_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\image_sample.png"

image = cv2.imread(image_path)

# # Process the image for contact angle calculation
process_image(image)

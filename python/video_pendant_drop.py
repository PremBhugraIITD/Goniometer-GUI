import os
import io
import math
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.widgets import Slider
from pathlib import Path
from scipy.interpolate import interp1d
# from ADB_extractor import image_retrieval_from_phone
from Thresholds import get_threshes

# Global variables for cropping
ref_point = []
cropping = False

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
            
            # Move the window (x, y position on screen)
            cv2.moveWindow("Crop Image", 0 , 0)
            cv2.imshow("Crop Image", clone)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        if len(ref_point) != 2:
            raise ValueError("Invalid cropping points. Please select a valid ROI.")
        
        # Draw the final rectangle around the region of interest
        top_left = (min(ref_point[0][0], x), min(ref_point[0][1], y))
        bottom_right = (max(ref_point[0][0], x), max(ref_point[0][1], y))
        cv2.rectangle(param, top_left, bottom_right, (0, 0, 255), 2)  # Draw the rectangle in red
        # Resize the window (width, height)
        
        # Move the window (x, y position on screen)
        cv2.moveWindow("Crop Image", 0 , 0)
        cv2.imshow("Crop Image", param)

# Function to resize the image to fit within the screen
def resize_to_fit_screen(image, screen_width=1920, screen_height=1080):
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
    global ref_point,cropping
    
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    
    # Resize the image to fit the screen
    resized_image, scale = resize_to_fit_screen(image)

    clone = resized_image.copy()

    # Set up the window and set the mouse callback function for cropping
    cv2.namedWindow("Crop Image")
    # Move the window (x, y position on screen)
    cv2.moveWindow("Crop Image", 0 , 0)
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

        # Press 'c' to confirm the crop and break the loop
        elif key == ord("c") and len(ref_point) == 2:
            cropping=True
            print("Cropping confirmed.")
            break

        # Press 'n' to skip cropping
        elif key == ord("n"):
            cv2.destroyAllWindows()  # Close the window
            return image,1  # Return the original image without cropping

        # Check if the window is closed
        if cv2.getWindowProperty("Crop Image", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            cv2.destroyAllWindows()
            return image,1 # Return the original image without cropping

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
            raise ValueError("Cropping region must have a non-zero area.")
        
        cropped_image = image[y0:y1, x0:x1]
        return cropped_image,scale
    else:
        return image,scale

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

def read_corrected_diameter(file_path='Calibration_result.txt'):
    """
    Reads the corrected diameter of the needle from the calibration result file.

    Args:
        file_path (str): Path to the calibration result file.

    Returns:
        float: Corrected diameter of the needle if found.
        None: If the file is missing or the diameter cannot be extracted.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "Corrected diameter of the needle after calibration" in line:
                    # Extract the numerical value before "mm"
                    diameter = float(line.split(':')[1].strip().split()[0])
                    return diameter
        print("No diameter value found in the file.")
        return None
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except ValueError:
        print("Error extracting the calibration value. Recalibrate again.")
        return None

def preprocess_image(image):
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
        global cropped_image
        if image is None:
            raise FileNotFoundError(f"Could not find Image.")
        # Get the cropped image from the user
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
    
    all_points = np.vstack(contours)  # Stack all contour points into one array
    contour = all_points.reshape(-1, 2)

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
        # points = np.vstack((new_x, new_y)).T
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
        all_points = np.vstack(contours)  # Stack all contour points into one array
        largest_contour = all_points.reshape(-1, 2)
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


def calculate_surface_tension(d_e, d_s, density=997, gravity=9.81):
    '''
    Calculate surface tension using measured diameters and known parameters.

    Parameters:
        d_e (float): Maximum diameter of the drop in meters.
        d_s (float): Secondary diameter of the drop in meters.
        density (float): Density of the liquid in kg/m^3.
        gravity (float): Acceleration due to gravity in m/s^2.

    Returns:
        float: Surface tension in mN/m.
    '''
    # H = ((d_s/d_e)**2.4898)/(0.35)  # correction coefficient H
    try:
        surface_tension = ((abs(density - 1.2) * gravity * (d_e ** 4.49) * 0.35) / (d_s ** 2.49))
        print("Surface tension calculated successfully.")
        return surface_tension * 1000
    except ZeroDivisionError:
        raise ValueError("Secondary diameter (d_s) must not be zero.")
    except Exception as e:
        raise ValueError(f"Error in calculating surface tension: {e}")


def main(image, O_n_d, density):
    '''
    Main function to process an image, measure diameters, 
    calculate surface tension, and visualize results.

    Parameters:
        image_path (str): Path to the input image file.
    '''
    edges, image = preprocess_image(image)
    contours, contour = find_contour(edges)
    de, ds, left_point_at_y, right_point_at_y = measure_diameters(contour)
    if ds is None:
        print("No points found at y = bottom_y - d_e level")
        return
    
    diameter = read_corrected_diameter()
    if diameter is not None:
        O_n_d=diameter
    needle_diameter,nlp,nrp=calculate_needle_diameter(image)
    
    d_e,d_s=(de*O_n_d)/needle_diameter, (ds*O_n_d)/needle_diameter
    surface_tension = calculate_surface_tension(d_e, d_s, density)

    # Visualize the results for the drop
    left_point = contour[np.argmin(contour[:, 0])]
    right_point = contour[np.argmax(contour[:, 0])]
    bottom_point = contour[np.argmax(contour[:, 1])]
    return surface_tension,contours, contour,nlp,nrp, left_point, right_point, bottom_point, left_point_at_y, right_point_at_y, d_e, d_s, O_n_d

# Initialize video input and processing logic
def process_video_for_Surface_Tension(video_path,density,O_n_d=0.0010194):
    """
    Process video for hysteresis analysis, calculating contact angles and hysteresis.

    Args:
        video_path (str): Path to the input video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Error: Unable to open video file.")

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read the first frame.")

    roi, scale = get_cropped_image(first_frame)
    if roi is None:
        raise ValueError("Failed to select ROI.")
    get_threshes(roi)
    if len(ref_point) == 2:
        x0, y0 = int(ref_point[0][0] / scale), int(ref_point[0][1] / scale)
        x1, y1 = int(ref_point[1][0] / scale), int(ref_point[1][1] / scale)
        # Get the coordinates in correct order
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
    
    # Initialize video writer (you can adjust the output path, codec, and FPS as needed)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('output_pendant_drop_video.mp4', fourcc, 30.0, (640, 480))  # Adjust size accordingly

    # Initialize matplotlib for live graph
    plt.ion()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)  # 6.4 * 100 = 640, 4.8 * 100 = 480
    surface_tensions = []
    frame_indices = []
    Max_diameter_values = []
    Sec_diameter_values = []
    line_surface_tension, = ax.plot([], [], label='Surface Tension (\xb0)', color='r')
    ax.set_xlim(0, 100)  # Initial limits, will be updated dynamically
    ax.set_ylim(90, 110)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Surface Tension (\xb0)")
    ax.legend()
      # Access the current figure's Tkinter window
    # canvas = plt.gcf().canvas
    # tk_window = canvas.manager.window
    # # Set the window size and position using Tkinter's geometry method
    # tk_window.geometry("960x540+0+0")  # Position at (0, 0) with size 960x540
    frame_index = 0
    with open('Surface_Tension_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Maximum diameter of the drop (d_e)", "Diameter at the lowest end (d_s)", "Surface_Tension"])

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("End of video stream or error in reading the video file.")
                break
            
            # Process the frame for contact angle calculations
            if cropping:
                frame = frame[y0:y1, x0:x1]
            result = main(frame,O_n_d=O_n_d,density=density)
            if result is None:
                print("The function returned None.")
                continue
            surface_tension,contours, contour,nlp,nrp, left_point, right_point, bottom_point, left_point_at_y, right_point_at_y, d_e, d_s, O_n_d=result
            if d_e == 0 or d_s == 0 or math.isinf(d_e) or math.isinf(d_s) or math.isnan(d_e) or math.isnan(d_s):
                print("Warning: d_e or d_s is either zero or not finite, skipping this calculation.")
                continue


            if surface_tension is not None and surface_tension<100:
                surface_tensions.append(surface_tension)
                Max_diameter_values.append(d_e)
                Sec_diameter_values.append(d_s)
                writer.writerow([frame_index, d_e, d_s, surface_tension])
                frame_indices.append(frame_index)


            filtered_Surface_Tensions = [surface_tension for surface_tension in surface_tensions if surface_tension is not None]
            filtered_Surface_Tensions = np.array(filtered_Surface_Tensions)
            filtered_Surface_Tensions = filtered_Surface_Tensions[np.isfinite(filtered_Surface_Tensions)]


            # Update live plot for all three lines
            line_surface_tension.set_xdata(frame_indices)
            line_surface_tension.set_ydata(surface_tensions)


            ax.set_xlim(0, max(100, frame_index))
            ax.set_ylim(0, 100)
            plt.draw()
            plt.pause(0.05)
            # if filtered_Surface_Tensions.size > 0:
            #     ax.set_ylim(min(filtered_Surface_Tensions) - 5, max(filtered_Surface_Tensions) + 5)
            # else:
            #     ax.set_ylim(90, 110)  # Default y-axis range
            

            if frame_index % 5 == 0:  # Show every 10th frame
                image = frame
                 # Draw all contours
                cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

                # Draw key points for the drop
                cv2.circle(image, (int(left_point[0]),int(left_point[1])), 5, (255, 0, 0), -1)
                cv2.circle(image, (int(right_point[0]),int(right_point[1])), 5, (255, 0, 0), -1)
                cv2.circle(image, (int(bottom_point[0]),int(bottom_point[1])), 5, (255, 0, 0), -1)
                cv2.circle(image, (int(left_point_at_y[0]),int(left_point_at_y[1])), 5, (255, 0, 255), -1)
                cv2.circle(image, (int(right_point_at_y[0]),int(right_point_at_y[1])), 5, (255, 0, 255), -1)
                cv2.circle(image, (int(nlp[0]),int(nlp[1])), 5, (0, 0, 255), -1)
                cv2.circle(image, (int(nrp[0]),int(nrp[1])), 5, (0, 0, 255), -1)

                # Draw lines between key points for visualization
                cv2.line(image, (int(left_point[0]),int(left_point[1])), (int(right_point[0]),int(right_point[1])), (0, 0, 255), 1)
                cv2.line(image, (int(left_point_at_y[0]),int(left_point_at_y[1])), (int(right_point_at_y[0]),int(right_point_at_y[1])), (0, 255, 255), 1)
                cv2.line(image, (int(nlp[0]),int(nlp[1])), (int(nrp[0]),int(nrp[1])), (0, 255, 255), 1)

                plt.title(f'Identified Contours and Key Points\nd_e: {d_e*1000:.2f} mm, d_s: {d_s*1000:.2f} mm, Needle Diameter: {O_n_d*1000:.2f} mm')
                cv2.imshow("Video Analysis", frame)
                

            output_video.write(frame)
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show(block=True)

def pendant_drop(O_n_d=1.0194*10**(-3), density=997):
    # image_retrieval_from_phone()
    os.chdir(r"C:\Users\91982\OneDrive\Desktop\SURA")
    # image_path= str(load_latest_image_path())
    # image_path = r"C:\Users\91982\Downloads\pdgoniochloro.jpg"
    video_path =  r"C:\Users\91982\Downloads\chloro.mp4"
    process_video_for_Surface_Tension(video_path,O_n_d,density)

# pendant_drop(density=1489.2)

# O_n_d=0.0010194
# density=997

video_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\video_final.mp4"
# video_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\video_sample.mp4"
# video_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\video_sample2.mp4"
# video_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\water.mp4"
# video_path = r"C:\Users\Prem\OneDrive - IIT Delhi\Desktop\GitHub\S.U.R.A.-2024\python\chloroform.mp4"

with open("input_pendant.txt", "r") as file:
        lines = file.readlines()
        density = float(lines[0].strip())  # Read density from the first line
        # O_n_d = float(lines[1].strip())   # Read needle diameter from the second line

# print(f"Density: {density}, Needle Diameter: {O_n_d}")

process_video_for_Surface_Tension(video_path,density)
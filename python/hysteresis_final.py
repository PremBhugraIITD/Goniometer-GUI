# Import necessary libraries
import cv2
import numpy as np
import csv
import scipy
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import defaultdict
warnings.filterwarnings("ignore", category=np.RankWarning)

# Global variables for cropping
ref_point = []
cropping = False
vertical_lines = [None, None]

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
        cv2.imshow("Crop Image", param)

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
        raise ValueError("Input image is None. Please provide a valid image.")
    
    clone = image.copy()

    # Set up the window and set the mouse callback function for cropping
    cv2.namedWindow("Crop Image")
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
            print("Cropping confirmed.")
            break

        # Press 'n' to skip cropping
        elif key == ord("n"):
            cv2.destroyAllWindows()  # Close the window
            return image  # Return the original image without cropping

        # Check if the window is closed
        if cv2.getWindowProperty("Crop Image", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            cv2.destroyAllWindows()
            return image  # Return the original image without cropping

    # Close the cropping window after confirmation
    cv2.destroyAllWindows()

    # Crop the selected area
    if len(ref_point) == 2:
        x0, y0 = ref_point[0]
        x1, y1 = ref_point[1]
        # Get the coordinates in correct order
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        
        if x0 == x1 or y0 == y1:
            raise ValueError("Cropping region must have a non-zero area.")
        
        cropped_image = image[y0:y1, x0:x1]
        return cropped_image
    else:
        return image

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

# Baseline selection from the first frame
def select_baseline(image):
    """
    Select the baseline y-position from the image by using a slider in a matplotlib plot.

    Args:
        image (numpy.ndarray): The input image for baseline selection.

    Returns:
        int: Selected baseline y-position.
    """
    if image is None:
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
    plt.show()

    return int(baseline_slider.val)

def select_vertical_lines(image):
    '''
    Allow the user to select two vertical lines around the needle using sliders.
    Args:
        image: The input image for vertical line selection.
    '''
    global vertical_lines

    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Adjust the vertical lines using the sliders')

    # Initial positions for the vertical lines (starting at one-third and two-thirds of image width)
    init_left_line = image.shape[1] // 3
    init_right_line = 2 * image.shape[1] // 3

    # Create sliders for vertical lines
    ax_left_line = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    left_slider = Slider(ax_left_line, 'Left Line X', 0, image.shape[1], valinit=init_left_line)
    ax_right_line = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    right_slider = Slider(ax_right_line, 'Right Line X', 0, image.shape[1], valinit=init_right_line)

    # Create vertical lines based on the sliders
    left_line = ax.axvline(x=init_left_line, color='g', linestyle='--')
    right_line = ax.axvline(x=init_right_line, color='g', linestyle='--')

    def update_lines(val):
        left_line.set_xdata([left_slider.val] * len(left_line.get_ydata()))
        right_line.set_xdata([right_slider.val] * len(right_line.get_ydata()))
        fig.canvas.draw_idle()

    left_slider.on_changed(update_lines)
    right_slider.on_changed(update_lines)

    plt.show()

    vertical_lines[0] = int(left_slider.val)
    vertical_lines[1] = int(right_slider.val)

    return vertical_lines

def average_y_for_same_x(points):
    '''
    Averages the y-coordinates for all points sharing the same x-coordinate.

    Args:
        points: List or array of [x, y] points.

    Returns:
        Numpy array of unique x-coordinates with averaged y-coordinates.
    '''
    if not points.any():
        raise ValueError("Input points array of points with same x-coordinate is empty.")
    
    x_dict = defaultdict(list)

    for point in points:
        x_dict[point[0]].append(point[1])

    averaged_points = []
    for x, y_values in x_dict.items():
        avg_y = np.mean(y_values)
        averaged_points.append([x, avg_y])

    return np.array(averaged_points)

def calculate_contact_angle(image, baseline_y, vertical_lines):
    '''
    Calculates the static contact angle of a droplet on a surface based on the contour.

    Args:
        image: Input image of the droplet.
        baseline_y: Baseline y-coordinate for contact angle calculation.

    Returns:
        Average contact angle or None if calculation fails.
    '''
    if image is None or baseline_y < 0 or baseline_y >= image.shape[0]:
        raise ValueError("Invalid image or baseline position.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lower_threshold, upper_threshold = read_last_thresholds()
    if lower_threshold is not None and upper_threshold is not None:
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    else:
        edges = cv2.Canny(blurred, 50, 150)
    # Mask out the area between the vertical lines
    mask = np.ones_like(edges)
    mask[:, vertical_lines[0]:vertical_lines[1]] = 0  # Set the area between the lines to 0
    edges = edges * mask  # Apply mask to the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        print("No contours found!")
        return None

    contour = max(contours, key=cv2.contourArea)
    contour_points = np.squeeze(contour)
    above_baseline_points = contour_points[contour_points[:, 1] < baseline_y]

    if len(above_baseline_points) < 3:
        print("Not enough points found above the baseline.")
        return None

    sorted_points = above_baseline_points[np.argsort(above_baseline_points[:, 0])]
    intersection_points = sorted_points[np.isclose(sorted_points[:, 1], baseline_y, atol=1.0)]

    if len(intersection_points) < 2:
        print("Could not find enough intersection points with the baseline.")
        return None

    left_intersection = intersection_points[0]
    right_intersection = intersection_points[-1]
    width = np.linalg.norm(np.array(right_intersection)-np.array(left_intersection))
    if np.all(sorted_points[0] == left_intersection) and np.all(sorted_points[-1] == right_intersection):
        obtuse = False
        left_points = sorted_points[:201]
        left_points = average_y_for_same_x(left_points)
        right_points = sorted_points[-201:]
        right_points = average_y_for_same_x(right_points)
    else:
        obtuse = True
        distance_threshold = 30
        distances_from_baseline = np.abs(sorted_points[:, 1] - baseline_y)

        left_points = sorted_points[(distances_from_baseline < distance_threshold) & 
                                    (sorted_points[:, 0] <= left_intersection[0])]
        left_points = average_y_for_same_x(left_points)

        right_points = sorted_points[(distances_from_baseline < distance_threshold) & 
                                     (sorted_points[:, 0] >= right_intersection[0])]
        right_points = average_y_for_same_x(right_points)

    def calculate_angle(points, intersection_point):
        poly_coeffs = np.polyfit(points[:, 0], points[:, 1], 2)
        poly = np.poly1d(poly_coeffs)
        slope = np.polyder(poly)(intersection_point[0])
        angle = np.arctan(slope) * (180 / np.pi)
        return angle

    if obtuse:
        left_angle = 180 - calculate_angle(left_points, left_intersection)
        right_angle = 180 + calculate_angle(right_points, right_intersection)
    else:
        left_angle = -calculate_angle(left_points, left_intersection)
        right_angle = calculate_angle(right_points, right_intersection)

    avg_contact_angle = (left_angle + right_angle) / 2

    return left_angle,right_angle,avg_contact_angle,width


def detect_hysteresis_points(widths, left_contact_angles, right_contact_angles):
    """
    Detect advancing and receding contact angles based on changes in width.

    Args:
        widths (list): List of width values.
        left_contact_angles (list): List of left contact angles.
        right_contact_angles (list): List of right contact angles.

    Returns:
        dict: Dictionary containing advancing and receding contact angles.
    """
    # Use peak detection to find advancing and receding points
    peaks, _ = scipy.signal.find_peaks(widths)  # Advancing (expansion)
    troughs, _ = scipy.signal.find_peaks(-np.array(widths))  # Receding (shrinkage)

    advancing_angles = [(left_contact_angles[i], right_contact_angles[i]) for i in peaks]
    receding_angles = [(left_contact_angles[i], right_contact_angles[i]) for i in troughs]

    return {
        "advancing_angles": advancing_angles,
        "receding_angles": receding_angles
    }

# Initialize video input and processing logic
def process_video_for_hysteresis(video_path):
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

    roi = get_cropped_image(first_frame)
    if roi is None:
        raise ValueError("Failed to select ROI.")
    baseline = select_baseline(roi)
    vertical_lines = select_vertical_lines(roi)
    if len(ref_point) == 2:
        x0, y0 = ref_point[0]
        x1, y1 = ref_point[1]
        # Get the coordinates in correct order
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
    
    # Initialize matplotlib for live graph
    plt.ion()
    fig, ax = plt.subplots()
    contact_angles = []
    left_contact_angles =[]
    right_contact_angles =[]
    frame_indices = []
    widths=[]
    step=[]
    line, = ax.plot([], [], label='Contact Angle (°)', color='r')
    ax.set_xlim(0, 100)  # Initial limits, will be updated dynamically
    ax.set_ylim(90, 110)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Contact Angle (°)")
    ax.legend()

    frame_index = 0
    with open('contact_angle_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Left Contact Angle", "Right Contact Angle", "Average Contact Angle", "Width"])

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("End of video stream or error in reading the video file.")
                break
            else:

                # Process the frame for contact angle calculations
                if cropping:
                    frame = frame[y0:y1, x0:x1]
                result = calculate_contact_angle(frame, baseline,vertical_lines)
                if result is None:
                    print("The function returned None.")
                else:
                    left_ca, right_ca, contact_angle, width = result
                if contact_angle is not None:
                    contact_angles.append(contact_angle)
                    left_contact_angles.append(left_ca)
                    right_contact_angles.append(right_ca)
                    widths.append(width)
                    writer.writerow([frame_index,left_ca,right_ca,contact_angle,width])
                # Filter out None values from contact_angles
                filtered_contact_angles = [angle for angle in contact_angles if angle is not None]

                frame_indices.append(frame_index)

                # Update live plot
                line.set_xdata(frame_indices)
                line.set_ydata(contact_angles)

                ax.set_xlim(0, max(100, frame_index))
                # Check if filtered_contact_angles is not empty
                if filtered_contact_angles:
                    ax.set_ylim(min(filtered_contact_angles) - 5, max(filtered_contact_angles) + 5)
                else:
                    print("Error: contact_angles contains only None values.")
                
                plt.pause(0.01)
                frame_index += 1
    # Determination of advancing and receding contact angles
    advancing_ca = None
    receding_ca = None
    for i in range(1,len(widths)):
        step.append(widths[i]-widths[i-1])
    for i in range(len(step)):
        if step[i]==max(step):
            advancing_ca = contact_angles[i]
        if step[i]==min(step):
            receding_ca = contact_angles[i]
    if advancing_ca == None:
        raise ValueError ("Failed to calculate the advancing contact angle. Ensure the needle method algorithm is correctly implemented and that the video provides clear boundary motion.")
    if receding_ca == None:
        raise ValueError ("Failed to calculate the receding contact angle. Ensure the needle method algorithm is correctly implemented and that the video provides clear boundary motion.")
    Contact_Angle_Hysteresis = advancing_ca-receding_ca
        
    # Create the text content
    text_content = (
        f"Advancing Contact Angle: {advancing_ca}\n"
        f"Receding Contact Angle: {receding_ca}\n"
        f"Contact Angle Hysteresis: {Contact_Angle_Hysteresis}\n"
    )

    # File path to save the data
    file_path = "Contact_Angle_Hysteresis.txt"

    # Write the data to a text file
    with open(file_path, "w") as file:
        file.write(text_content)
    

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()



if __name__ == "__main__":
    video_path = r"C:\Users\91982\Videos\hysteresis.mp4"  # Replace with the path to your video file
    process_video_for_hysteresis(video_path)

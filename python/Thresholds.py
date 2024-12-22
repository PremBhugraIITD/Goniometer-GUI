import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Global list to keep track of the contour plot objects
contour_lines = []

def log_thresholds(lower_threshold, upper_threshold):
    """Logs the final threshold values to a text file when the window is closed."""
    with open('thresholds_log.txt', 'w') as file:
        file.write(f"Final Lower Threshold: {lower_threshold}, Final Upper Threshold: {upper_threshold}")

def update(val):
    """Update function to adjust the Canny thresholds and update contour display."""
    # Get the current values from the sliders
    lower_threshold = int(slider_lower.val)
    upper_threshold = int(slider_upper.val)

    # Apply Canny edge detection with the selected thresholds
    edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove the previous contour lines from the plot
    for contour in contour_lines:
        contour.remove()
    contour_lines.clear()

    # Display the updated image with contours
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Canny Thresholds: {lower_threshold} - {upper_threshold}')
    
    # Draw the new contours on the image
    for contour in contours:
        contour_lines.append(ax.plot([point[0][0] for point in contour], 
                                     [point[0][1] for point in contour], color='g', linewidth=2)[0])

    # Redraw the image with contours
    fig.canvas.draw_idle()

    # Pause for a short while to reduce lag
    plt.pause(0.01)

    return lower_threshold, upper_threshold, contours

def on_close(event):
    """Callback function when the window is closed to log the final thresholds."""
    lower_threshold = int(slider_lower.val)
    upper_threshold = int(slider_upper.val)
    log_thresholds(lower_threshold, upper_threshold)
    print("Final thresholds logged.")

if __name__ == "__main__":
    # Load the image
    image = cv2.imread(r"C:\Users\91982\OneDrive\Pictures\Screenshots\Screenshot 2024-12-22 021154.png")  # Replace with your image path

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    # Display the image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Adjust the Canny Thresholds')

    # Add sliders for adjusting the lower and upper threshold values
    ax_slider_lower = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_lower = Slider(ax_slider_lower, 'Lower Threshold', 0, 255, valinit=100, valstep=1)

    ax_slider_upper = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_upper = Slider(ax_slider_upper, 'Upper Threshold', 0, 255, valinit=200, valstep=1)

    # Attach the update function to the sliders
    slider_lower.on_changed(update)
    slider_upper.on_changed(update)

    # Connect the close event to log the final thresholds
    fig.canvas.mpl_connect('close_event', on_close)

    # Show the plot
    plt.show()

## YOLOv8 Vehicle Detection Model 

### Overview
This script demonstrates how to use a trained YOLOv8 model to perform object detection on a single image. The script includes functionality to resize the image while maintaining its aspect ratio to ensure the entire image fits within the display window.

### Prerequisites
- Python 3.11.0
- Ultralytics YOLOv8 library
- OpenCV library
- A trained YOLOv8 model

### Installation
Ensure you have the required libraries installed. You can install them using pip:
```bash
pip install -r requirements.txt
```

### Script Details

#### Import Required Libraries
```python
from ultralytics import YOLO
import cv2
```
- `YOLO` is imported from the Ultralytics library to load and use the YOLOv8 model.
- `cv2` is imported from the OpenCV library for image processing.

#### Define Image Resizing Function
```python
def resize_image(image, max_width=None, max_height=None):
    h, w = image.shape[:2]
    if max_width and w > max_width:
        new_width = max_width
        new_height = int(h * (max_width / w))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    if max_height and h > max_height:
        new_height = max_height
        new_width = int(w * (max_height / h))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image
```
- This function resizes an image while maintaining its aspect ratio.
- Parameters:
  - `image`: The input image to be resized.
  - `max_width`: The maximum allowed width for the resized image.
  - `max_height`: The maximum allowed height for the resized image.

#### Load the YOLOv8 Model
```python
model = YOLO(r'runs\train\exp5\weights\best.pt') #change this as needed
```
- Loads the trained YOLOv8 model from the specified path.

#### Specify the Image Path
```python
image_path = r'path\to\your\image'
```
- Specifies the path to the image on which you want to perform object detection.

#### Load and Resize the Image
```python
image = cv2.imread(image_path)
image = resize_image(image, max_width=800, max_height=800)
```
- Reads the image using OpenCV.
- Resizes the image to ensure it fits within the specified dimensions (800x800).

#### Perform Object Detection
```python
results = model.predict(image)
```
- Runs the YOLOv8 model on the resized image and returns the prediction results.

#### Print and Display the Results
```python
print(results)
annotated_frame = results[0].plot()
cv2.imshow('YOLOv8 Inference', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- Prints the prediction results to the console.
- Plots the detected bounding boxes and labels on the image.
- Displays the annotated image in a window using OpenCV.
- Waits for a key press to close the display window.

### Running the Script
1. Save the script as `test.py`.
2. Run the script using Python:
   ```bash
   python c:/PyProj/newyolo/main/test.py
   ```

### Example Output
- The script will display a window showing the image with detected objects annotated with bounding boxes and labels.

### Troubleshooting
- Ensure the paths to the model weights and image are correct.
- Adjust the `max_width` and `max_height` values as needed for your display or processing requirements.



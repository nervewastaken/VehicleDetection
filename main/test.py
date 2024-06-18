from ultralytics import YOLO
import cv2

# Function to resize image while maintaining aspect ratio
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

# Load the model
model = YOLO(r'C:\PyProj\newyolo\runs\train\exp5\weights\best.pt')

# Path to the image you want to test
image_path = r'C:\Users\krish\Downloads\tstest.png'

# Load the image
image = cv2.imread(image_path)

# Resize the image
image = resize_image(image, max_width=800, max_height=800)

# Make predictions
results = model.predict(image)

# Print the results
print(results)

# Display the results
annotated_frame = results[0].plot()
cv2.imshow('YOLOv8 Inference', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# EXP-12 – PROJECT: Face Detection with Haar Cascades

## Student Details

**Name:** manorajapriyan.l.e  
**Reg No:** 212225040227  

---

# Experiment Name

**Face Detection with Haar Cascades, ROI Segmentation, Handwriting Detection, and Object Detection using MobileNet-SSD**

---

# Aim

To implement image processing and object detection techniques using OpenCV, including:

1. ROI Segmentation in an image using Bitwise AND.
2. Handwriting Detection in an image using Canny Edge Detection and Contours.
3. Object Detection with Labels using the MobileNet-SSD model.

---

# Required Software

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab / Python IDE

---

# Required Files

The following input files are required:

- `plant.png`
- `hand text.jpg`
- `catimg.jpg`

For MobileNet-SSD object detection, the following model files are required:

- `deploy.prototxt`
- `mobilenet_iter_73000.caffemodel`

---

# Procedure

## I) ROI Segmentation in an Image using Bitwise AND

1. Import OpenCV, NumPy, and Matplotlib.
2. Read the input image using `cv2.imread()`.
3. Convert the image from BGR to RGB for displaying.
4. Define the required Region of Interest (ROI) using image coordinates.
5. Create a blank mask having the same size as the original image.
6. Copy the selected ROI into the mask.
7. Perform a Bitwise AND operation between the original image and the mask.
8. Convert the result to RGB.
9. Display the segmented ROI using Matplotlib.

---

## II) Handwriting Detection in an Image

1. Read the handwriting image using OpenCV.
2. Convert the image from BGR to RGB for display.
3. Convert the image into grayscale.
4. Apply Gaussian Blur to reduce image noise.
5. Apply Canny Edge Detection to identify edges.
6. Find contours from the detected edges.
7. Filter contours based on their area.
8. Draw bounding rectangles around the detected regions.
9. Convert the result to RGB.
10. Display the handwriting detection result.

---

## III) Object Detection with Labels using MobileNet-SSD

1. Place the MobileNet-SSD configuration and trained model files in the project folder.
2. Load the `deploy.prototxt` and `.caffemodel` files.
3. Define the object class labels.
4. Read the input image.
5. Obtain the image width and height.
6. Create a blob from the image using `blobFromImage()`.
7. Set the blob as the input to the neural network.
8. Perform forward propagation using `net.forward()`.
9. Check the confidence score of each detected object.
10. Extract the class index and object label.
11. Calculate the bounding box coordinates.
12. Draw bounding boxes and labels around detected objects.
13. Display the final object detection result.

---

# Program

## I) ROI Segmentation in an Image using Bitwise AND

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('plant.png')

# Check whether image is loaded
if image is None:
    print("ERROR: Image not found!")
else:

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display original image
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Define Region of Interest
    # Format: image[startY:endY, startX:endX]
    roi = image[100:420, 200:550]

    # Create a blank mask of the same size
    mask = np.zeros_like(image)

    # Place ROI on the mask
    mask[100:420, 200:550] = roi

    # Perform Bitwise AND
    segmented_roi = cv2.bitwise_and(image, mask)

    # Convert result to RGB
    segmented_roi_rgb = cv2.cvtColor(
        segmented_roi,
        cv2.COLOR_BGR2RGB
    )

    # Display segmented ROI
    plt.imshow(segmented_roi_rgb)
    plt.title("Segmented ROI")
    plt.axis('off')
    plt.show()

II) HANDWRITING DETECTION IN AN IMAGE

Procedure:
1. Read the handwriting image using OpenCV.
2. Convert the image from BGR to RGB.
3. Convert the image into grayscale.
4. Apply Gaussian Blur to reduce noise.
5. Apply Canny Edge Detection to detect edges.
6. Find contours from the edge image.
7. Filter contours based on area.
8. Draw bounding rectangles around the detected regions.
9. Display the handwriting detection result.


PROGRAM:

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('hand text.jpg')

# Check whether image is loaded
if image is None:
    print("ERROR: Image not found!")
else:

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Display original image
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Convert image to grayscale
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        0
    )

    # Apply Canny Edge Detection
    edges = cv2.Canny(
        blurred_image,
        50,
        150
    )

    # Display Canny Edge Detection
    plt.imshow(
        edges,
        cmap='gray'
    )
    plt.title("Canny Edge Detection")
    plt.axis('off')
    plt.show()

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a copy of the original image
    result_image = image.copy()

    # Filter contours and draw bounding boxes
    for contour in contours:

        if cv2.contourArea(contour) > 50:

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Draw rectangle
            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Convert result to RGB
    result_rgb = cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )

    # Display handwriting detection result
    plt.imshow(result_rgb)
    plt.title("Handwriting Detection")
    plt.axis('off')
    plt.show()


III) OBJECT DETECTION WITH LABELS USING MOBILENET-SSD

PROGRAM:

import cv2
import numpy as np
import matplotlib.pyplot as plt

# MobileNet-SSD model files
config_file = 'deploy.prototxt'
weights = 'mobilenet_iter_73000.caffemodel'

# Load pretrained MobileNet-SSD model
net = cv2.dnn.readNet(
    config_file,
    weights
)

print("Model loaded successfully!")

# Object class labels
class_labels = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

# Read input image
image = cv2.imread('catimg.jpg')

# Check whether image is loaded
if image is None:

    print("ERROR: Image not found!")

else:

    # Get image dimensions
    (h, w) = image.shape[:2]

    print("Image loaded successfully!")
    print("Width:", w)
    print("Height:", h)

    # Convert image to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Create blob for DNN processing
    blob = cv2.dnn.blobFromImage(
        image,
        0.007843,
        (300, 300),
        127.5
    )

    # Set input to the neural network
    net.setInput(blob)

    # Perform object detection
    detections = net.forward()

    # Process each detection
    for i in range(detections.shape[2]):

        # Get confidence score
        confidence = detections[0, 0, i, 2]

        # Check confidence threshold
        if confidence > 0.5:

            # Get class index
            index = int(
                detections[0, 0, i, 1]
            )

            # Get object label
            label = class_labels[index]

            # Calculate bounding box
            box = detections[
                0, 0, i, 3:7
            ] * np.array([
                w, h, w, h
            ])

            # Convert coordinates to integers
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box
            cv2.rectangle(
                image_rgb,
                (startX, startY),
                (endX, endY),
                (0, 255, 0),
                2
            )

            # Draw object label
            cv2.putText(
                image_rgb,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    # Display object detection result
    plt.imshow(image_rgb)
    plt.title(
        "Object Detection with MobileNet-SSD"
    )
    plt.axis("off")
    plt.show()
```

## Output


### I) ROI Segmentation in an Image using Bitwise AND

<img width="626" height="535" alt="image" src="https://github.com/user-attachments/assets/b1276c24-8beb-4c7f-b6e0-acf7b9ae8912" />


### II) Handwriting Detection

<img width="481" height="510" alt="image" src="https://github.com/user-attachments/assets/2503f603-2bdd-42d0-b9a8-2f72fe352838" />


### III) Object Detection using MobileNet-SSD

<img width="572" height="510" alt="image" src="https://github.com/user-attachments/assets/43e33a18-665d-4fd6-97c9-09102240cae7" />


RESULT:

The objects in the image were successfully detected using the MobileNet-SSD model. Bounding boxes and corresponding object labels were displayed for detections having a confidence greater than 0.5.


======================================================================


FINAL RESULT:

Thus, handwriting detection using Canny Edge Detection and Contours, and object detection using the MobileNet-SSD model were successfully implemented using OpenCV.

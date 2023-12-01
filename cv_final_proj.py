import cv2

def detect_pedestrians(image_path, output_path):
    # Load the pre-trained pedestrian detector
    pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # Read the input image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect pedestrians in the image
    pedestrians = pedestrian_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the result image
    cv2.imwrite(output_path, image)

# Replace 'your_image_path.jpg' and 'output_result.jpg' with the actual paths
detect_pedestrians('pedestrians2.jpg', 'pedestrians2output.jpg')

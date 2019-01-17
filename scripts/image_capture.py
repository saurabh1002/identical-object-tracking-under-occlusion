# Capture images to generate a training dataset for the Haar Classifier.
# Press 'Spacebar' to capture an image.
# Press 'Esc' to close.
# Generate a dataset each for Positive images and Negative images

import cv2
cam = cv2.VideoCapture(2)
cv2.namedWindow("image_capture")

img_counter = 1

while True:
    ret, frame = cam.read()
    cv2.imshow("image_capture", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        img_name = "../training_data/negative_left/cup_negative_{}.png".format(img_counter)	# Enter the path to store the images captured
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

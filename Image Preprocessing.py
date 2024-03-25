import cv2
import numpy as np
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur which reduces noise
    #blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Thicken countours
    gray1 = gray_image.copy()

    et, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gray_image, contours, -1, (0, 0, 0), 2,cv2.LINE_AA)  # this method of cv2 draws out the contours again on original.
    cv2.rectangle(gray_image, (0, 0), (gray1.shape[1], gray1.shape[0]), (255, 255, 255), 13)

    cv2.imshow("Pre processed image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gray_image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    cv2.imshow("Resized Image ", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return the resized image
    return resized


def segment_characters(word_image):
    # Apply connected component analysis (CCA) to segment characters
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(word_image, connectivity=8)

    # Filter out small components (noise)
    min_area = 10  # Minimum area threshold for a component to be considered as a character
    char_boxes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # Extract bounding box coordinates (x, y, width, height)
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            char_boxes.append((x, y, w, h))
            # Draw bounding box on the original image
            cv2.rectangle(word_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

     # Display the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", word_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return char_boxes

def save_characters_as_images(word_image, char_boxes):
    # Iterate over each bounding box
    for i, box in enumerate(char_boxes):
        x, y, w, h = box
        # Extract the character region from the word image
        character_image = word_image[y:y+h, x:x+w]

        # Save the character image
        character_filename = f"character_{i}.png"
        add_spacing_to_image(character_image,character_filename)


#def predict_character(chaacterimg):
    #predicted_character = recognize_character(preprocessed_character)
def add_spacing_to_image(image, output_path):
    # Calculate the dimensions
    height, width = image.shape[:2]

    # Check if dimensions are odd and adjust if necessary
    if height % 2 != 0:
        height += 1
    if width % 2 != 0:
        width += 1

    # Resize the image if necessary
    if height != image.shape[0] or width != image.shape[1]:
        image = cv2.resize(image, (width, height))

    # Create blank image with double the dimensions
    blank_image = np.full((height * 2, width * 2, 3), 255, dtype=np.uint8)

    # Calculate coordinates for placing the image in the center
    start_y = int(blank_image.shape[0] / 2) - int(round(height / 2))
    end_y = start_y + height
    start_x = int(blank_image.shape[1] / 2) - int(round(width / 2))
    end_x = start_x + width

    # Place the image in the center of the blank image
    blank_image[start_y:end_y, start_x:end_x] = image

    # Write the resulting image to file
    cv2.imwrite(output_path, blank_image)

img = cv2.imread('img.png')
charboxes = segment_characters(img)
#characters = segment_characters(processed_img)
# save_characters_as_images(processed_img,characters)
#  char_boxes = segment_characters(processed_img)
# save_characters_as_images(processed_img, char_boxes)

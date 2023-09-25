import random
import cv2
import numpy as np

# Private function
def intersect_area(box1, box2):
    x0_inter = max(box1[0], box2[0])
    y0_inter = max(box1[1], box2[1])
    x1_inter = min(box1[2], box2[2])
    y1_inter = min(box1[3], box2[3])
    
    return max(x1_inter - x0_inter, 0) * max(y1_inter - y0_inter, 0)


def randomMixture(rgb_masked, background, max_mix=0.5):
    if background.shape[0] < rgb_masked.shape[0] or background.shape[1] < rgb_masked.shape[1]:
        background = cv2.resize(background, (rgb_masked.shape[1]+1, rgb_masked.shape[0]+1))
    
    # random bounding box
    r = random.uniform(0.02, max_mix)
    if random.random() < 0.2:
        random_masked = np.full_like(rgb_masked, fill_value=np.random.randint(low=0, high=150, size=(3,)))
        rgb_masked = cv2.addWeighted(rgb_masked, 1 / (1 + r), random_masked, r / (1 + r), 0)
    else:
        h, w = background.shape[:2]
        h_, w_ = rgb_masked.shape[:2]
        x0, y0 = np.random.randint(low=[0, 0], high=[w - w_, h - h_])
        
        crop_img = background[y0:y0+h_, x0:x0+w_]
        crop_img = crop_img if random.random() < 0.5 else crop_img[:, :, ::-1]
        rgb_masked = cv2.addWeighted(rgb_masked, 1 / (1 + r), crop_img, r / (1 + r), 0)
    
    return rgb_masked


def replace_background(image, mask, background):
    mask = mask.copy()
    # if background > image ==> crop
    if background.shape[0] > image.shape[0] and background.shape[1] > image.shape[1]:
        x = random.randint(0, background.shape[1] - image.shape[1] -1)
        y = random.randint(0, background.shape[0] - image.shape[0] -1)
        background = background[y:y + image.shape[0], x: x + image.shape[1]]
    else:
        background = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Create a binary mask for the object(s)
    mask[mask>0] = 255
    object_mask = mask

    # Invert the binary mask to get the background mask
    background_mask = cv2.bitwise_not(object_mask)

    # Extract the object(s) from the image
    object_region = cv2.bitwise_and(image, image, mask=object_mask)

    # Extract the background region from the background image
    background_region = cv2.bitwise_and(background, background, mask=background_mask)

    # Combine the object(s) and background regions to get the final result
    result = cv2.add(object_region, background_region)

    return result

def rotate_image(img, mask, angle_range):
    angle = random.randint(-angle_range, angle_range)

    # Get the center and size of the image for rotation
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Apply the rotation to the image
    img = cv2.warpAffine(img, rotation_matrix, (width, height))
    mask = cv2.warpAffine(mask, rotation_matrix, (width, height))
    return img, mask
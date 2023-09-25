import os
import cv2

def remove_objects_touching_border(image_folder, mask_folder):
    # List all files in the image and mask folders
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    count = 0
    # Iterate through image files and corresponding mask files
    for image_file in image_files:
        # Find the corresponding mask file
        mask_file = image_file.replace(".jpg", ".png")

        # Check if the mask file exists in the mask folder
        if mask_file in mask_files:
            # Load the image and mask
            # image = cv2.imread(os.path.join(image_folder, image_file))
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)

            # Check if any non-zero pixels touch the image border
            if (mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()):
                # If any non-zero pixels touch the border, remove both the image and mask
                os.remove(os.path.join(image_folder, image_file))
                os.remove(os.path.join(mask_folder, mask_file))
                print(f"Removed {image_file} and {mask_file}")
                count+=1
                print(count)

# Example usage:
image_folder = '/home/jay2/TOMO_new/Unet/data/imgs'
mask_folder = '/home/jay2/TOMO_new/Unet/data/masks'

remove_objects_touching_border(image_folder, mask_folder)
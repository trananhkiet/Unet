import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import glob

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.1,
                image_size= 512):
    net.eval()
    img = BasicDataset.padding_resize(full_img, is_mask=False, training_size=image_size)
    
    # brightness_factor = 1.5  # You can adjust this value as needed
    # # Apply the brightness adjustment
    # img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    
    # cv2.imwrite("test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img = torch.from_numpy(BasicDataset.preprocess(None, img, is_mask=False, training_size = image_size))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.softmax(output, dim=1)
        
        # # Define a threshold for each class (3 classes in this example)
        # thresholds = [0.0, 0.99, 0.0]  # Adjust these thresholds as needed

        # for class_idx, threshold in enumerate(thresholds):
        #     print(threshold)
        #     output[:, class_idx, :, :] = torch.where(output[:, class_idx, :, :] < threshold, torch.tensor(0, dtype=torch.float32), output[:, class_idx, :, :])
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1) 
        else:
            mask = torch.sigmoid(output) > out_threshold
            
    mask = mask[0].long().squeeze().cpu().numpy()
    
    origin_size = max(full_img.size)
    min_size = min(full_img.size[:2])
    mask = mask.astype(np.uint8)
    print(mask.dtype, "shape of mask")
    mask = cv2.resize(mask, (origin_size,origin_size), interpolation = cv2.INTER_NEAREST)
    
    padding_size = (origin_size-min_size)//2
    mask = mask[padding_size: origin_size - padding_size,:]

    return mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--image-size', dest='image_size', help='image size to feed to model', type=int)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    print("shape of mask: ", mask.shape)
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = glob.glob(os.path.join(args.input,"*"))
    output_dir = "_result"
    os.makedirs(output_dir, exist_ok=True)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    
    
    class_colors = {
        0: (0, 0, 0),       # Class 0 - Black
        100: (255, 0, 0),     # Class 1 - Red
        255: (0, 255, 0),     # Class 2 - Green
    }

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           image_size = args.image_size)

        if not args.no_save:
            out_filename = os.path.join(output_dir, os.path.basename(filename).replace("jpg","png"))
            result = mask_to_image(mask, mask_values)
            
            # # Overlap mask
            # opacity = 60  # Adjust as needed
            
            # # Create a mask image with color mapping
            # print(mask.size, "mask size *******")
            # colored_mask = Image.new('RGB', mask.size)
            # for class_id, color in class_colors.items():
            #     class_mask = np.array(mask) == class_id
            #     colored_pixels = np.array(colored_mask)
            #     colored_pixels[class_mask] = color
            #     colored_mask = Image.fromarray(colored_pixels)

            # # Adjust the opacity of the colored mask
            # colored_mask.putalpha(opacity)

            # # Paste the masked image onto the RGB image
            # result = Image.alpha_composite(img.convert('RGBA'), colored_mask)
            
            # Save the result image
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

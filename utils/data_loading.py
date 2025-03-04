import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import random, cv2
from utils.augmentation import randomMixture, replace_background, rotate_image
import glob



def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    
def square_padding(image, is_mask):
        
    # Find the dimensions of the image
    width, height = image.size
    # Calculate the size of the square (maximum of height and width)
    size = max(width, height)

    if is_mask:
        square_image = Image.new('L', (size, size), 0)
    else:
        # Create a new square canvas filled with black (zeros)
        square_image = Image.new('RGB', (size, size), (0, 0, 0))

    # Calculate the position to paste the rectangular image in the center of the square canvas
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    # Paste the rectangular image onto the square canvas
    square_image.paste(image, (x_offset, y_offset))
    
    return square_image

def translate(image, mask, max_shift_x = 20, max_shift_y = 20):
    shift_x = random.randint(-max_shift_x, max_shift_x)
    shift_y = random.randint(-max_shift_y, max_shift_y)
    
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    augmented_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    augmented_mask = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))
    
    return augmented_image, augmented_mask

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', training_size = 512):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mask_suffix = mask_suffix
        self.training_size = training_size

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, img, is_mask, training_size):
        newW, newH = training_size, training_size
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
    
    @staticmethod
    def padding_resize(pil_img, is_mask, training_size):
        
        newW, newH = training_size, training_size
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = square_padding(pil_img, is_mask)
        
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        
        return img
    
    def crop_image_and_mask(self, image, mask):
        mask_array = np.array(mask.copy())
        image_array = np.array(image.copy())
        # mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)

        mask_array[mask_array >0] = 255
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)> 0:
            contour= contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            self.have_label = True
        else:
            w = random.randint(int(mask_array.shape[0]*0.2), int(mask_array.shape[0]*0.5))
            h = w
            x = random.randint(0, mask_array.shape[1] - w - 1)
            y = random.randint(0, mask_array.shape[0] - h -1)
            self.have_label = True
        
        if x > 30 and y > 30 and x < mask_array.shape[1] - 31 and y < mask_array.shape[0] - 31:
            translation_value = random.randint(0, 8) if w < 50 else random.randint(0, 30)
            translation_value2 = random.randint(0, 8) if w < 50 else random.randint(0, 30)
        else:
            translation_value = translation_value2 = 0
            
        mask_array = np.array(mask.copy())
        mask_array = mask_array[y - translation_value: y + h + translation_value2 , x - translation_value: x + w + translation_value2]
        image_array = image_array[y - translation_value: y + h + translation_value2 , x - translation_value: x + w+ translation_value2]
        
        image = Image.fromarray(image_array)
        mask = Image.fromarray(mask_array)
        return image, mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        
        # Function to crop mask area
        img, mask = self.crop_image_and_mask(img, mask)
        origin_size = mask.size

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # Padding and then resize image
        img = self.padding_resize(img, is_mask=False, training_size=self.training_size)
        mask = self.padding_resize(mask, is_mask=True, training_size=self.training_size)
     
        # Augmentation
        if random.random() < 0.8 and self.have_label and origin_size[0] > 50:
            background_path = random.choice(glob.glob("/home/jay2/TOMO_new/Raw_data/Background_adding/rgb/*"))
            background_image = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
            
            if random.random()<0.5:
                img = replace_background(img, mask, background_image)
                img = randomMixture(img, background_image, max_mix= 0.14)
            else:
                img = randomMixture(img, background_image, max_mix= 0.14)
                if random.random() < 0.2:
                    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    background_image = np.full((512, 512, 3), random_color, dtype=np.uint8)
                    img = replace_background(img, mask, background_image)
        
        if random.random() < 0.7:
            img, mask = rotate_image(img, mask, angle_range=180)
        
        if random.random() < 0.2:
            img = np.flipud(img)
            mask = np.flipud(mask)
        if random.random() < 0.2:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        if random.random() <0.3:
            if origin_size[0] > 70:
                img = cv2.GaussianBlur(img, (5, 5), 0)
        
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.25,1.1)  # You can adjust this value as needed
            # Apply the brightness adjustment
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if random.random() < 0.45:
            img, mask = translate(img, mask, max_shift_x = 35, max_shift_y= 35)
            
        # save_img = np.concatenate((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)), axis = 0)
        # cv2.imwrite("/home/jay2/TOMO_new/Unet/checkpoints/example_image/"+ str(random.randint(0,1000)) +".png",save_img)
        
        img = self.preprocess(self.mask_values, img, is_mask=False, training_size= self.training_size)
        mask = self.preprocess(self.mask_values, mask, is_mask=True, training_size= self.training_size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, training_size = 512):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', training_size = training_size)

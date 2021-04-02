import torch
import numpy as np
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numbers
import math
import logging


class GroupBBoxRandomCrop(object):
    def __init__(self, size=(256, 128), pad=0):
        self.worker_pad = transforms.Pad(pad)
        self.worker_RandomCrop = transforms.RandomCrop(size)

    def __call__(self, img_group):
        padded = [self.worker_pad(img) for img in img_group]

        return [self.worker_RandomCrop(img) for img in padded]


class GroupRandomCrop(object):
    def __init__(self, size=(224, 224), pad=0):
        self.worker_RandomCrop = transforms.RandomCrop(size, padding=pad)

    def __call__(self, img_group):
        return [self.worker_RandomCrop(img) for img in img_group]


class GroupCornerCrop(object):
    """Corner crop the given PIL.Image group.

    Args:
        size: size of the smaller edge

    Return:
        a list of cropped PIL.Image
    """
    def __init__(self, size):
        self.size = size
        self.worker = transforms.FiveCrop(size)

    def __call__(self, img_group):

        w, h = img_group[0].size
        crop_h, crop_w = self.size
        if crop_w > w or crop_h > h:
            raise ValueError(
                "Requested cropsize {} is bigger than inputsize {}".format(
                    self.size, (h, w)))
        _img_group = []
        for img in img_group:
            position = self._randomize_position  # 'c' or 'tl' or 'tr' ...
            if position == 'tl':
                _img_group.append(img.crop((0, 0, crop_w, crop_h)))
            elif position == 'tr':
                _img_group.append(img.crop((w - crop_w, 0, w, crop_h)))
            elif position == 'bl':
                _img_group.append(img.crop((0, h - crop_h, crop_w, h)))
            elif position == 'br':
                _img_group.append(img.crop((w - crop_w, h - crop_h, w, h)))
            elif position == 'c':
                _img_group.append(transforms.CenterCrop(
                    self.size)(img))
        return _img_group

    def _randomize_position(self):
        _positions = ['c', 'tl', 'tr', 'bl', 'br']
        return _positions[random.randint(0, 4)]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


def group_random_flip(images, is_flow=False):
    flipped = False
    v = random.random()
    if v < 0.5:
        flipped = True
        images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        if is_flow:
            for i in range(0, len(images), 2):
                # invert flow pixel values when flipping
                images[i] = ImageOps.invert(images[i])

    return images, flipped


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a
     probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) \
            else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            True, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) \
            else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group]
        ret_img_group = [img.resize(
            (self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3
                  else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3
                  else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0)
     of the original size and and a random aspect ratio of 3/4 to 4/3
     of the original aspect ratio. This is popularly used to train the
     Inception networks

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR

    Return:
        a list of cropped PIL.Image
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size),
                                 self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class GroupColorJitter(object):
    def __init__(self, colorjitter):
        logging.info("Using ColorJitter!!!!!")
        self.colorjitter = colorjitter
        self.transform = transforms.ColorJitter(*colorjitter)

    def __call__(self, img_group):
        # self.worker = transforms.ColorJitter.get_params(*self.colorjitter)
        brightness, contrast = self.transform.brightness, self.transform.contrast
        saturation, hue = self.transform.saturation, self.transform.hue
        self.worker = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)

        return [self.worker(img) for img in img_group]


# class GroupColorJitter(object):
#     def __init__(self, colorjitter):
#         logging.info("Using ColorJitter!!!!!")
#         self.colorjitter = colorjitter

#     def __call__(self, img_group):
#         self.worker = transforms.ColorJitter.get_params(*self.colorjitter)

#         return [self.worker(img) for img in img_group]


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec is None:
            eig_vec = torch.Tensor([
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ])
        if eig_val is None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class GroupRandomErasing(object):
    """docstring for GroupRandomErase
    """

    def __init__(self, arg=[1, 0.02, 0.4, 0.3]):
        print("Using random_erasing!!!!")
        super(GroupRandomErasing, self).__init__()
        self.arg = arg
        self.worker = RandomErasing(*arg)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation
    by Zhong et al.

    Args:
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[152, 142, 127]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            # convert PIL Image object to Numpy Array
            # PIL Image size is (224, 224) [W, H]->[x, y]->[col, row]
            # numpy array shape is (224, 224, 3) [W, H, C]
            _img = np.array(img)
            area = _img.shape[0] * _img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # print("h is {}, w is {}".format(str(h), str(w)))
            if w < _img.shape[0] and h < _img.shape[1]:
                x1 = random.randint(0, _img.shape[0] - w)
                y1 = random.randint(0, _img.shape[1] - h)

                if len(_img.shape) == 2:
                    # for gray image
                    _img[x1:x1 + w, y1:y1 + h, 0] = self.mean[0]
                else:
                    # for RGB image
                    _img[x1:x1 + w, y1:y1 + h, 0] = self.mean[0]
                    _img[x1:x1 + w, y1:y1 + h, 1] = self.mean[1]
                    _img[x1:x1 + w, y1:y1 + h, 2] = self.mean[2]

                # convert Numpy Array back to PIL Image
                img = Image.fromarray(_img)
                return img

        return img

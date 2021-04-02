"""
    Some useful operations on PIL images.
    Such as, save, load, mean, std ...

"""
import io
import numpy as np
import skimage.io
from PIL import Image
import torch
import torchvision.transforms as transforms
from mmf.datasets.builders.sensetimeVAD.spatial_transforms import *

from springvision import Normalize, ToTensor, Resize, \
    Compose, CenterCrop, AdjustGamma, \
    RandomCrop, ToGrayscale, KestrelDevice
import kestrel as ks
# IMG_FILE_EXTs = ['jpg', 'jpeg', 'png', 'bmp']

torch_transforms_dict = {
    'resize': GroupScale,
    'center_crop': GroupCenterCrop,
    'random_crop': GroupRandomCrop,
    'compose': transforms.Compose
}

kestrel_transforms_dict = {
    'resize': Resize,
    'center_crop': CenterCrop,
    'random_crop': RandomCrop,
    'normalize': Normalize,
    'to_tensor': ToTensor,
    'adjust_gamma': AdjustGamma,
    'to_grayscale': ToGrayscale,
    'compose': Compose
}


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return [img]


def kestrel_loader(image_data):
    input_frame = ks.Frame()
    input_frame.create_from_mem(image_data, len(image_data))
    if ks.Device().mem_type() == ks.KESTREL_MEM_DEVICE:
        input_frame = input_frame.upload()

    spatial_transform = kestrel_transforms_dict['compose']([
        kestrel_transforms_dict['to_tensor']()
    ])
    image_data = spatial_transform(input_frame)
    image_data = image_data.permute(1, 2, 0).numpy()

    return [Image.fromarray(image_data.astype(np.uint8))]
    # return [input_frame]


def frame_to_numpy(frame):
    """
    frame should be downloaded to host
    """
    # get bytes
    buf = ks.Buffer(frame._data.contents.buffer).data()
    # construct numpy
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = arr.reshape(frame.height, frame.width, 3)
    if frame.fmt == ks.KESTREL_VIDEO_RGB:
        # opencv BGR
        img = img[:, :, ::-1].copy()
    return img  # .copy()


def numpy_to_frame(image):
    frame = ks.Frame()
    frame.create_from_numpy(image)
    frame._numpy_backup = image  # TODO: fix bug
    # kestrel frame does not copy data.
    # After numpy image freed, frame will become black.

    return frame


def get_detected_roi(seg_imgs, region):
    roi_imgs = list()
    for seg_img in seg_imgs:
        img = np.asarray(seg_img, dtype=np.uint8)
        new_img = get_roi_img_by_region(img, region)
        new_img = Image.fromarray(new_img)
        roi_imgs.append(new_img)
    return roi_imgs


def get_roi_img_by_region(img, region):

    h, w, c = img.shape
    left = int(region[0])
    top = int(region[1])
    right = int(min(w, region[2]))
    bottom = int(min(h, region[3]))

    new_img = img[top:bottom, left:right]
    return new_img


def resize_random_crop(image, scale_size, input_size, center_bbox=None, image_reader_type='pil'):
    '''
    Random crop the bbox around the center-bbox.
    params:
        image: origin frame
        center_bbox: center_bbox inside the origin frame
    return:
        image: cropped image
        factor: resize factor
        center_bbox_trans: transform bbox of center_bbox
        trans_: center_bbox correspoinding to the center
    '''
    scale_size = scale_size
    input_size = input_size
    width = image[0].size[0]
    height = image[0].size[1]
    if width < height:
        factor = float(scale_size) / width
        new_height = int(factor * height)
        resize_ = (scale_size, new_height)
    else:
        factor = float(scale_size) / height
        new_width = int(factor * width)
        resize_ = (new_width, scale_size)

    crop_size = (input_size, input_size)
    if center_bbox is not None:
        # resize and roi center-based random crop
        if image_reader_type == 'pil':
            spatial_transform_1 = torch_transforms_dict['compose']([
                torch_transforms_dict['resize']((resize_[1], resize_[0]))
            ])
            image = spatial_transform_1(image)
        else:
            spatial_transform_1 = kestrel_transforms_dict['compose']([
                kestrel_transforms_dict['resize']((resize_[1], resize_[0])),
                kestrel_transforms_dict['to_tensor']()
            ])
            image = numpy_to_frame(np.array(image[0]))
            image = spatial_transform_1(image)
            image = image.permute(1, 2, 0).numpy()
            image = [Image.fromarray(image.astype(np.uint8))]

        center_bbox = center_bbox * factor
        image, center_bbox_trans, trans_ = \
            bbox_orient_crop(image[0], center_bbox, input_size)

        if image_reader_type == 'pil':
            spatial_transform_2 = torch_transforms_dict['compose']([
                torch_transforms_dict['random_crop'](size=crop_size, pad=10)
            ])
            image = spatial_transform_2(image)
        else:
            spatial_transform_2 = kestrel_transforms_dict['compose']([
                kestrel_transforms_dict['random_crop'](size=crop_size, padding=10),
                kestrel_transforms_dict['to_tensor']()
            ])
            image = numpy_to_frame(np.array(image[0]))
            image = spatial_transform_2(image)
            image = image.permute(1, 2, 0).numpy()
            image = [Image.fromarray(image.astype(np.uint8))]
    else:
        # resize and random crop
        if image_reader_type == 'pil':
            spatial_transform = torch_transforms_dict['compose']([
                torch_transforms_dict['resize']((resize_[1], resize_[0])),
                torch_transforms_dict['random_crop'](size=crop_size)
            ])
            image = spatial_transform(image)
        else:
            spatial_transform = kestrel_transforms_dict['compose']([
                kestrel_transforms_dict['resize']((resize_[1], resize_[0])),
                kestrel_transforms_dict['random_crop'](size=crop_size),
                kestrel_transforms_dict['to_tensor']()
            ])
            image = numpy_to_frame(np.array(image[0]))
            image = spatial_transform(image)
            image = image.permute(1, 2, 0).numpy()
            image = [Image.fromarray(image.astype(np.uint8))]

        center_bbox_trans = None
        trans_ = None

    return image, factor, center_bbox_trans, trans_


def resize_center_crop(image, scale_size, input_size, center_bbox=None, image_reader_type='pil'):
    width = image[0].size[0]
    height = image[0].size[1]
    if width < height:
        factor = float(scale_size) / width
        new_height = int(factor * height)
        resize_ = (scale_size, new_height)
    else:
        factor = float(scale_size) / height
        new_width = int(factor * width)
        resize_ = (new_width, scale_size)
    crop_size = (input_size, input_size)

    if center_bbox is not None:
        # resize and region-center based crop
        if image_reader_type == 'pil':
            spatial_transform = torch_transforms_dict['compose']([
                torch_transforms_dict['resize']((resize_[1], resize_[0]))
            ])
            image = spatial_transform(image)
        else:
            spatial_transform_1 = kestrel_transforms_dict['compose']([
                kestrel_transforms_dict['resize']((resize_[1], resize_[0])),
                kestrel_transforms_dict['to_tensor']()
            ])
            image = numpy_to_frame(np.array(image[0]))
            image = spatial_transform_1(image)
            image = image.permute(1, 2, 0).numpy()
            image = [Image.fromarray(image.astype(np.uint8))]

        center_bbox = center_bbox * factor
        image, center_bbox_trans, trans_ = \
            bbox_orient_crop(image[0], center_bbox, input_size)
    else:
        if image_reader_type == 'pil':
            spatial_transform = torch_transforms_dict['compose']([
                torch_transforms_dict['resize']((resize_[1], resize_[0])),
                torch_transforms_dict['center_crop'](crop_size)
            ])
            image = spatial_transform(image)
        else:
            spatial_transform = kestrel_transforms_dict['compose']([
                kestrel_transforms_dict['resize']((resize_[1], resize_[0])),
                kestrel_transforms_dict['center_crop'](crop_size),
                kestrel_transforms_dict['to_tensor']()
            ])
            image = numpy_to_frame(np.array(image[0]))
            image = spatial_transform(image)
            image = image.permute(1, 2, 0).numpy()
            image = [Image.fromarray(image.astype(np.uint8))]

        center_bbox_trans = None
        # trans_ = None
        trans_ = [(resize_[0] - input_size) / (2 * factor), (resize_[1] - input_size) / (2 * factor),
                  input_size / factor, input_size / factor]
        trans_ = [trans_[0], trans_[1], trans_[0] + trans_[2], trans_[1] + trans_[3]]
    return image, factor, center_bbox_trans, trans_


def bbox_orient_crop(image, center_bbox, input_size):
    """
    Crop the bbox around the center-bbox.
    params:
        image: resized image
        center_bbox: center_bbox inside the resized image
    return:
        [image]: cropped image
        center_bbox_trans: relative value of center to resize image
        (crop_x1, crop_y1, crop_x1, crop_y1): center bbox coordinate in the image.
    """
    width = image.size[0]
    height = image.size[1]
    final_size = input_size

    x1, y1, x2, y2 = center_bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    crop_x1, crop_y1, crop_x2, crop_y2 = \
        (center_x - final_size // 2, center_y - final_size // 2,
            center_x + final_size // 2, center_y + final_size // 2)
    if crop_x1 <= 0:
        crop_x1, crop_x2 = 0, final_size
    if crop_y1 <= 0:
        crop_y1, crop_y2 = 0, final_size
    if crop_x2 >= width:
        crop_x1, crop_x2 = width - final_size - 1, width - 1
    if crop_y2 >= height:
        crop_y1, crop_y2 = height - final_size - 1, height - 1

    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
    image = image.crop(crop_box)
    center_bbox_trans = [x1 - crop_x1, y1 - crop_y1,
                         x2 - crop_x1, y2 - crop_y1]
    assert image.size == (final_size, final_size), image.size
    return [image], center_bbox_trans, (crop_x1, crop_y1, crop_x2, crop_y2)


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.

    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def save(imgPath, img):
    """
    Save an image to the path.

    Input
      imgPath  -  image path
      img      -  an image with type np.float32 in range [0, 1], h x w x nChan
    """
    skimage.io.imsave(imgPath, img)


def load(imgPath, color=True, verbose=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Input
      imgPath  -  image path
      color    -  flag for color format. True (default) loads as RGB
                  while False loads as intensity
                  (if image is already grayscale).

    Output
      image    -  an image with type np.float32 in range [0, 1]
                    of size (h x w x 3) in RGB or
                    of size (h x w x 1) in grayscale.
    """
    # load
    try:
        img0 = skimage.io.imread(imgPath)
        img = skimage.img_as_float(img0).astype(np.float32)

    except KeyboardInterrupt as e:
        raise e
    except:     # noqa
        if verbose:
            print('unable to open img: {}'.format(imgPath))
        return None

    # color channel
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))

    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def oversample(img0s, h=224, w=224, view='mul'):
    """
    Crop images as needed. Inspired by pycaffe.

    Input
      img0s  -  n0 x, h0 x w0 x k0
      h      -  crop height, {224}
      w      -  crop width, {224}
      view   -  view, 'sin' | 'flip' | {'mul'}
                  'sin': center crop (m = 1)
                  'flip': center crop and its mirrored version (m = 2)
                  'mul': four corners, center,
                         and their mirrored versions (m = 10)

    Output
      imgs   -  crops, (m n0) x h x w x k
    """
    # dimension
    n0 = len(img0s)
    im_shape = np.array(img0s[0].shape)
    crop_dims = np.array([h, w])
    im_center = im_shape[:2] / 2.0
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])

    # make crop coordinates
    if view == 'sin':
        # center crop
        crops_ix = np.empty((1, 4), dtype=int)
        crops_ix[0] = np.tile(im_center, (1, 2)) + np.concatenate([
            -crop_dims / 2.0, crop_dims / 2.0
        ])

    elif view == 'flip':
        # center crop + flip
        crops_ix = np.empty((1, 4), dtype=int)
        crops_ix[0] = np.tile(im_center, (1, 2)) + np.concatenate([
            -crop_dims / 2.0, crop_dims / 2.0
        ])
        crops_ix = np.tile(crops_ix, (2, 1))

    elif view == 'mul':
        # multiple crop
        crops_ix = np.empty((5, 4), dtype=int)
        curr = 0
        for i in h_indices:
            for j in w_indices:
                crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
                curr += 1
        crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
            -crop_dims / 2.0, crop_dims / 2.0
        ])
        crops_ix = np.tile(crops_ix, (2, 1))
    m = len(crops_ix)

    # extract crops
    crops = np.empty((m * n0, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in img0s:
        for crop in crops_ix:
            try:
                crops[ix] = im[crop[0]: crop[2], crop[1]: crop[3], :]
            except ValueError:
                import pdb
                pdb.set_trace()
            ix += 1

        # flip for mirrors
        if view == 'flip' or view == 'mul':
            m2 = m / 2
            crops[ix - m2: ix] = crops[ix - m2: ix, :, ::-1, :]

    return crops

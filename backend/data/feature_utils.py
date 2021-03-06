import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import jittor as jt
import data.jimm as jimm
from skimage.transform import resize


def get_img(path):
    return Image.open(path).convert('RGB')

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    if im_max - im_min != 0:
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    else:
        print('dividend=0')
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Normalize
    gradient = gradient - gradient.min()
    if gradient.max() != 0:
        gradient /= gradient.max()
    else:
        print('dividend=0')
    # Save image
    path_to_file = os.path.join('./results', file_name + '.jpg')
    save_image(gradient, path_to_file)

def normalize(gradient):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    # Normalize
    gradient = gradient - gradient.min()
    if gradient.max() != 0:
        gradient /= gradient.max()
    else:
        print('dividend=0')
    gradient = (gradient*255).astype(np.uint8)
    return gradient

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('./results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)

def get_class_activation_image(org_img, activation_map):
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')

    return np.asarray(heatmap_on_image.convert("RGB")).transpose(2,0,1)

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    if gradient.max() != 0:
        pos_saliency = (np.maximum(0, gradient) / gradient.max())
    else:
        pos_saliency = np.maximum(0, gradient)
        print('dividend=0')
    if gradient.min() != 0:
        neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    else:
        neg_saliency = np.maximum(0, -gradient)
        print('dividend=0')
    return pos_saliency, neg_saliency

def extract_target_class(img_path):
    return int(img_path.split('/')[-2])

def handle_seq(seq):
    res = []
    for i in seq.children():
        if isinstance(i, jt.nn.Sequential):
            res.extend(handle_seq(i))
        elif isinstance(i, jimm.resnet.Bottleneck):
            for j in i.children():
                if isinstance(j, jt.nn.Sequential):
                    res.extend(handle_seq(j))
                else:
                    res.append(j)
        else:
            res.append(i)  
    return res

def unpack_children(model):
    """
    only for jimm.resnet
    """
    children = model.children()
    res = []
    for i in children:
        if isinstance(i, jt.nn.Sequential):
            res.extend(handle_seq(i))
        else:
            res.append(i)
    return res

def activation_to_mask(feature_map, size, r):
    _size = feature_map.shape
    max_index = [
        int(d) for d in np.unravel_index(np.argmax(feature_map), _size)
    ]
    center = [
        int((x + 0.5) / _size[i] * size[0]) for i, x in enumerate(max_index)
    ]
    mask = np.zeros(size, dtype=np.float32)
    mask[max(center[0] - r, 0):min(center[0] + r, size[0]),
         max(center[1] - r, 0):min(center[1] + r, size[1])] = 1.0
    return mask

def generate_discrepancy_map(feature_map, original_image, **kwargs):
    """get discrepancy_map

    Args:
        feature_map (list): output of certain filter of a layer
        original_image ()
        optional in kwargs :
            sparsity (float): control the sparsity of the highlight region;
            radius (int): only highlight area around the max value position with specific radius;
            threshold_scale (float): highlight the positions that values of them are larger this threshold. 

    Returns:
        list: discrepancy map of original image
    """
    feature_map = np.array(feature_map)
    max_value = np.max(feature_map)
    feature_map = feature_map / max_value
    segment_size = feature_map.shape

    if kwargs.get('sparsity') is not None:
        # binary search for threshold
        mask = resize(feature_map, segment_size)
        tgt_sp = kwargs['sparsity']
        th_l = 0
        th_r = 1
        while (th_r - th_l) > 1e-3:
            th_mid = (th_l + th_r) / 2
            sp = (mask < th_mid).mean()
            #print(sp, th_mid)
            if sp < tgt_sp:
                th_l = th_mid
            else:
                th_r = th_mid

        # binarize the mask,
        mask[mask <= th_l] = 0.0
        mask[mask > th_l] = 1.0
    elif kwargs.get('radius') is not None:
        radius = kwargs['radius']
        mask = activation_to_mask(feature_map, segment_size, radius)
    else:
        mask = resize(feature_map, segment_size)
        th = kwargs['threshold_scale']
        # binarize the mask
        mask[mask <= th] = 0.0
        mask[mask > th] = 1.0

    original_image = np.squeeze(original_image)
    original_image = original_image / 255.0
    original_image = resize(original_image, segment_size)  #0-255

    img_mask = (original_image * mask[:, :, None] * 255.0).astype('uint8').tolist()

    return img_mask


if __name__ == '__main__':
    pass
import numpy as np
import torch
from PIL import Image
from skimage.transform import rotate

def get_bbox_valid(joints, rescale):
    # we keep the images even if the joints are not visible

    # Get bbox using keypoints
    # valid_j = []
    # joints = np.copy(joints)
    # for j in joints:
    #     if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
    #         continue
    #     else:
    #         valid_j.append(j)

    # if len(valid_j) < 1:
    #     return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    # joints = np.array(valid_j)

    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

    scale *= rescale
    return center, scale, 22, bbox

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # allow complete black image as condition, if all the joints are outside the image
    # the model should infer the pose from the context
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        return img, np.zeros((res[0], res[1], img.shape[2])).astype(np.uint8)
    # If the human is too large in the image, also return black image
    if br[1] - ul[1] > 3 * img.shape[0] or br[0] - ul[0] > 3 * img.shape[1]:
        return img, np.zeros((res[0], res[1], img.shape[2])).astype(np.uint8)

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]

    if not rot == 0:
        # Remove padding
        new_img = rotate(new_img, rot)  # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    
    # Use PIL to resize instead

    # resize image
    # new_img = resize(new_img, res)  # scipy.misc.imresize(new_img, res)
    new_img = new_img.astype(np.uint8)
    new_img = Image.fromarray(new_img, mode="RGB")
    new_img = new_img.resize(res, Image.BILINEAR)
    new_img = np.array(new_img)

    return img, new_img

def j2d_processing(j2d, center, scale, res):
    """
    Get the 2D keypoints in the cropped image space.
    """
    num_joints = j2d.shape[0]
    for i in range(num_joints):
        j2d[i, :2] = transform(j2d[i, :2] + 1, center, scale, res)
    # convert to normalized coordinates
    j2d[:, :2] = 2 * j2d[:, :2] / res - 1
    j2d = j2d.astype(np.float32)
    return j2d
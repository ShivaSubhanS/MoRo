
import numpy as np

"""
Adapted from https://github.com/sanweiliti/LEMO/blob/a4c082555e62a4be6f0856ec8ef7516ad85eb911/temp_prox/misc_utils.py
"""
def get_keypoint_mapping(model_type="smplx"):
    """ Returns the indices of the permutation that maps SMPL to OpenPose25 + Extra

        Parameters
        ----------
        model_type: str, optional
            The type of keypoint template that is used. The default mapping
            returned is for the SMPLX model
            We only support "smpl24" and "smplx"
            The "smpl24" keypoint format is given here: https://github.com/open-mmlab/mmhuman3d/blob/8c95747175f8c4bd76d5e0be7be244f8a9a4a6de/mmhuman3d/core/conventions/keypoints_mapping/smpl.py
            Note that this is not the SMPL keypoints...
            It is used in the real image datasets downloaded from BEDLAM

        The output format is OpenPose25 + Extra keypoints
        Returns
        -------
        mapping: np.array
            The mapping from SMPL to OpenPose25 + Extra keypoints.
        valid: np.array
            The valid keypoints in the mapping.
    """
    if model_type == 'smpl24':
        mapping_openpose = np.zeros(25, dtype=np.int32)
        mapping_openpose[0] = -5
        mapping_openpose[15:19] = [-3, -4, -1, -2]
        mapping_extra = np.arange(19, dtype=np.int32)

        valid_openpose = np.zeros(25, dtype=np.bool_)
        valid_openpose[0] = True
        valid_openpose[15:19] = True
        valid_extra = np.array([True] * 19, dtype=np.bool_)

        mapping = np.concatenate([mapping_openpose, mapping_extra])
        valid = np.concatenate([valid_openpose, valid_extra])
    elif model_type == 'smplx':
        # ex: body_mapping[0]=55: smplx joint 55 = openpose joint 0
        mapping_openpose = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                    8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                    63, 64, 65], dtype=np.int32)   # len of 25
        mapping_extra = np.array([0] * 19, dtype=np.int32)
        mapping = np.concatenate([mapping_openpose, mapping_extra])
        valid = np.array([True] * 25 + [False] * 19, dtype=np.bool_)
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))
    return mapping, valid

def smpl_to_openpose(model_type='smplx', use_hands=False, use_face=False,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            # return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
            #                  7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            #                 dtype=np.int32)
            # NOTE: this is a rough mapping, as SMPL head does not correspond exactly to OpenPose nose, but it's enough for cropping 
            return np.array([15, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 10, 10, 31, 11, 11, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            # ex: body_mapping[0]=55: smplx joint 55 = openpose joint 0
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)   # len of 25
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)   # len of 51
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

        
from typing import Tuple
import torch
"""
Utils for cropping regions of interest based on openpose25 keypoints.
Adapted from TokenHMR https://github.com/saidwivedi/TokenHMR/blob/main/tokenhmr/lib/datasets/utils.py
"""
def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25+0, 25+1, 25+4, 25+5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height

def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height

def crop_torso_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the torso.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nontorso_body_keypoints = [0, 3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 4, 5, 6, 7, 10, 11, 13, 17, 18]]
    keypoints_2d[nontorso_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the right arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightarm_body_keypoints = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the left arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftarm_body_keypoints = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_legs_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the legs.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonlegs_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18] + [25 + i for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]
    keypoints_2d[nonlegs_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the right leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + [25 + i for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the left leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24] + [25 + i for i in [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def full_body(keypoints_2d: np.array) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0).sum() == len(body_keypoints)

def upper_body(keypoints_2d: np.array):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25+8, 25+9, 25+12, 25+13, 25+17, 25+18]
    return ((keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0)\
       and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)

def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.array): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale

def extreme_cropping(center, scale, keypoints_2d: np.array) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center: (Tuple[float, float]): Tuple containing x and y coordinates of the bounding box center.
        scale (float): Scale factor for the bounding box.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    center_x, center_y = center
    width = height = scale * 200
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)

    THRESH = 4
    if width < THRESH or height < THRESH:
        pass
    else:
        center = np.array([center_x, center_y]) 
        scale = max(width, height) / 200.0
    return center, scale

def extreme_cropping_aggressive(center, scale, keypoints_2d: np.array) -> Tuple:
    """
    Perform aggressive extreme cropping
    Args:
        center (Tuple[float, float]): Tuple containing x and y coordinates of the bounding box center.
        scale (float): Scale factor for the bounding box.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    center_x, center_y = center
    width = height = scale * 200
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.3:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.5:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.7:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_legs_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_rightleg_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftleg_only(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
    THRESH = 4
    if width < THRESH or height < THRESH:
        pass
    else:
        center = np.array([center_x, center_y]) 
        scale = max(width, height) / 200.0
    return center, scale

def crop_bbox(center, scale, keypoints_2d, extreme_crop_lvl=2):
    if extreme_crop_lvl == 0:
        return center, scale 
    elif extreme_crop_lvl == 1:
        return extreme_cropping(center, scale, keypoints_2d)
    elif extreme_crop_lvl == 2:
        return extreme_cropping_aggressive(center, scale, keypoints_2d)
    else:
        raise ValueError(f'Unknown extreme crop level: {extreme_crop_lvl}')

def crop_bbox_seq(center_list, scale_list, keypoints_2d_list, extreme_crop_lvl=2, ratio=0.6):
    """
    Crop a sequence of bounding boxes.
    Args:
        center_list (list): List of tuples containing x and y coordinates of the bounding box centers.
        scale_list (list): List of scale factors for the bounding boxes.
        keypoints_2d_list (list): List of arrays of shape (N, 3) containing 2D keypoint locations.
    Returns:
        list: List of tuples containing the new bounding box centers and scales.
    """
    # get crop mode, fixed across the sequence
    p = torch.rand(1).item()
    if extreme_crop_lvl == 0:
        return center_list, scale_list
    elif extreme_crop_lvl == 1:
        if p < 0.7:
            crop_fn = crop_to_hips
        elif p < 0.9:
            crop_fn = crop_to_shoulders
        else:
            crop_fn = crop_to_head
    elif extreme_crop_lvl == 2:
        if p < 0.2:
            crop_fn = crop_to_hips
        elif p < 0.3:
            crop_fn = crop_to_shoulders
        elif p < 0.4:
            crop_fn = crop_to_head
        elif p < 0.5:
            crop_fn = crop_torso_only
        elif p < 0.6:
            crop_fn = crop_rightarm_only
        elif p < 0.7:
            crop_fn = crop_leftarm_only
        elif p < 0.8:
            crop_fn = crop_legs_only
        elif p < 0.9:
            crop_fn = crop_rightleg_only
        else:
            crop_fn = crop_leftleg_only
    
    crop_center_list = center_list.copy()
    crop_scale_list = scale_list.copy()
    num_frames = center_list.shape[0]

    crop_ratio = torch.rand(1).item() * (1 - ratio) + ratio
    # Create a mask where int(crop_ratio*num_frames) frames will be cropped
    num_crop_frames = round(crop_ratio * num_frames)
    num_crop_frames = max(num_crop_frames, 1)  # Ensure at least one frame is cropped
    crop_mask = torch.zeros(num_frames, dtype=torch.bool)
    if num_crop_frames > 0:
        crop_indices = torch.randperm(num_frames)[:num_crop_frames]
        crop_mask[crop_indices] = True
        
    for i in range(num_frames):
        # sample to see if we should crop this frame
        if not crop_mask[i]:
            continue

        center = center_list[i]
        scale = scale_list[i]
        center_x, center_y = center
        width = height = scale * 200
        center_x, center_y, width, height = crop_fn(center_x, center_y, width, height, keypoints_2d_list[i])

        THRESH = 4
        if width < THRESH or height < THRESH:
            pass
        else:
            center = np.array([center_x, center_y]) 
            scale = max(width, height) / 200.0
            crop_center_list[i] = center
            crop_scale_list[i] = scale

    return crop_center_list, crop_scale_list
import os
import sys
import cv2
import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys

# Absolute path to the GFPGAN repo so face.py works regardless of cwd
GFPGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GFPGAN')


def enhance_face_with_gfpgan(image_path: str, upscale: int = 2) -> str:
    if GFPGAN_DIR not in sys.path:
        sys.path.insert(0, GFPGAN_DIR)

    from gfpgan import GFPGANer  # noqa: PLC0415
    bg_upsampler = None
    bg_model = None
    if torch.cuda.is_available():
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: PLC0415
        from realesrgan import RealESRGANer             # noqa: PLC0415
        bg_model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=2,
        )
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=bg_model,
            tile=400,
            tile_pad=10,
            pre_pad=0
        )
    else:
        import warnings
        warnings.warn('CUDA not available – background upsampler disabled.')

    model_name = 'GFPGANv1.3'
    model_path = os.path.join(GFPGAN_DIR, 'experiments', 'pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join(GFPGAN_DIR, 'gfpgan', 'weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
    )

    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if input_img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')

    _, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=True,
        paste_back=True,
        weight=0.5,
    )

    enhanced = restored_faces[0] if restored_faces else (restored_img if restored_img is not None else input_img)

    enhanced_path = image_path.rsplit('.', 1)
    enhanced_path = enhanced_path[0] + '_gfpgan_enhanced.png'
    cv2.imwrite(enhanced_path, enhanced)

    del restorer
    if bg_upsampler is not None:
        del bg_upsampler
    if bg_model is not None:
        del bg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return enhanced_path


def reconstruct_face(image_path: str, output_dir: str = 'head_output'):
    os.makedirs(output_dir, exist_ok=True)

    print('Enhancing face with GFPGAN...')
    enhanced_path = enhance_face_with_gfpgan(image_path, upscale=2)

    print('Loading head reconstruction model...')
    recon_pipeline = pipeline(
        Tasks.head_reconstruction,
        model='iic/cv_HRN_head-reconstruction',
        model_revision='v0.1',
        hair_tex=True,
    )

    print(f'Running inference on: {enhanced_path}')
    result = recon_pipeline(enhanced_path)

    print('Result keys:', list(result.keys()))

    mesh = result[OutputKeys.OUTPUT]['mesh']
    mesh['texture_map'] = result[OutputKeys.OUTPUT_IMG]
    obj_path = os.path.join(output_dir, 'head_mesh.obj')
    write_obj(obj_path, mesh)
    print(f'OBJ saved to {obj_path}')

    # if os.path.isfile(enhanced_path):
    #     os.remove(enhanced_path)

    return result


if __name__ == '__main__':
    reconstruct_face(sys.argv[1])
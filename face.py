import os
import sys
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.cv.face_reconstruction.utils import write_obj
from modelscope.outputs import OutputKeys

def reconstruct_face(image_path: str, output_dir: str = 'head_output'):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading head reconstruction model...")
    recon_pipeline = pipeline(
        Tasks.head_reconstruction,
        model='iic/cv_HRN_head-reconstruction',
        model_revision='v0.1',
        hair_tex=True
    )

    print(f"Running inference on: {image_path}")
    result = recon_pipeline(image_path)

    print("Result keys:", list(result.keys()))

    # Save OBJ mesh
    mesh = result[OutputKeys.OUTPUT]['mesh']
    mesh['texture_map'] = result[OutputKeys.OUTPUT_IMG]
    obj_path = os.path.join(output_dir, 'head_mesh.obj')
    write_obj(obj_path, mesh)
    print(f"OBJ saved to {obj_path}")

    return result

if __name__ == '__main__':
    # Usage: python face.py <image>
    reconstruct_face(sys.argv[1])
# Copyright 2024 Perceiving Systems, Max Planck Institute for Intelligent Systems

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "SMPL-X for Blender",
    "author": "Joachim Tesch, Max Planck Institute for Intelligent Systems",
    "version": (2024, 11, 29),
    "blender": (3, 6, 0),
    "location": "Viewport > Right panel",
    "description": "SMPL-X for Blender",
    "wiki_url": "https://smpl-x.is.tue.mpg.de/",
    "category": "SMPL-X"}

import bpy
from bpy_extras.io_utils import ImportHelper,ExportHelper # ImportHelper/ExportHelper is a helper class, defines filename and invoke() function which calls the file selector.
from mathutils import Vector, Quaternion
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty )
from bpy.types import ( PropertyGroup )

import json
from math import radians
import numpy as np
import os
import pickle

# SMPL-X globals
USE_SMPLX_2020 = False
SMPLX_MODELFILE = "smplx_model_20210421.blend"
SMPLX_MODELFILE_300 = "smplx_model_20230302.blend"
SMPLX_MODELFILE_LH = "smplx_model_lh_20230302.blend"
SMPLX_MODELFILE_2020 = "smplx_model_2020_20230227.blend"
SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15
SHAPEKEY_VALUE_RANGE=5
# End SMPL-X globals

def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

def update_corrective_poseshapes(self, context):
    if self.smplx_corrective_poseshapes:
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
    else:
        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return

# Ensure that we have valid slider ranges, this needed for imported FBX files where the default range will be set to [0,1] on import
def smplx_ensure_valid_shapekey_slider_ranges(skinned_mesh):
    update_slider_ranges = False
    for key_name in ["Shape000", "Exp000", "Pose000"]:
        if key_name in skinned_mesh.data.shape_keys.key_blocks:
            key_block = skinned_mesh.data.shape_keys.key_blocks[key_name]
            if (key_block.slider_min > -SHAPEKEY_VALUE_RANGE) or (key_block.slider_max < SHAPEKEY_VALUE_RANGE):
                update_slider_ranges = True
                break

    if update_slider_ranges:
        for index, key_block in enumerate(skinned_mesh.data.shape_keys.key_blocks):
            if index == 0:
                continue # skip Base shape key

            key_block.slider_min = -SHAPEKEY_VALUE_RANGE
            key_block.slider_max = SHAPEKEY_VALUE_RANGE

# Property groups for UI
class PG_SMPLXProperties(PropertyGroup):

    if USE_SMPLX_2020:
        smplx_version: EnumProperty(
            name = "Version",
            description = "SMPL-X version",
            items = [ ("2020", "2020", "SMPL-X with FLAME 2020 expression blendshapes")]
        )

        smplx_gender: EnumProperty(
            name = "Model",
            description = "SMPL-X model",
            items = [ ("neutral", "Neutral", "")]
        )
    else:
        smplx_version: EnumProperty(
            name = "Version",
            description = "SMPL-X version",
            items = [ ("locked_head", "Locked Head", "Locked head model with removed head bun"), ("v1.1", "v1.1", "") ]
        )

        smplx_gender: EnumProperty(
            name = "Model",
            description = "SMPL-X model",
            items = [ ("female", "Female", ""), ("male", "Male", ""), ("neutral", "Neutral", "")]
        )

    smplx_uv: EnumProperty(
        name = "UV",
        description = "SMPL-X UV version",
        items = [ ("UV_2023", "2023", "Latest UV layout with two eyeball regions"), ("UV_2021", "2021", "Original Blender add-on UV layout") ]
    )

    smplx_texture: EnumProperty(
        name = "",
        description = "SMPL-X model texture",
        items = [ ("NONE", "None", ""), ("smplx_texture_f_2023.png", "Female (UV 2023)", ""), ("smplx_texture_m_2023.png", "Male (UV 2023)", ""), ("smplx_texture_f_alb.png", "Female (UV 2021)", ""), ("smplx_texture_m_alb.png", "Male (UV 2021)", ""), ("smplx_texture_rainbow.png", "Rainbow (UV 2021)", ""), ("UV_GRID", "UV Grid", ""), ("COLOR_GRID", "Color Grid", "") ]
    )

    smplx_corrective_poseshapes: BoolProperty(
        name = "Corrective Pose Shapes",
        description = "Enable/disable corrective pose shapes of SMPL-X model",
        update = update_corrective_poseshapes,
        default = True
    )

    smplx_handpose: EnumProperty(
        name = "",
        description = "SMPL-X hand pose",
        items = [ ("relaxed", "Relaxed", ""), ("flat", "Flat", "") ]
    )

    smplx_height: FloatProperty(name="Target Height [m]", default=1.70, min=1.4, max=2.2)

    smplx_weight: FloatProperty(name="Target Weight [kg]", default=60, min=40, max=110)

    # Comma-separated mesh names waiting to be bound via "Bind to Selected Bones"
    smplx_pending_meshes: StringProperty(
        name="Pending Meshes",
        description="Mesh names stored for partial-bone binding (set by Remember Meshes)",
        default=""
    )


class SMPLXAddGender(bpy.types.Operator):
    bl_idname = "scene.smplx_add_gender"
    bl_label = "Add"
    bl_description = ("Add SMPL-X model of selected gender to scene")
    bl_options = {'REGISTER', 'UNDO'}

    uv_2023 = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            if (context.active_object is None) or (context.active_object.mode == 'OBJECT'):
                return True
            else: 
                return False
        except: return False

    def execute(self, context):
        gender = context.window_manager.smplx_tool.smplx_gender
        print("Adding gender: " + gender)

        path = os.path.dirname(os.path.realpath(__file__))

        if context.window_manager.smplx_tool.smplx_version == "locked_head":
            model_file = SMPLX_MODELFILE_LH
        elif context.window_manager.smplx_tool.smplx_version == "2020":
            model_file = SMPLX_MODELFILE_2020
        else:
            # v1.1
            # Use 300 shape model if available
            model_path = os.path.join(path, "data", SMPLX_MODELFILE_300)
            if os.path.exists(model_path):
                model_file = SMPLX_MODELFILE_300
            else:
                model_file = SMPLX_MODELFILE

        objects_path = os.path.join(path, "data", model_file, "Object")
        object_name = "SMPLX-mesh-" + gender

        bpy.ops.wm.append(filename=object_name, directory=str(objects_path))

        # Select imported mesh
        object_name = context.selected_objects[0].name
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.objects[object_name].select_set(True)
        obj = bpy.context.active_object

        # Set currently selected hand pose
        bpy.ops.object.smplx_set_handpose('EXEC_DEFAULT')

        # Set target UV if needed, default UV in .blend is UV_2021
        uv_version = context.window_manager.smplx_tool.smplx_uv
        print(f"UV map: {uv_version}")
        obj["smplx_uv"] = uv_version # store UV version as custom property

        if uv_version == "UV_2023":
            if self.uv_2023 is None:
                path = os.path.dirname(os.path.realpath(__file__))
                uv_npz_path = os.path.join(path, "data", "smplx_uv_2023.npz")
                with np.load(uv_npz_path) as data:
                    self.uv_2023 = data["uv_coordinates"]

            # Write loaded UV coordinates to the UV map
            uv_map = obj.data.uv_layers.active.data
            for i, face in enumerate(obj.data.polygons):
                for j, loop_index in enumerate(face.loop_indices):
                    uv_map[loop_index].uv = self.uv_2023[i * len(face.loop_indices) + j]

        return {'FINISHED'}

class SMPLXSetTexture(bpy.types.Operator):
    bl_idname = "object.smplx_set_texture"
    bl_label = "Set"
    bl_description = ("Set selected texture")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in active object is mesh
            if (context.object.type == 'MESH'):
                return True
            else:
                return False
        except: return False

    def execute(self, context):
        texture = context.window_manager.smplx_tool.smplx_texture
        print("Setting texture: " + texture)

        obj = bpy.context.object
        if (len(obj.data.materials) == 0) or (obj.data.materials[0] is None):
            self.report({'WARNING'}, "Selected mesh has no material: %s" % obj.name)
            return {'CANCELLED'}

        mat = obj.data.materials[0]
        links = mat.node_tree.links
        nodes = mat.node_tree.nodes

        # Find texture node
        node_texture = None
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                node_texture = node
                break

        # Find shader node
        node_shader = None
        for node in nodes:
            if node.type.startswith('BSDF'):
                node_shader = node
                break

        if texture == 'NONE':
            # Unlink texture node
            if node_texture is not None:
                for link in node_texture.outputs[0].links:
                    links.remove(link)

                nodes.remove(node_texture)

                # 3D Viewport still shows previous texture when texture link is removed via script.
                # As a workaround we trigger desired viewport update by setting color value.
                node_shader.inputs[0].default_value = node_shader.inputs[0].default_value
        else:
            if node_texture is None:
                node_texture = nodes.new(type="ShaderNodeTexImage")

            if (texture == 'UV_GRID') or (texture == 'COLOR_GRID'):
                if texture not in bpy.data.images:
                    bpy.ops.image.new(name=texture, generated_type=texture)
                image = bpy.data.images[texture]
            else:
                if texture not in bpy.data.images:
                    path = os.path.dirname(os.path.realpath(__file__))
                    texture_path = os.path.join(path, "data", texture)
                    image = bpy.data.images.load(texture_path)
                else:
                    image = bpy.data.images[texture]

            node_texture.image = image

            # Link texture node to shader node if not already linked
            if len(node_texture.outputs[0].links) == 0:
                links.new(node_texture.outputs[0], node_shader.inputs[0])

        # Switch viewport shading to Material Preview to show texture
        if bpy.context.space_data:
            if bpy.context.space_data.type == 'VIEW_3D':
                bpy.context.space_data.shading.type = 'MATERIAL'

        return {'FINISHED'}

class SMPLXMeasurementsToShape(bpy.types.Operator):
    bl_idname = "object.smplx_measurements_to_shape"
    bl_label = "Measurements To Shape"
    bl_description = ("Calculate and set shape parameters for specified measurements")
    bl_options = {'REGISTER', 'UNDO'}

    betas_regressor = {}
    betas_regressor["female"] = None
    betas_regressor["male"] = None
    betas_regressor["neutral"] = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for gender in ["female", "male", "neutral"]:
            if self.betas_regressor[gender] is None:
                path = os.path.dirname(os.path.realpath(__file__))
                regressor_path = os.path.join(path, "data", f"smplx_measurements_to_betas_{gender}.json")
                with open(regressor_path) as f:
                    data = json.load(f)
                    self.betas_regressor[gender] = (np.asarray(data["A"]).reshape(-1, 2), np.asarray(data["B"]).reshape(-1, 1))

        gender = obj["smplx_gender"]
        (A, B) = self.betas_regressor[gender]

        # Calculate beta values from measurements
        height_m = context.window_manager.smplx_tool.smplx_height
        height_cm = height_m * 100.0
        weight_kg = context.window_manager.smplx_tool.smplx_weight

        v_root = pow(weight_kg, 1.0/3.0)
        measurements = np.asarray([[height_cm], [v_root]])
        betas = A @ measurements + B

        num_betas = betas.shape[0]
        for i in range(num_betas):
            name = f"Shape{i:03d}"
            key_block = obj.data.shape_keys.key_blocks[name]
            value = betas[i, 0]

            # Adjust key block min/max range to value
            if value < key_block.slider_min:
                key_block.slider_min = value
            elif value > key_block.slider_max:
                key_block.slider_max = value

            key_block.value = value

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXRandomShape(bpy.types.Operator):
    bl_idname = "object.smplx_random_shape"
    bl_label = "Random"
    bl_description = ("Sets all shape blend shape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        smplx_ensure_valid_shapekey_slider_ranges(obj)
        randomized_betas = 0
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                beta = np.random.normal(0.0, 1.0)
                beta = np.clip(beta, -1.0, 1.0)
                key_block.value = beta

                randomized_betas += 1
                if randomized_betas >= 16:
                    break

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXResetShape(bpy.types.Operator):
    bl_idname = "object.smplx_reset_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blend shape keys for shape")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                key_block.value = 0.0

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXRandomExpressionShape(bpy.types.Operator):
    bl_idname = "object.smplx_random_expression_shape"
    bl_label = "Random Face Expression"
    bl_description = ("Sets all face expression blend shape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        smplx_ensure_valid_shapekey_slider_ranges(obj)
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = np.random.uniform(-2, 2)

        return {'FINISHED'}

class SMPLXResetExpressionShape(bpy.types.Operator):
    bl_idname = "object.smplx_reset_expression_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blend shape keys for face expression")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = 0.0

        return {'FINISHED'}

class SMPLXSnapGroundPlane(bpy.types.Operator):
    bl_idname = "object.smplx_snap_ground_plane"
    bl_label = "Snap To Ground Plane"
    bl_description = ("Snaps mesh to the XY ground plane")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ((context.object.type == 'MESH') or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        obj = bpy.context.object
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        # Get vertices with applied skin modifier in object coordinates
        depsgraph = context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_from_eval = object_eval.to_mesh()

        # Get vertices in world coordinates
        matrix_world = obj.matrix_world
        vertices_world = [matrix_world @ vertex.co for vertex in mesh_from_eval.vertices]
        z_min = (min(vertices_world, key=lambda item: item.z)).z
        object_eval.to_mesh_clear() # Remove temporary mesh

        # Adjust height of armature so that lowest vertex is on ground plane.
        # Do not apply new armature location transform so that we are later able to show loaded poses at their desired height.
        armature.location.z = armature.location.z - z_min

        return {'FINISHED'}

class SMPLXUpdateJointLocations(bpy.types.Operator):
    bl_idname = "object.smplx_update_joint_locations"
    bl_label = "Update Joint Locations"
    bl_description = ("Update joint locations after shape changes")
    bl_options = {'REGISTER', 'UNDO'}

    j_regressor = {}
    j_regressor["female"] = { "10": None, "300": None, "300_lh": None }
    j_regressor["male"] = { "10": None, "300": None, "300_lh": None }
    j_regressor["neutral"] = { "10": None, "300": None, "300_lh": None }

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def load_regressor(self, gender, betas):
        path = os.path.dirname(os.path.realpath(__file__))
        prefix = ""
        if betas == "10":
            suffix = ""
        elif betas == "300":
            suffix = "_300"
        elif betas == "300_lh":
            suffix = "_300"
            prefix = "lh_"
        else:
            print(f"ERROR: No betas-to-joints regressor for desired beta shapes [{betas}]")
            return (None, None)

        regressor_path = os.path.join(path, "data", f"smplx_betas_to_joints_{prefix}{gender}{suffix}.json")
        with open(regressor_path) as f:
            data = json.load(f)
            return (np.asarray(data["betasJ_regr"]), np.asarray(data["template_J"]))

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        # Get beta shapes
        betas = []
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                betas.append(key_block.value)
        num_betas = len(betas)
        betas = np.array(betas)

        # Cache regressor files on first call
        for target_betas in ["10", "300", "300_lh"]:
            for gender in ["female", "male", "neutral"]:
                if self.j_regressor[gender][target_betas] is None:
                    self.j_regressor[gender][target_betas] = self.load_regressor(gender, target_betas)

        key = f"{num_betas}"
        if obj["smplx_version"] == "locked_head":
            key += "_lh"
        gender = obj["smplx_gender"]
        (betas_to_joints, template_j) = self.j_regressor[gender][key]
        joint_locations = betas_to_joints @ betas + template_j

        # Set new bone joint locations
        armature = obj.parent
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        for index in range(NUM_SMPLX_JOINTS):
            bone = armature.data.edit_bones[SMPLX_JOINT_NAMES[index]]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)

            # Convert SMPL-X joint locations to Blender joint locations
            joint_location_smplx = joint_locations[index]
            bone_start = Vector( (joint_location_smplx[0], -joint_location_smplx[2], joint_location_smplx[1]) )
            bone.translate(bone_start)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj

        return {'FINISHED'}

class SMPLXSetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_set_poseshapes"
    bl_label = "Update Pose Shapes"
    bl_description = ("Sets and updates corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=object)
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Calculate weights of pose corrective blend shapes
    # Input is pose of all 55 joints, output is weights for all joints except pelvis
    def rodrigues_to_posecorrective_weight(self, pose):
        joints_posecorrective = NUM_SMPLX_JOINTS
        rod_rots = np.asarray(pose).reshape(joints_posecorrective, 3)
        mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return(bshapes)

    def execute(self, context):
        obj = bpy.context.object

        # Get armature pose in rodrigues representation
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        smplx_ensure_valid_shapekey_slider_ranges(obj)

        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        poseweights = self.rodrigues_to_posecorrective_weight(pose)

        # Set weights for pose corrective shape keys
        for index, weight in enumerate(poseweights):
            obj.data.shape_keys.key_blocks["Pose%03d" % index].value = weight

        # Set checkbox without triggering update function
        context.window_manager.smplx_tool["smplx_corrective_poseshapes"] = True

        return {'FINISHED'}

class SMPLXResetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_reset_poseshapes"
    bl_label = "Reset"
    bl_description = ("Resets corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'ARMATURE':
            obj = bpy.context.object.children[0]

        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Pose"):
                key_block.value = 0.0

        return {'FINISHED'}

class SMPLXSetHandpose(bpy.types.Operator):
    bl_idname = "object.smplx_set_handpose"
    bl_label = "Set"
    bl_description = ("Set selected hand pose")
    bl_options = {'REGISTER', 'UNDO'}

    hand_poses = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        if self.hand_poses is None:
            path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(path, "data", "smplx_handposes.npz")
            with np.load(data_path, allow_pickle=True) as data:
                self.hand_poses = data["hand_poses"].item()

        hand_pose_name = context.window_manager.smplx_tool.smplx_handpose
        print("Setting hand pose: " + hand_pose_name)

        if hand_pose_name not in self.hand_poses:
            self.report({"ERROR"}, f"Desired hand pose not existing: {hand_pose_name}")
            return {"CANCELLED"}

        (left_hand_pose, right_hand_pose) = self.hand_poses[hand_pose_name]

        hand_pose = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        hand_joint_start_index = 1 + NUM_SMPLX_BODYJOINTS + 3
        for index in range(2 * NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = hand_pose[index]            
            bone_name = SMPLX_JOINT_NAMES[index + hand_joint_start_index]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        # Update corrective poseshapes if used
        if context.window_manager.smplx_tool.smplx_corrective_poseshapes:
            bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXWritePose(bpy.types.Operator):
    bl_idname = "object.smplx_write_pose"
    bl_label = "Write Pose To Console"
    bl_description = ("Writes SMPL-X flat hand pose thetas to console window")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return (context.object.type == 'MESH') or (context.object.type == 'ARMATURE')
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        # Get armature pose in rodrigues representation
        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        print("\npose = " + str(pose))

        return {'FINISHED'}

class SMPLXResetPose(bpy.types.Operator):
    bl_idname = "object.smplx_reset_pose"
    bl_label = "Reset Pose"
    bl_description = ("Resets pose to default zero pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        for bone in armature.pose.bones:
            if bone.rotation_mode != 'QUATERNION':
                bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion()

        # Reset corrective pose shapes
        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXParentCustomMesh(bpy.types.Operator):
    bl_idname = "object.smplx_parent_custom_mesh"
    bl_label = "Parent Custom Mesh (Pose-Correct)"
    bl_description = (
        "Parent selected custom mesh(es) to the SMPL-X armature with a correct bind pose. "
        "HOW TO USE: (1) Go to the frame where your custom mesh visually aligns with the "
        "SMPL-X body. (2) Select your custom mesh(es). (3) Shift-click the SMPL-X mesh or "
        "armature to make it active. (4) Click this button. "
        "A 'bind proxy' armature is created whose rest-pose equals the current dance pose, "
        "so vertex weights are calculated correctly. The proxy follows the original animation "
        "via Copy-Transforms constraints."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            active = context.active_object
            return (
                active is not None and
                len(context.selected_objects) >= 2 and
                (
                    active.type == 'ARMATURE' or
                    (active.type == 'MESH' and
                     active.parent is not None and
                     active.parent.type == 'ARMATURE')
                )
            )
        except:
            return False

    def execute(self, context):
        # ------------------------------------------------------------------ #
        # 1. Resolve the SMPL-X armature from the active object               #
        # ------------------------------------------------------------------ #
        active = context.active_object
        if active.type == 'ARMATURE':
            smplx_armature = active
        elif active.type == 'MESH' and active.parent and active.parent.type == 'ARMATURE':
            smplx_armature = active.parent
        else:
            self.report({'ERROR'}, "Active object must be a SMPL-X armature or its skinned mesh.")
            return {'CANCELLED'}

        # ------------------------------------------------------------------ #
        # 2. Collect custom mesh(es) – everything selected that is not the    #
        #    SMPL-X armature or any of its existing children.                 #
        # ------------------------------------------------------------------ #
        smplx_family = {smplx_armature} | set(smplx_armature.children)
        custom_meshes = [
            obj for obj in context.selected_objects
            if obj not in smplx_family and obj.type == 'MESH'
        ]

        if not custom_meshes:
            self.report(
                {'ERROR'},
                "No custom mesh found in selection. "
                "Select your mesh(es) first, then Shift-click the SMPL-X armature/mesh."
            )
            return {'CANCELLED'}

        # ------------------------------------------------------------------ #
        # 3. Duplicate the SMPL-X armature to create a bind proxy.           #
        #    The proxy will have its pose baked as rest-pose so that the      #
        #    stickman's current world position becomes the bind pose.         #
        # ------------------------------------------------------------------ #
        bpy.ops.object.select_all(action='DESELECT')
        smplx_armature.select_set(True)
        context.view_layer.objects.active = smplx_armature
        bpy.ops.object.duplicate(linked=False)
        proxy_armature = context.view_layer.objects.active
        proxy_armature.name = smplx_armature.name + "_bind_proxy"

        # Give the proxy its own armature data (not linked to the original)
        proxy_armature.data = proxy_armature.data.copy()
        proxy_armature.data.name = smplx_armature.data.name + "_bind_proxy"

        # Remove any animation action that was copied from the original
        if proxy_armature.animation_data:
            proxy_armature.animation_data_clear()

        # ------------------------------------------------------------------ #
        # 4. Apply the current (dance) pose as rest pose on the proxy.       #
        #    After this, proxy.data.pose_position == 'POSE' but every bone   #
        #    will show identity rotation in pose mode (the dance frame IS     #
        #    the new rest).                                                   #
        # ------------------------------------------------------------------ #
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        # ------------------------------------------------------------------ #
        # 5. Add Copy Transforms constraints on every proxy bone so the proxy #
        #    mirrors the original armature's animation exactly.               #
        # ------------------------------------------------------------------ #
        original_bone_names = {b.name for b in smplx_armature.pose.bones}
        bpy.ops.object.select_all(action='DESELECT')
        proxy_armature.select_set(True)
        context.view_layer.objects.active = proxy_armature
        bpy.ops.object.mode_set(mode='POSE')
        for bone in proxy_armature.pose.bones:
            if bone.name not in original_bone_names:
                continue
            ct = bone.constraints.new('COPY_TRANSFORMS')
            ct.name = "Follow_Body"
            ct.target = smplx_armature
            ct.subtarget = bone.name
        bpy.ops.object.mode_set(mode='OBJECT')

        # ------------------------------------------------------------------ #
        # 6. Parent the custom mesh(es) to the proxy with automatic weights. #
        #    The proxy's rest-pose == the stickman's current position,        #
        #    so heat-weighting is correct and zero deformation is produced    #
        #    at the current frame.                                            #
        # ------------------------------------------------------------------ #
        bpy.ops.object.select_all(action='DESELECT')
        for mesh in custom_meshes:
            mesh.select_set(True)
        proxy_armature.select_set(True)
        context.view_layer.objects.active = proxy_armature
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        mesh_names = ", ".join(m.name for m in custom_meshes)
        self.report(
            {'INFO'},
            f"Parented [{mesh_names}] to bind proxy '{proxy_armature.name}' "
            f"which follows '{smplx_armature.name}' via Copy Transforms. "
        )
        return {'FINISHED'}


class SMPLXLoadPose(bpy.types.Operator, ImportHelper):
    bl_idname = "object.smplx_load_pose"
    bl_label = "Load Pose"
    bl_description = ("Load relaxed-hand model pose from file")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.pkl",
        options={'HIDDEN'}
    )

    update_shape: BoolProperty(
        name="Update shape parameters",
        description="Update shape parameters using the beta shape information in the loaded file",
        default=True
    )

    hand_pose_relaxed = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj
            obj = armature.children[0]
            context.view_layer.objects.active = obj # mesh needs to be active object for recalculating joint locations

        if self.hand_pose_relaxed is None:
            path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(path, "data", "smplx_handposes.npz")
            with np.load(data_path, allow_pickle=True) as data:
                hand_poses = data["hand_poses"].item()
                (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
                self.hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        print("Loading: " + self.filepath)

        translation = None
        global_orient = None
        body_pose = None
        jaw_pose = None
        #leye_pose = None
        #reye_pose = None
        left_hand_pose = None
        right_hand_pose = None
        betas = None
        expression = None
        with open(self.filepath, "rb") as f:
            data = pickle.load(f, encoding="latin1")

            if "transl" in data:
                translation = np.array(data["transl"]).reshape(3)

            if "global_orient" in data:
                global_orient = np.array(data["global_orient"]).reshape(3)

            body_pose = np.array(data["body_pose"])
            if body_pose.shape != (1, NUM_SMPLX_BODYJOINTS * 3):
                print(f"Invalid body pose dimensions: {body_pose.shape}")
                body_data = None
                return {'CANCELLED'}

            body_pose = np.array(data["body_pose"]).reshape(NUM_SMPLX_BODYJOINTS, 3)

            jaw_pose = np.array(data["jaw_pose"]).reshape(3)
            #leye_pose = np.array(data["leye_pose"]).reshape(3)
            #reye_pose = np.array(data["reye_pose"]).reshape(3)
            left_hand_pose = np.array(data["left_hand_pose"]).reshape(-1, 3)
            right_hand_pose = np.array(data["right_hand_pose"]).reshape(-1, 3)

            betas = np.array(data["betas"]).reshape(-1).tolist()
            expression = np.array(data["expression"]).reshape(-1).tolist()

        # Update shape if selected
        if self.update_shape:
            bpy.ops.object.mode_set(mode='OBJECT')
            for index, beta in enumerate(betas):
                key_block_name = f"Shape{index:03}"

                if key_block_name in obj.data.shape_keys.key_blocks:
                    obj.data.shape_keys.key_blocks[key_block_name].value = beta
                else:
                    print(f"ERROR: No key block for: {key_block_name}")

            bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        if global_orient is not None:
            set_pose_from_rodrigues(armature, "pelvis", global_orient)

        for index in range(NUM_SMPLX_BODYJOINTS):
            pose_rodrigues = body_pose[index]
            bone_name = SMPLX_JOINT_NAMES[index + 1] # body pose starts with left_hip
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        set_pose_from_rodrigues(armature, "jaw", jaw_pose)

        # Left hand
        start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
        for i in range(0, NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = left_hand_pose[i]
            bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
            pose_relaxed_rodrigues = self.hand_pose_relaxed[i]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

        # Right hand
        start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
        for i in range(0, NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = right_hand_pose[i]
            bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
            pose_relaxed_rodrigues = self.hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

        if translation is not None:
            # Set translation
            armature.location = (translation[0], -translation[2], translation[1])

        # Activate corrective poseshapes
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        # Set face expression
        for index, exp in enumerate(expression):
            key_block_name = f"Exp{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = exp
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        return {'FINISHED'}

class SMPLXAddAnimation(bpy.types.Operator, ImportHelper):
    bl_idname = "object.smplx_add_animation"
    bl_label = "Add Animation"
    bl_description = ("Load AMASS/SMPL-X animation and create animated SMPL-X body")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'}
    )

    anim_format: EnumProperty(
        name="Format",
        items=(
            ("AMASS", "AMASS", ""),
            ("SMPL-X", "SMPL-X", ""),
        ),
    )

    rest_position: EnumProperty(
        name="Body rest position",
        items=(
            ("SMPL-X", "SMPL-X", "Use default SMPL-X rest position (feet below the floor)"),
            ("GROUNDED", "Grounded", "Use feet-on-floor rest position"),
        ),
    )

    hand_reference: EnumProperty(
        name="Hand pose reference",
        items=(
            ("FLAT", "Flat", "Use flat hand as hand pose reference"),
            ("RELAXED", "Relaxed", "Use relaxed hand as hand pose reference"),
        ),
    )

    keyframe_corrective_pose_weights: BoolProperty(
        name="Use keyframed corrective pose weights",
        description="Keyframe the weights of the corrective pose shapes for each frame. This increases animation load time and slows down editor real-time playback.",
        default=False
    )

    target_framerate: IntProperty(
        name="Target framerate [fps]",
        description="Target framerate for animation in frames-per-second. Lower values will speed up import time.",
        default=30,
        min = 1,
        max = 120
    )

    hand_pose_relaxed = None

    @classmethod
    def poll(cls, context):
        try:
            # Always enable button
            return True
        except: return False

    def execute(self, context):

        target_framerate = self.target_framerate

        if self.hand_reference == "RELAXED":
            if self.hand_pose_relaxed is None:
                path = os.path.dirname(os.path.realpath(__file__))
                data_path = os.path.join(path, "data", "smplx_handposes.npz")
                with np.load(data_path, allow_pickle=True) as data:
                    hand_poses = data["hand_poses"].item()
                    (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
                    self.hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        # Load .npz file
        print("Loading: " + self.filepath)
        with np.load(self.filepath) as data:
            # Check for valid AMASS file
            if ("trans" not in data) or ("gender" not in data) or (("mocap_frame_rate" not in data) and ("mocap_framerate" not in data)) or ("betas" not in data) or ("poses" not in data):
                self.report({"ERROR"}, "Invalid AMASS animation data file")
                return {"CANCELLED"}

            trans = data["trans"]
            gender = str(data["gender"])
            mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
            betas = data["betas"]
            poses = data["poses"]

            if mocap_framerate < target_framerate:
                self.report({"ERROR"}, f"Mocap framerate ({mocap_framerate}) below target framerate ({target_framerate})")
                return {"CANCELLED"}

        if (context.active_object is not None):
            bpy.ops.object.mode_set(mode='OBJECT')

        # Add gender specific model
        context.window_manager.smplx_tool.smplx_gender = gender
        context.window_manager.smplx_tool.smplx_handpose = "flat"
        bpy.ops.scene.smplx_add_gender()

        obj = context.view_layer.objects.active
        armature = obj.parent

        # Append animation name to armature name
        armature.name = armature.name + "_" + os.path.basename(self.filepath).replace(".npz", "")

        context.scene.render.fps = target_framerate
        context.scene.frame_start = 1

        # Set shape and update joint locations
        bpy.ops.object.mode_set(mode='OBJECT')
        for index, beta in enumerate(betas):
            key_block_name = f"Shape{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = beta
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        height_offset = 0
        if self.rest_position == "GROUNDED":
            bpy.ops.object.smplx_snap_ground_plane('EXEC_DEFAULT')
            height_offset = armature.location[2]

            obj["smplx_bind_pose_height_offset"] = height_offset

            # Apply location offsets to armature and skinned mesh
            bpy.context.view_layer.objects.active = armature
            armature.select_set(True)
            obj.select_set(True)
            bpy.ops.object.transform_apply(location = True, rotation=False, scale=False) # apply to selected objects
            armature.select_set(False)

            # Fix root bone location
            bpy.ops.object.mode_set(mode='EDIT')
            bone = armature.data.edit_bones["root"]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = obj

        # Keyframe poses
        step_size = int(mocap_framerate / target_framerate)

        num_frames = trans.shape[0]
        num_keyframes = int(num_frames / step_size)

        if self.keyframe_corrective_pose_weights:
            print(f"Adding pose keyframes with keyframed corrective pose weights: {num_keyframes}")
        else:
            print(f"Adding pose keyframes: {num_keyframes}")

        if len(bpy.data.actions) == 0:
            # Set end frame if we don't have any previous animations in the scene
            context.scene.frame_end = num_keyframes
        elif num_keyframes > context.scene.frame_end:
            context.scene.frame_end = num_keyframes

        for index, frame in enumerate(range(0, num_frames, step_size)):
            if (index % 100) == 0:
                print(f"  {index}/{num_keyframes}")
            current_frame = index + 1
            current_pose = poses[frame].reshape(-1, 3)
            current_trans = trans[frame]
            for bone_index, bone_name in enumerate(SMPLX_JOINT_NAMES):
                if bone_name == "pelvis":
                    # Keyframe pelvis location
                    if self.rest_position == "GROUNDED":
                        current_trans[1] = current_trans[1] - height_offset # SMPL-X local joint coordinates are Y-Up

                    armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                    armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)

                # Keyframe bone rotation
                pose_rodrigues = current_pose[bone_index]

                if self.hand_reference == "FLAT":
                    set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
                else:
                    # Relaxed hand pose uses different coordinate system for fingers
                    finger_names = ["index", "middle", "pinky", "ring", "thumb"]
                    if not any([x in bone_name for x in finger_names]):
                        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
                    else:
                        # Finger rotations are relative to relaxed hand pose
                        hand_start_index = 1 + NUM_SMPLX_BODYJOINTS + 3
                        relaxed_hand_joint_index = bone_index - hand_start_index
                        pose_relaxed_rodrigues = self.hand_pose_relaxed[relaxed_hand_joint_index]
                        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

            if self.keyframe_corrective_pose_weights:
                # Calculate corrective poseshape weights for current pose and keyframe them.
                # Note: This significantly increases animation load time and also reduces real-time playback speed in Blender viewport.
                bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
                for key_block in obj.data.shape_keys.key_blocks:
                    if key_block.name.startswith("Pose"):
                        key_block.keyframe_insert("value", frame=current_frame)

        if self.anim_format == "AMASS":
            # AMASS target floor is XY ground plane for SMPL-X template in OpenGL Y-up space (XZ ground plane).
            # Since SMPL-X Blender model is Z-up (and not Y-up) for rest/template pose, we need to adjust root node rotation to ensure that the resulting animated body is on Blender XY ground plane.
            bone_name = "root"
            if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
                armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
            armature.pose.bones[bone_name].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), radians(-90))
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=1)

        print(f"  {num_keyframes}/{num_keyframes}")
        context.scene.frame_set(1)

        return {'FINISHED'}

class SMPLXExportAlembic(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_alembic"
    bl_label = "Export Alembic ABC"
    bl_description = ("Export as Alembic geometry cache")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".abc"

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):
        bpy.ops.wm.alembic_export(filepath=self.filepath, selected=True, packuv=False, face_sets=True)
        print("Exported: " + self.filepath)

        return {'FINISHED'}

class SMPLXExportFBX(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_fbx"
    bl_label = "Export FBX"
    bl_description = ("Export skinned mesh in FBX format")
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper mixin class uses this
    filename_ext = ".fbx"

    export_shape_keys: EnumProperty(
        name = "Blend Shapes",
        description = "Blend shape export settings",
        items = [ ("SHAPE_POSECORRECTIVES", "All: Shape + Posecorrectives", "Export shape keys for body shape and pose correctives"),
                  ("SHAPE", "Reduced: Shape space only", "Export only shape keys for body shape"),
                  ("POSECORRECTIVES", "Reduced: Posecorrectives only", "Bake shape and expression into mesh, export only shape keys for pose correctives"),
                  ("NONE", "None: Apply shape space", "Do not export any shape keys, shape keys for body shape will be baked into mesh") ],
    )


    target_format: EnumProperty(
        name="Format",
        items=(
            ("UNITY", "Unity", ""),
            ("UNREAL", "Unreal", ""),
        ),
    )

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):

        obj = bpy.context.object

        armature_original = obj.parent
        skinned_mesh_original = obj

        # Operate on temporary copy of skinned mesh and armature
        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh_original.select_set(True)
        armature_original.select_set(True)
        bpy.context.view_layer.objects.active = skinned_mesh_original
        bpy.ops.object.duplicate()
        skinned_mesh = bpy.context.object
        armature = skinned_mesh.parent

        # Apply armature object location to armature root bone and skinned mesh so that armature and skinned mesh are at origin before export
        context.view_layer.objects.active = armature
        armature_offset = Vector(armature.location)
        armature.location = (0, 0, 0)
        bpy.ops.object.mode_set(mode='EDIT')
        for edit_bone in armature.data.edit_bones:
            if edit_bone.name != "root":
                edit_bone.translate(armature_offset)

        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = skinned_mesh
        mesh_location = Vector(skinned_mesh.location)
        skinned_mesh.location = mesh_location + armature_offset
        bpy.ops.object.transform_apply(location = True)

        # Reset pose
        bpy.ops.object.smplx_reset_pose('EXEC_DEFAULT')

        if ( (self.export_shape_keys == 'SHAPE') or (self.export_shape_keys == 'NONE') ):
            # Remove pose corrective shape keys
            print("Removing pose corrective shape keys")
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

            current_shape_key_index = 0
            for index in range(0, num_shape_keys):
                bpy.context.object.active_shape_key_index = current_shape_key_index

                if bpy.context.object.active_shape_key is not None:
                    if bpy.context.object.active_shape_key.name.startswith('Pose'):
                        bpy.ops.object.shape_key_remove(all=False)
                    else:
                        current_shape_key_index = current_shape_key_index + 1        

        if self.export_shape_keys == 'NONE':
            # Bake and remove shape keys
            print("Baking shape and removing shape keys for shape")

            # Zero out all pose corrective weights so that they do not contribute to baked shape
            for key_block in skinned_mesh.data.shape_keys.key_blocks:
                if key_block.name.startswith("Pose"):
                    key_block.value = 0.0

            # Create shape mix for current shape
            bpy.ops.object.shape_key_add(from_mix=True)
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

            # Remove all shape keys except newly added one
            bpy.context.object.active_shape_key_index = 0
            for count in range(0, num_shape_keys):
                bpy.ops.object.shape_key_remove(all=False)

        elif self.export_shape_keys == 'POSECORRECTIVES':
            # Bake shape and expression into Base shape key
            print("Baking shape and expression into Base shape key")

            # Zero out all pose corrective weights so that they do not contribute to baked shape
            for key_block in skinned_mesh.data.shape_keys.key_blocks:
                if key_block.name.startswith("Pose"):
                    key_block.value = 0.0

            # Create shape mix from current shape and expression
            bpy.ops.object.shape_key_add(from_mix=True)
            bpy.context.object.active_shape_key.name = "ShapeMix"

            # Copy shape mix vertices intp Base shape key
            bpy.context.object.active_shape_key_index = 0 # Select Base shape key
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.blend_from_shape(shape="ShapeMix", blend=1, add=False)
            bpy.ops.object.mode_set(mode='OBJECT')

            # Remove all shape and expression keys and shape mix
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())
            current_shape_key_index = 1
            for _ in range(1, num_shape_keys):
                bpy.context.object.active_shape_key_index = current_shape_key_index

                if bpy.context.object.active_shape_key is not None:
                    if (bpy.context.object.active_shape_key.name.startswith('Shape') or
                        bpy.context.object.active_shape_key.name.startswith('Exp')):
                        bpy.ops.object.shape_key_remove(all=False)
                    else:
                        current_shape_key_index = current_shape_key_index + 1
            bpy.context.object.active_shape_key_index = 0

        # Model (skeleton and skinned mesh) needs to have rotation of (90, 0, 0) when exporting so that it will have rotation (0, 0, 0) when imported into Unity
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh.select_set(True)
        skinned_mesh.rotation_euler = (radians(-90), 0, 0)
        bpy.context.view_layer.objects.active = skinned_mesh
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = False)
        skinned_mesh.rotation_euler = (radians(90), 0, 0)
        skinned_mesh.select_set(False)

        armature.select_set(True)
        armature.rotation_euler = (radians(-90), 0, 0)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = False)
        armature.rotation_euler = (radians(90), 0, 0)

        if self.target_format == "UNREAL":
            # Scale armature by 100 so that Unreal FBX importer can be used with default scale 1.
            # This ensures that attached objects to imported skeleton in Unreal will keep scale 1.

            armature.scale = (100, 100, 100)

            # Scale keyframed pelvis locations if available
            if armature.animation_data is not None:
                action = armature.animation_data.action
                for fcurve in action.fcurves:
                    if fcurve.data_path.endswith("location"):
                        for keyframe_point in fcurve.keyframe_points:
                            keyframe_point.co[1] = keyframe_point.co[1] * 100
                            keyframe_point.handle_left[1] = keyframe_point.handle_left[1] * 100
                            keyframe_point.handle_right[1] = keyframe_point.handle_right[1] * 100

            bpy.ops.object.transform_apply(location = False, rotation = False, scale = True)

        # Select armature and skinned mesh for export
        skinned_mesh.select_set(True)

        # Rename armature and skinned mesh to not contain Blender copy suffix
        gender = skinned_mesh_original["smplx_gender"]

        target_mesh_name = "SMPLX-mesh-%s" % gender
        target_armature_name = "SMPLX-%s" % gender

        if target_mesh_name in bpy.data.objects:
            bpy.data.objects[target_mesh_name].name = "SMPLX-temp-mesh"
        skinned_mesh.name = target_mesh_name

        if target_armature_name in bpy.data.objects:
            bpy.data.objects[target_armature_name].name = "SMPLX-temp-armature"
        armature.name = target_armature_name

        # Default FBX export settings export all animations. Since we duplicated the armature we have a copy of the animation and the original animation.
        # We avoid export of both by only exporting the active animation for the armature (bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False).
        # Disable keyframe simplification to ensure that exported FBX animation properly matches up with exported Alembic cache.
        bpy.ops.export_scene.fbx(filepath=self.filepath,
                                 use_selection=True,
                                 apply_scale_options="FBX_SCALE_ALL",
                                 use_custom_props=True,
                                 add_leaf_bones=False,
                                 bake_anim_use_nla_strips=False,
                                 bake_anim_use_all_actions=False,
                                 bake_anim_simplify_factor=0)

        print("Exported: " + self.filepath)

        # Remove temporary copies of armature and skinned mesh
        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh.select_set(True)
        armature.select_set(True)
        bpy.ops.object.delete()

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh_original.select_set(True)
        bpy.context.view_layer.objects.active = skinned_mesh_original

        if "SMPLX-temp-mesh" in bpy.data.objects:
            bpy.data.objects["SMPLX-temp-mesh"].name = target_mesh_name

        if "SMPLX-temp-armature" in bpy.data.objects:
            bpy.data.objects["SMPLX-temp-armature"].name = target_armature_name

        return {'FINISHED'}

class SMPLXExportShape(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_shape"
    bl_label = "Export Shape"
    bl_description = ("Export shape beta values in NPZ format")
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper mixin class uses this
    filename_ext = ".npz"

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):

        obj = bpy.context.object
        armature = obj.parent

        betas = []
        for index in range(300):
            key_block_name = f"Shape{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                beta = obj.data.shape_keys.key_blocks[key_block_name].value
                betas.append(beta)

        data = {}
        data["gender"] = obj["smplx_gender"]
        data["mocap_frame_rate"] = 30
        data["model"] = "smplx_" + obj["smplx_version"]
        data["betas"] = betas
        data["poses"] = [ [0.0] * 3 * NUM_SMPLX_JOINTS ]
        data["trans"] = [ [0.0, 0.0, 0.0] ]
        data["info"] = "Shape only,default pose"

        bind_pose_height_offset = 0.0
        if "smplx_bind_pose_height_offset" in obj:
            bind_pose_height_offset = obj["smplx_bind_pose_height_offset"]
        else:
            # Store current SnapToGroundPlane armature height offset which for default pose equals the distance used to map default bind pose to grounded bind pose.
            # Armature must be in default pose for correct SnapToGroundPlane results.
            bind_pose_height_offset = armature.location[2]

        data["bind_pose_height_offset"] = bind_pose_height_offset

        np.savez_compressed(self.filepath, **data)
        print("Exported: " + self.filepath)

        return {'FINISHED'}

class SMPLX_PT_Model(bpy.types.Panel):
    bl_label = "SMPL-X Model"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):

        layout = self.layout
        col = layout.column(align=True)
        
        row = col.row(align=True)
        col.prop(context.window_manager.smplx_tool, "smplx_version")
        col.prop(context.window_manager.smplx_tool, "smplx_gender")
        col.prop(context.window_manager.smplx_tool, "smplx_uv")
        col.operator("scene.smplx_add_gender", text="Add")

        col.separator()
        col.label(text="Texture:")
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.prop(context.window_manager.smplx_tool, "smplx_texture")
        split.operator("object.smplx_set_texture", text="Set")

class SMPLX_PT_Shape(bpy.types.Panel):
    bl_label = "Shape"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.window_manager.smplx_tool, "smplx_height")
        col.prop(context.window_manager.smplx_tool, "smplx_weight")
        col.operator("object.smplx_measurements_to_shape")
        col.separator()

        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.operator("object.smplx_random_shape")
        split.operator("object.smplx_reset_shape")
        col.separator()

        col.operator("object.smplx_snap_ground_plane")
        col.separator()

        col.operator("object.smplx_update_joint_locations")
        col.separator()
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.operator("object.smplx_random_expression_shape")
        split.operator("object.smplx_reset_expression_shape")

class SMPLX_PT_Pose(bpy.types.Panel):
    bl_label = "Pose"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.window_manager.smplx_tool, "smplx_corrective_poseshapes")
        col.separator()
        col.operator("object.smplx_set_poseshapes")

        col.separator()
        col.label(text="Hand Pose:")
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.prop(context.window_manager.smplx_tool, "smplx_handpose")
        split.operator("object.smplx_set_handpose", text="Set")

        col.separator()
        col.operator("object.smplx_write_pose")
        col.separator()
        col.operator("object.smplx_load_pose")

class SMPLX_PT_Animation(bpy.types.Panel):
    bl_label = "Animation"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("object.smplx_add_animation")


class SMPLXRigidAttachToBone(bpy.types.Operator):
    """Attach a face/accessory mesh RIGIDLY to one bone via a Child Of constraint.
    Zero vertex weights, zero deformation math, zero scars.
    The mesh object moves as a single rigid unit with the target bone."""
    bl_idname = "object.smplx_rigid_attach_to_bone"
    bl_label  = "Attach Rigidly to Bone"
    bl_description = (
        "Attach the pending mesh(es) rigidly to the SINGLE selected bone using a "
        "Child Of constraint. Perfect for face mesh, glasses, earrings, hat, hair. "
        "No vertex weights, no LBS, no scars. The mesh follows the bone as a solid unit. "
        "Step 1: In Object Mode, select mesh(es) + Shift-click armature, click "
        "'Remember Selected Meshes'. "
        "Step 2: Go to Pose Mode, select exactly ONE bone, click this button."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (
                context.mode == 'POSE' and
                context.active_object is not None and
                context.active_object.type == 'ARMATURE' and
                len(context.selected_pose_bones) >= 1 and
                context.window_manager.smplx_tool.smplx_pending_meshes != ""
            )
        except:
            return False

    def execute(self, context):
        smplx_armature = context.active_object
        target_bone    = context.active_pose_bone or context.selected_pose_bones[0]
        bone_name      = target_bone.name

        pending    = context.window_manager.smplx_tool.smplx_pending_meshes
        mesh_names = [n.strip() for n in pending.split(',') if n.strip()]
        garments   = [
            bpy.data.objects.get(n) for n in mesh_names
            if bpy.data.objects.get(n) is not None
        ]
        if not garments:
            self.report({'ERROR'},
                "No valid pending meshes. Run 'Remember Selected Meshes' first.")
            return {'CANCELLED'}

        # ── Step 1: Add Child Of constraint on every mesh (identity inverse) ──
        # Do this first while still in Pose Mode so we know bone_name is valid.
        constraint_name = f"RigidBone_{bone_name}"
        for mesh_obj in garments:
            for con in list(mesh_obj.constraints):
                if con.name.startswith("RigidBone_"):
                    mesh_obj.constraints.remove(con)
            con               = mesh_obj.constraints.new('CHILD_OF')
            con.name          = constraint_name
            con.target        = smplx_armature
            con.subtarget     = bone_name
            con.use_location_x = con.use_location_y = con.use_location_z = True
            con.use_rotation_x = con.use_rotation_y = con.use_rotation_z = True
            con.use_scale_x    = con.use_scale_y    = con.use_scale_z    = True
            # Leave inverse_matrix as identity for now — set correctly below.

        # ── Step 2: Switch to Object Mode, let Blender set the inverse ────────
        # bpy.ops.constraint.childof_set_inverse is the ONLY reliable way to
        # handle the correct inverse when the armature has a non-unit scale,
        # a root-bone -90° AMASS rotation, or the mesh has a parent.
        # It does exactly what the "Set Inverse" button does in the UI.
        bpy.ops.object.mode_set(mode='OBJECT')

        bound  = []
        failed = []
        for mesh_obj in garments:
            try:
                # Make this mesh the active object so the operator targets it.
                bpy.ops.object.select_all(action='DESELECT')
                mesh_obj.select_set(True)
                context.view_layer.objects.active = mesh_obj

                # childof_set_inverse reads the active constraint by name and
                # computes: inverse_matrix = inv(target_mat) @ owner_world_mat
                # accounting for all parent chains and armature transforms.
                bpy.ops.constraint.childof_set_inverse(
                    constraint=constraint_name,
                    owner='OBJECT'
                )
                bound.append(mesh_obj.name)
                print(f"[Rigid Attach] '{mesh_obj.name}' → bone '{bone_name}'")

            except Exception as e:
                failed.append(mesh_obj.name)
                print(f"[Rigid Attach] Error on '{mesh_obj.name}': {e}")
                import traceback; traceback.print_exc()

        # ── Step 3: Restore Pose Mode on the armature ─────────────────────────
        bpy.ops.object.select_all(action='DESELECT')
        smplx_armature.select_set(True)
        context.view_layer.objects.active = smplx_armature
        bpy.ops.object.mode_set(mode='POSE')

        context.window_manager.smplx_tool.smplx_pending_meshes = ""

        if bound:
            self.report({'INFO'},
                f"Rigidly attached [{', '.join(bound)}] to bone '{bone_name}'. "
                f"Mesh follows bone as a solid unit — no deformation, no size change.")
        if failed:
            self.report({'WARNING'},
                f"Failed: [{', '.join(failed)}]. Check the Blender console.")
        return {'FINISHED'}


class SMPLXRememberMeshes(bpy.types.Operator):
    """Step 1 of partial-bone binding. Call this in Object Mode with the target
    mesh(es) selected and the SMPL-X body mesh (or its armature) active.
    It stores the mesh names so that 'Bind to Selected Bones' can find them
    after you switch to Pose Mode and pick bones."""
    bl_idname = "object.smplx_remember_meshes"
    bl_label  = "Remember Selected Meshes"
    bl_description = (
        "Step 1/2 for partial-bone binding. "
        "Select the mesh(es) you want to attach, Shift-click the SMPL-X body "
        "mesh or armature (active), then click this button. "
        "Then go to Pose Mode, select the bones you want, and click "
        "'Bind Mesh to Selected Bones'."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            active = context.active_object
            return (
                active is not None and
                context.mode == 'OBJECT' and
                (active.type == 'ARMATURE' or
                 (active.type == 'MESH' and active.parent and
                  active.parent.type == 'ARMATURE')) and
                len(context.selected_objects) >= 2
            )
        except:
            return False

    def execute(self, context):
        active = context.active_object
        # Resolve SMPL-X family (armature + its children)
        if active.type == 'ARMATURE':
            arm = active
        else:
            arm = active.parent
        family = {arm} | set(arm.children)

        meshes = [
            obj.name for obj in context.selected_objects
            if obj not in family and obj.type == 'MESH'
        ]
        if not meshes:
            self.report({'ERROR'}, "No target mesh found in selection.")
            return {'CANCELLED'}

        context.window_manager.smplx_tool.smplx_pending_meshes = ",".join(meshes)
        self.report({'INFO'},
            f"Remembered: {meshes}. "
            f"Now go to Pose Mode, select bones, and click 'Bind Mesh to Selected Bones'.")
        return {'FINISHED'}


class SMPLXBindToSelectedBones(bpy.types.Operator):
    """Step 2 of partial-bone binding. Call this in Pose Mode with the desired
    bones selected on the SMPL-X armature. Transfers weights from ONLY those
    bones to every mesh remembered in Step 1, un-poses them via inverse LBS,
    and adds a driven Armature modifier — identical math to the full garment
    bind but limited to the user-chosen bone subset."""
    bl_idname = "object.smplx_bind_selected_bones"
    bl_label  = "Bind Mesh to Selected Bones"
    bl_description = (
        "Step 2/2 for partial-bone binding. "
        "Must be called in Pose Mode with the desired bones selected. "
        "Binds every mesh from Step 1 using ONLY the selected bones."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (
                context.mode == 'POSE' and
                context.active_object is not None and
                context.active_object.type == 'ARMATURE' and
                len(context.selected_pose_bones) > 0 and
                context.window_manager.smplx_tool.smplx_pending_meshes != ""
            )
        except:
            return False

    # KD-tree weight transfer limited to a specific set of bone names
    def _transfer_weights_partial(self, context, smplx_mesh, smplx_armature,
                                   garment, allowed_bones):
        import mathutils

        depsgraph     = context.evaluated_depsgraph_get()
        body_eval     = smplx_mesh.evaluated_get(depsgraph)
        body_eval_mesh = body_eval.to_mesh()
        body_wmat     = smplx_mesh.matrix_world

        kd = mathutils.kdtree.KDTree(len(body_eval_mesh.vertices))
        for i, v in enumerate(body_eval_mesh.vertices):
            kd.insert(body_wmat @ v.co, i)
        kd.balance()

        body_vg_name = {vg.index: vg.name for vg in smplx_mesh.vertex_groups}
        all_bone_names = {b.name for b in smplx_armature.data.bones}

        body_weights = [[] for _ in range(len(smplx_mesh.data.vertices))]
        for vi, bv in enumerate(smplx_mesh.data.vertices):
            for vge in bv.groups:
                name = body_vg_name.get(vge.group)
                # Only include bones the user explicitly selected
                if name and name in allowed_bones and vge.weight > 1e-6:
                    body_weights[vi].append((name, vge.weight))

        # Remove old bone vertex groups from garment, create fresh ones
        for vg in list(garment.vertex_groups):
            if vg.name in all_bone_names:
                garment.vertex_groups.remove(vg)
        needed_bones = {
            name for wlist in body_weights for (name, _) in wlist
        }
        for bname in needed_bones:
            garment.vertex_groups.new(name=bname)

        gar_vg   = {vg.name: vg for vg in garment.vertex_groups}
        gar_wmat = garment.matrix_world

        n_weighted = 0
        for gi, gv in enumerate(garment.data.vertices):
            gv_world = gar_wmat @ gv.co
            _, bi, _ = kd.find(gv_world)
            wlist = body_weights[bi]
            if not wlist:
                continue
            total = sum(w for (_, w) in wlist)
            for (bname, w) in wlist:
                vg_obj = gar_vg.get(bname)
                if vg_obj:
                    vg_obj.add([gi], w / total, 'REPLACE')
            n_weighted += 1

        body_eval.to_mesh_clear()
        return n_weighted

    # Inverse LBS — identical to SMPLXBindGarmentToBody._inverse_lbs
    def _inverse_lbs(self, garment, smplx_armature):
        import numpy as np
        from mathutils import Vector

        arm_mat     = np.array(smplx_armature.matrix_world, dtype=np.float64)
        gar_mat     = np.array(garment.matrix_world,         dtype=np.float64)
        gar_mat_inv = np.linalg.inv(gar_mat)

        bone_def = {}
        for pbone in smplx_armature.pose.bones:
            rest_w  = arm_mat @ np.array(pbone.bone.matrix_local, dtype=np.float64)
            posed_w = arm_mat @ np.array(pbone.matrix,             dtype=np.float64)
            bone_def[pbone.name] = posed_w @ np.linalg.inv(rest_w)

        vg_def = {
            vg.index: bone_def[vg.name]
            for vg in garment.vertex_groups
            if vg.name in bone_def
        }

        n_moved = 0
        mesh = garment.data
        for vert in mesh.vertices:
            v_w = gar_mat @ np.array([*vert.co, 1.0], dtype=np.float64)
            M       = np.zeros((4, 4), dtype=np.float64)
            total_w = 0.0
            for vge in vert.groups:
                D = vg_def.get(vge.group)
                if D is None:
                    continue
                M       += vge.weight * D
                total_w += vge.weight
            if total_w < 1e-6:
                continue
            M /= total_w
            try:
                v_rest_w = np.linalg.inv(M) @ v_w
            except np.linalg.LinAlgError:
                continue
            vert.co = Vector((gar_mat_inv @ v_rest_w)[:3])
            n_moved += 1
        mesh.update()
        return n_moved

    def execute(self, context):
        import traceback

        smplx_armature = context.active_object   # we are in Pose Mode
        allowed_bones  = {pb.name for pb in context.selected_pose_bones}

        # Resolve the driven SMPL-X body mesh (first child mesh of the armature)
        smplx_mesh = next(
            (c for c in smplx_armature.children if c.type == 'MESH'), None
        )
        if smplx_mesh is None:
            self.report({'ERROR'}, "No mesh child found on the armature.")
            return {'CANCELLED'}

        pending = context.window_manager.smplx_tool.smplx_pending_meshes
        mesh_names = [n.strip() for n in pending.split(',') if n.strip()]
        garments = [
            bpy.data.objects.get(n) for n in mesh_names
            if bpy.data.objects.get(n) is not None
        ]
        if not garments:
            self.report({'ERROR'},
                "No valid pending meshes. Run 'Remember Selected Meshes' first.")
            return {'CANCELLED'}

        bound  = []
        failed = []
        for garment in garments:
            try:
                # Remove previous partial-binding modifiers
                for mod in list(garment.modifiers):
                    if mod.name.startswith("Partial_"):
                        garment.modifiers.remove(mod)

                n_w = self._transfer_weights_partial(
                    context, smplx_mesh, smplx_armature, garment, allowed_bones)
                if n_w == 0:
                    raise RuntimeError(
                        "No vertices received weights from the selected bones. "
                        "Is the mesh near those bones?"
                    )

                n_m = self._inverse_lbs(garment, smplx_armature)
                if n_m == 0:
                    raise RuntimeError("Inverse LBS moved 0 vertices.")

                arm_mod                   = garment.modifiers.new("Partial_Armature", 'ARMATURE')
                arm_mod.object            = smplx_armature
                arm_mod.use_vertex_groups = True

                cs_mod                 = garment.modifiers.new("Partial_Smooth", 'CORRECTIVE_SMOOTH')
                cs_mod.factor          = 0.25
                cs_mod.iterations      = 5
                cs_mod.smooth_type     = 'SIMPLE'
                cs_mod.use_only_smooth = False

                bound.append(garment.name)

            except Exception as e:
                failed.append(garment.name)
                print(f"[Partial Bind] Error on '{garment.name}': {e}")
                traceback.print_exc()

        # Clear pending list after binding
        context.window_manager.smplx_tool.smplx_pending_meshes = ""

        if bound:
            self.report({'INFO'},
                f"Bound [{', '.join(bound)}] to bones: {sorted(allowed_bones)}. ")
        if failed:
            self.report({'WARNING'},
                f"Failed: [{', '.join(failed)}]. Check the Blender console.")
        return {'FINISHED'}


class SMPLXBindGarmentToBody(bpy.types.Operator):
    bl_idname = "object.smplx_bind_garment_to_body"
    bl_label = "Bind Garment To SMPL-X Body"
    bl_description = (
        "Bind selected garment/dress to the SMPL-X body using inverse LBS — "
        "the same math SMPL-X uses internally, run backwards. "
        "HOW TO USE: (1) Select the garment mesh(es). "
        "(2) Shift-click the SMPL-X BODY MESH to make it active. "
        "(3) Click this button at any frame. "
        "No shrinkwrap. No proxy. No physics. Dress animates identically to the body."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            active = context.active_object
            return (
                active is not None and
                active.type == 'MESH' and
                active.parent is not None and
                active.parent.type == 'ARMATURE' and
                len(context.selected_objects) >= 2
            )
        except:
            return False

    # ---------------------------------------------------------------------- #
    # Transfer bone weights from body to garment using a KD-tree.             #
    # Both are at the same pose so nearest-vertex lookup is spatially correct.#
    # Uses ONLY the depsgraph and direct mesh data — no bpy.ops calls at all, #
    # so it never fails silently inside another operator's execute().          #
    # ---------------------------------------------------------------------- #
    def _transfer_weights(self, context, smplx_mesh, smplx_armature, garment):
        import mathutils   # for kdtree

        depsgraph = context.evaluated_depsgraph_get()

        # ── Evaluated SMPL-X body positions at current frame (world space) ──
        body_eval     = smplx_mesh.evaluated_get(depsgraph)
        body_eval_mesh = body_eval.to_mesh()
        body_wmat     = smplx_mesh.matrix_world

        kd = mathutils.kdtree.KDTree(len(body_eval_mesh.vertices))
        for i, v in enumerate(body_eval_mesh.vertices):
            kd.insert(body_wmat @ v.co, i)
        kd.balance()

        # ── Map vertex-group index → bone name on the BODY mesh ──
        body_vg_name = {vg.index: vg.name for vg in smplx_mesh.vertex_groups}
        # Bone names actually present in the armature
        bone_names   = {b.name for b in smplx_armature.data.bones}

        # Finger joints to exclude — wrist is the terminal influence for garments.
        # Any bone whose name contains one of these substrings is skipped entirely.
        _FINGER_KEYWORDS = ('index', 'middle', 'pinky', 'ring', 'thumb', 'jaw',
                            'left_eye', 'right_eye')
        def _is_finger(name):
            nl = name.lower()
            return any(k in nl for k in _FINGER_KEYWORDS)

        # ── Per-body-vertex weight list (built once, indexed by body vert idx) ──
        # body_weights[i] = list of (bone_name, weight)
        body_weights = [[] for _ in range(len(smplx_mesh.data.vertices))]
        for vi, bv in enumerate(smplx_mesh.data.vertices):
            for vge in bv.groups:
                name = body_vg_name.get(vge.group)
                if name and name in bone_names and vge.weight > 1e-6 and not _is_finger(name):
                    body_weights[vi].append((name, vge.weight))

        # ── Remove old bone vertex groups from garment, create fresh ones ──
        for vg in list(garment.vertex_groups):
            if vg.name in bone_names:
                garment.vertex_groups.remove(vg)
        # Pre-create vertex groups for every bone that appears in body weights
        needed_bones = {
            name
            for wlist in body_weights for (name, _) in wlist
        }
        for bname in needed_bones:
            garment.vertex_groups.new(name=bname)

        # ── Build garment vertex-group lookup by name ──
        gar_vg = {vg.name: vg for vg in garment.vertex_groups}

        # ── Garment world matrix for lookup ──
        gar_wmat = garment.matrix_world

        # ── For each garment vertex, find nearest body vertex, copy weights ──
        n_weighted = 0
        for gi, gv in enumerate(garment.data.vertices):
            gv_world = gar_wmat @ gv.co
            _, bi, _ = kd.find(gv_world)     # nearest body vertex index
            wlist = body_weights[bi]
            if not wlist:
                continue
            total = sum(w for (_, w) in wlist)
            for (bname, w) in wlist:
                vg_obj = gar_vg.get(bname)
                if vg_obj:
                    vg_obj.add([gi], w / total, 'REPLACE')
            n_weighted += 1

        body_eval.to_mesh_clear()
        print(f"[SMPL-X Garment] Weight transfer: {n_weighted}/{len(garment.data.vertices)} vertices weighted.")
        return n_weighted

    # ---------------------------------------------------------------------- #
    # Inverse LBS: move every garment vertex from dance-pose back to T-pose.  #
    # Forward LBS: v_posed = sum_b(w_b * D_b * v_rest)                        #
    # Inverse:     v_rest  = inv(sum_b(w_b * D_b)) * v_posed                  #
    # ---------------------------------------------------------------------- #
    def _inverse_lbs(self, garment, smplx_armature):
        import numpy as np
        from mathutils import Vector

        arm_mat     = np.array(smplx_armature.matrix_world, dtype=np.float64)
        gar_mat     = np.array(garment.matrix_world,         dtype=np.float64)
        gar_mat_inv = np.linalg.inv(gar_mat)

        # D_b = posed_world_b @ inv(rest_world_b)
        bone_def = {}
        for pbone in smplx_armature.pose.bones:
            rest_w  = arm_mat @ np.array(pbone.bone.matrix_local, dtype=np.float64)
            posed_w = arm_mat @ np.array(pbone.matrix,             dtype=np.float64)
            bone_def[pbone.name] = posed_w @ np.linalg.inv(rest_w)

        # Map garment vertex-group index → D_b matrix
        vg_def = {
            vg.index: bone_def[vg.name]
            for vg in garment.vertex_groups
            if vg.name in bone_def
        }

        n_moved = 0
        mesh = garment.data
        for vert in mesh.vertices:
            v_w = gar_mat @ np.array([*vert.co, 1.0], dtype=np.float64)

            M       = np.zeros((4, 4), dtype=np.float64)
            total_w = 0.0
            for vge in vert.groups:
                D = vg_def.get(vge.group)
                if D is None:
                    continue
                M       += vge.weight * D
                total_w += vge.weight

            if total_w < 1e-6:
                continue

            M /= total_w
            try:
                v_rest_w   = np.linalg.inv(M) @ v_w
            except np.linalg.LinAlgError:
                continue
            v_rest_local = gar_mat_inv @ v_rest_w
            vert.co      = Vector(v_rest_local[:3])
            n_moved += 1

        mesh.update()
        print(f"[SMPL-X Garment] Inverse LBS: {n_moved}/{len(mesh.vertices)} vertices un-posed to T-pose.")
        return n_moved

    def execute(self, context):
        import traceback

        smplx_mesh     = context.active_object
        smplx_armature = smplx_mesh.parent
        if smplx_armature is None or smplx_armature.type != 'ARMATURE':
            self.report({'ERROR'}, "SMPL-X body mesh must be parented to an armature.")
            return {'CANCELLED'}

        garments = [
            obj for obj in context.selected_objects
            if obj != smplx_mesh and obj.type == 'MESH'
        ]
        if not garments:
            self.report({'ERROR'},
                "No garment found. Select garment(s) first, "
                "then Shift-click the SMPL-X body mesh.")
            return {'CANCELLED'}

        bound  = []
        failed = []

        for garment in garments:
            try:
                # Remove previous garment modifiers
                for mod in list(garment.modifiers):
                    if mod.name.startswith("Garment_"):
                        garment.modifiers.remove(mod)

                # ── Step 1: Transfer bone weights (KD-tree, no bpy.ops) ──────
                n_w = self._transfer_weights(context, smplx_mesh, smplx_armature, garment)
                if n_w == 0:
                    raise RuntimeError(
                        "No garment vertices received weights. "
                        "Is the garment near the SMPL-X body?"
                    )

                # ── Step 2: Inverse LBS → move dress to T-pose ───────────────
                n_m = self._inverse_lbs(garment, smplx_armature)
                if n_m == 0:
                    raise RuntimeError("Inverse LBS moved 0 vertices. Check console.")

                # ── Step 3: Armature modifier → original SMPL-X armature ─────
                # Dress is now at T-pose with correct weights. The Armature
                # modifier applies forward LBS from T-pose → every dance frame.
                arm_mod                  = garment.modifiers.new("Garment_Armature", 'ARMATURE')
                arm_mod.object           = smplx_armature
                arm_mod.use_vertex_groups = True

                # ── Step 4: Corrective Smooth ────────────────────────────────
                cs_mod                 = garment.modifiers.new("Garment_Smooth", 'CORRECTIVE_SMOOTH')
                cs_mod.factor          = 0.25
                cs_mod.iterations      = 5
                cs_mod.smooth_type     = 'SIMPLE'
                cs_mod.use_only_smooth = False

                bound.append(garment.name)

            except Exception as e:
                failed.append(garment.name)
                print(f"[SMPL-X Garment] Error binding '{garment.name}': {e}")
                traceback.print_exc()

        if bound:
            self.report({'INFO'},
                f"Bound [{', '.join(bound)}] via inverse LBS. "
                f"Dress now animates like the SMPL-X body.")
        if failed:
            self.report({'WARNING'},
                f"Failed: [{', '.join(failed)}]. Check the Blender console.")
        return {'FINISHED'}


class SMPLX_PT_Rigging(bpy.types.Panel):
    bl_label = "Rigging"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        # ── WORKFLOW A: Garment / Dress ──────────────────────────────────── #
        col.label(text="Garment / Dress:", icon='MOD_CLOTH')
        box = col.box()
        bcol = box.column(align=True)
        bcol.label(text="How to use:", icon='INFO')
        bcol.label(text="1. Select dress mesh(es).")
        bcol.label(text="2. Shift-click BODY mesh.")
        bcol.label(text="3. Press the button below.")
        bcol.label(text="Works at any frame.")
        col.operator(
            "object.smplx_bind_garment_to_body",
            text="Bind Garment to Body",
            icon='MOD_ARMATURE'
        )
        col.separator()

        # ── WORKFLOW B: Stickman / rigid object ──────────────────────────── #
        col.label(text="Rigid Object:", icon='ARMATURE_DATA')
        box = col.box()
        bcol = box.column(align=True)
        bcol.label(text="How to use:", icon='INFO')
        bcol.label(text="1. Go to frame where mesh")
        bcol.label(text="   aligns with the BODY mesh.")
        bcol.label(text="2. Select mesh.")
        bcol.label(text="3. Shift-click armature.")
        bcol.label(text="4. Press the button below.")
        col.operator(
            "object.smplx_parent_custom_mesh",
            text="Parent Mesh to Armature",
            icon='LINKED'
        )
        col.separator()

        # ── WORKFLOW C: Partial / custom bone selection ───────────────────── #
        col.label(text="Partial Bone Binding:", icon='BONE_DATA')
        box = col.box()
        bcol = box.column(align=True)
        bcol.label(text="For torso/limb accessories — bind", icon='INFO')
        bcol.label(text="to only the bones you pick.")
        bcol.separator()
        bcol.label(text="Step 1 (Object Mode):")
        bcol.label(text="  Select target mesh(es).")
        bcol.label(text="  Shift-click body mesh/armature.")
        col.operator(
            "object.smplx_remember_meshes",
            text="1. Remember Selected Meshes",
            icon='PINNED'
        )
        pending = context.window_manager.smplx_tool.smplx_pending_meshes
        if pending:
            box2 = col.box()
            box2.label(text=f"Pending: {pending}", icon='CHECKMARK')
        col.separator()
        bcol2 = col.column(align=True)
        bcol2.label(text="Step 2 — LBS (Pose Mode):")
        bcol2.label(text="  Select bones you want.")
        col.operator(
            "object.smplx_bind_selected_bones",
            text="2a. Bind via Bone Weights (LBS)",
            icon='MOD_ARMATURE'
        )
        col.separator()

        # ── WORKFLOW D: Rigid attachment (face, accessories) ──────────────── #
        col.label(text="Rigid Attachment (Face / Accessories):", icon='FACE_MAPS')
        box = col.box()
        bcol = box.column(align=True)
        bcol.label(text="For face mesh, glasses, earrings,", icon='INFO')
        bcol.label(text="hat, hair — ZERO deformation.")
        bcol.label(text="Uses Child Of constraint, NOT weights.")
        bcol.label(text="No scars, no LBS, perfectly rigid.")
        bcol.separator()
        bcol.label(text="Step 1: same 'Remember' button above.")
        bcol.label(text="Step 2 (Pose Mode): select 1 bone")
        bcol.label(text="  (e.g. 'head'), click below.")
        col.operator(
            "object.smplx_rigid_attach_to_bone",
            text="2b. Attach Rigidly to Bone",
            icon='CONSTRAINT_BONE'
        )

class SMPLX_PT_Export(bpy.types.Panel):
    bl_label = "Export"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.operator("object.smplx_export_alembic")
        col.separator()

        col.operator("object.smplx_export_fbx")
        col.separator()

        col.operator("object.smplx_export_shape")
        col.separator()

#        export_button = col.operator("export_scene.obj", text="Export OBJ [m]", icon='EXPORT')
#        export_button.global_scale = 1.0
#        export_button.use_selection = True
#        col.separator()

        row = col.row(align=True)
        row.operator("ed.undo", icon='LOOP_BACK')
        row.operator("ed.redo", icon='LOOP_FORWARDS')
        col.separator()

        (year, month, day) = bl_info["version"]
        col.label(text="Version: %s-%s-%s" % (year, month, day))

classes = [
    PG_SMPLXProperties,
    SMPLXAddGender,
    SMPLXSetTexture,
    SMPLXMeasurementsToShape,
    SMPLXRandomShape,
    SMPLXResetShape,
    SMPLXRandomExpressionShape,
    SMPLXResetExpressionShape,
    SMPLXSnapGroundPlane,
    SMPLXUpdateJointLocations,
    SMPLXSetPoseshapes,
    SMPLXResetPoseshapes,
    SMPLXSetHandpose,
    SMPLXWritePose,
    SMPLXLoadPose,
    SMPLXResetPose,
    SMPLXParentCustomMesh,
    SMPLXRememberMeshes,
    SMPLXBindToSelectedBones,
    SMPLXRigidAttachToBone,
    SMPLXBindGarmentToBody,
    SMPLXAddAnimation,
    SMPLXExportAlembic,
    SMPLXExportFBX,
    SMPLXExportShape,
    SMPLX_PT_Model,
    SMPLX_PT_Shape,
    SMPLX_PT_Pose,
    SMPLX_PT_Animation,
    SMPLX_PT_Rigging,
    SMPLX_PT_Export
]

def register():
    from bpy.utils import register_class
    for cls in classes:
        bpy.utils.register_class(cls)

    # Store properties under WindowManager (not Scene) so that they are not saved in .blend files and always show default values after loading
    bpy.types.WindowManager.smplx_tool = PointerProperty(type=PG_SMPLXProperties)

def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.smplx_tool

if __name__ == "__main__":
    register()

import cv2
import os
import json
import torch
import smplx
import trimesh
import math
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, DETECTRON_CKPT, DETECTRON_CFG
from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

def rotation_matrix_to_axis_angle(R):
    """
    R: [B, N, 3, 3] rotation matrices
    returns: [B, N, 3] axis-angle vectors
    """
    # Compute rotation angle
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # [B, N]

    # Compute rotation axis
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)  # [B, N, 3]

    # Normalize and multiply by angle
    eps = 1e-8
    sin_theta = torch.sin(theta).unsqueeze(-1)
    k = r / (2 * sin_theta + eps)

    # handle small angles (near zero)
    mask = (theta < 1e-5).unsqueeze(-1)
    k = torch.where(mask, r * 0.5, k)

    axis_angle = k * theta.unsqueeze(-1)
    return axis_angle

def compute_masked_mesh(mask, vertices, faces):
    # mask: shape (1, 6890), convert to 1D boolean
    mask = mask.reshape(-1).astype(bool)

    # Keep only the vertices that are True
    new_vertices = vertices[mask]

    # Map old vertex indices to new ones
    old_to_new = -np.ones(len(vertices), dtype=int)
    old_to_new[mask] = np.arange(np.sum(mask))

    # Filter faces: keep only those with all vertices inside the mask
    new_faces = []
    for f in faces:
        if mask[f].all():  # all 3 vertices must be included
            new_faces.append(old_to_new[f])
    new_faces = np.array(new_faces)

    # Build new mesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    return new_mesh

class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model
    
    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    
    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img, fov=None):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        if fov is None:
            estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
            # print(f"Estimated FOVs: {estimated_fov}")
            vfov = estimated_fov[0, 1]
        else:
            vfov = torch.tensor(fov)

        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()

        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)

        return cam_int, math.degrees(vfov)


    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)

    def looking_direction(self, facing_direction, cam_vector = np.array([0,0,1])):
        facing_direction[1] = 0
        cam_vector[1] = 0
        facing_direction = facing_direction / np.linalg.norm(facing_direction)
        cam_vector = cam_vector / np.linalg.norm(cam_vector)
        # print(f"Facing direction (normalized): {facing_direction} Cam vector (normalized): {cam_vector}")
        dot_product = np.dot(facing_direction, cam_vector)
        angle = np.arccos(dot_product)
        cross_product = np.cross(cam_vector, facing_direction)
        if cross_product[1] < 0:
            angle = -angle
        return angle


    def process_image(self, img_path, output_img_folder, render_overlay = False, fov=None, save_partial_mesh=False):
        img_full = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img_full is not None and img_full.shape[2] == 4:
            img_cv2 = img_full[:, :, :3]
            alpha = img_full[:, :, 3]
            # dilate the alpha channel to create a foreground mask
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8))
            alpha = alpha / 255.0
            foreground_mask = alpha > 0.1
        else:
            img_cv2 = img_full
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        fname = f"{img_path.split('/')[-2]}_{fname}"
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}{img_ext}')
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}.obj')

        # extra saves
        smpl_params_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_smpl.npz')

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Get Camera intrinsics using HumanFoV Model
        cam_int, vfov = self.get_cam_intrinsics(img_cv2, fov)

        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

        for batch in dataloader:

            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            global_orient = rotation_matrix_to_axis_angle(out_smpl_params['global_orient']).view(-1, 3)
            body_pose = rotation_matrix_to_axis_angle(out_smpl_params['body_pose']).view(-1, 23*3)
            
            # save SMPL parameters out_s
            np.savez(smpl_params_fname,
                global_orient=global_orient.cpu().numpy(),
                body_pose=body_pose.cpu().numpy(),
                betas=out_smpl_params['betas'].cpu().numpy(),
            )

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
            

            # mesh = trimesh.Trimesh((output_vertices[0] + output_cam_trans[0].reshape(-1,3)).detach().cpu().numpy(), self.smpl_model.faces,
            #                 process=False)
            mesh = trimesh.Trimesh((output_vertices[0]).detach().cpu().numpy(), self.smpl_model.faces,
                            process=False)

            ## ONLY FOR DEBUGGING
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh.export(mesh_fname)

            if save_partial_mesh:
                projected_mesh_vertices = (cam_int @ mesh.vertices.T).T
                projected_mesh_vertices = projected_mesh_vertices[:, :2] / projected_mesh_vertices[:, 2:].reshape(-1, 1)
                # flip horizontally I have no idea why I had to do this, but it works
                projected_mesh_vertices[:, 0] = 512 - projected_mesh_vertices[:, 0]

                points = np.rint(projected_mesh_vertices).astype(int)
                # clip points to be within image bounds
                points[:, 0] = np.clip(points[:, 0], 0, 512-1)
                points[:, 1] = np.clip(points[:, 1], 0, 512-1)

                overlap_mask = foreground_mask[points[:, 1], points[:, 0]]

                partial_mesh = compute_masked_mesh(overlap_mask, mesh.vertices, mesh.faces)
                
                partial_mesh.export(mesh_fname.replace('.obj', '_partial.obj'))
                # cv2.imwrite(mesh_fname.replace('.obj', '_mask.png'), overlap_mask.astype(np.uint8)*255)
                # points = np.clip(points, 0, 511)

                # img = np.zeros((512, 512, 3), dtype=np.uint8)

                # # copy foreground_mask and make it white
                # img[foreground_mask] = [255, 255, 255]

                # # 3. Draw each point as a red dot with radius
                # radius = 2
                # color = (0, 0, 255)  # Red in BGR
                # for (x, y) in points:
                #     cv2.circle(img, (x, y), radius, color, -1)

                # # 4. Save the image
                # cv2.imwrite(mesh_fname.replace('.obj', '_partial_mesh.png'), img)
                # breakpoint()

            if render_overlay is False:
                continue
            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.smpl_model.faces, same_mesh_color=True)
            front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=img_cv2.copy())
            final_img = front_view
            # Write overlay
            cv2.imwrite(overlay_fname, final_img)
            renderer.delete()

    def return_front_frame(self, img_dir_path, start_frame, num_views=8, fov=None):
        img_path = os.path.join(img_dir_path, f"{start_frame:03d}.png")
        if not os.path.exists(img_path):
            print(f"WARNING: Image {start_frame:03d}.png not found in {img_dir_path}. Something went wrong...")
            return None
        img_cv2 = cv2.imread(img_path)

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        if valid_idx.sum().item() == 0:
            return None # No humans detected in this frame
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Get Camera intrinsics using HumanFoV Model
        cam_int, vfov = self.get_cam_intrinsics(img_cv2, fov)

        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

        for batch in dataloader:

            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
                            
            facing_direction = output_vertices[0].cpu().numpy()[5267] - output_joints[0][0].cpu().numpy()

            # we rotate facing_direction with 180 degrees around x axis to align with camera coordinate system
            # the same is done for SMPL during the output mesh generation
            facing_direction[1] = -facing_direction[1]
            facing_direction[2] = -facing_direction[2]

            angle = self.looking_direction(facing_direction)

            # print(f"Angle (in degrees): {math.degrees(angle)}")

            if angle < 0:
                angle = 360 - abs(math.degrees(angle))
            else:
                angle = math.degrees(angle)

            per_view_angle = 360 / num_views

            front_view_id = angle / per_view_angle
            # print(f"Front view ID (before rounding): {front_view_id}")
            front_view_id = (int(round(front_view_id)) + start_frame) % num_views
            return front_view_id

    def run_on_directory(self, parent_folder, out_folder, num_views= 8, render_overlay = False, vfov=None, save_partial_mesh=False):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        for image_dir in os.listdir(parent_folder):

            print(f"Processing directory: {image_dir}")
            image_dir_full = os.path.join(parent_folder, image_dir)

            if not os.path.isdir(image_dir_full):
                continue

            first_image = os.path.join(image_dir_full, '000.png')

            if not os.path.exists(first_image):
                print(f"WARNING: First image not found in {image_dir_full}. Something went wrong...")
                continue
            
            for start_view in range(num_views):

                front_view_id = self.return_front_frame(image_dir_full, start_view, num_views, vfov)

                if front_view_id is not None:
                    break
                else:
                    print(f"No humans detected in frame {start_view:03d}.png. Trying next frame...")
            
            if front_view_id is None:
                print(f"WARNING: No humans detected in any of the first {num_views} frames in {image_dir_full}. Skipping this directory...")
                continue

            front_image_path = os.path.join(image_dir_full, f"{front_view_id:03d}.png")

            print(f"Selected front view id: {front_view_id}")

            # print(f"For the directory {image_dir}, the front view id is {front_view_id}")
            if not os.path.exists(front_image_path):
                print(f"WARNING: Front view image {front_view_id:03d}.png not found in {image_dir_full}. Something went wrong...")
                continue

            self.process_image(front_image_path, out_folder, render_overlay, vfov, save_partial_mesh=save_partial_mesh)
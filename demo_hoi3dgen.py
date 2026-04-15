import argparse
from mesh_estimator_hoi3d import HumanMeshEstimator
import math


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    parser.add_argument("--input_folder", "--input_folder", type=str, default="",
        help="Path to input image folder.")
    parser.add_argument("--output_folder", "--output_folder", type=str, default="",
        help="Path to folder output folder.")
    parser.add_argument("--num_views", "--num_views", type=int, default=8,
        help="Number of views per object.")
    parser.add_argument("--save_partial_mesh", type=bool, default=False,
        help="Whether to save the partial SMPL mesh or not.")
    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator()
    vfov = 40/180 * math.pi  # Set a fixed vertical field of view (in radians)
    estimator.run_on_directory(args.input_folder, args.output_folder, args.num_views, render_overlay=True, vfov=vfov, save_partial_mesh=args.save_partial_mesh)

if __name__=='__main__':
    main()


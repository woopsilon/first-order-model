from concurrent.futures import process
from pickle import FALSE
import matplotlib
matplotlib.use('Agg')
import os, psutil, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

import warnings
warnings.filterwarnings("ignore")

import subprocess

from PIL import Image

import face_alignment
import time
import gc

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

def normalize_kp_facealignment(kp):
    kp = kp - kp.mean(axis=0, keepdims=True)
    area = ConvexHull(kp[:, :2]).volume
    area = np.sqrt(area)
    kp[:, :2] = kp[:, :2] / area
    return kp

class PrettyPrinter(object):
    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

class SourceConfig(PrettyPrinter):
  def __init__(self, path, prefix, use_best_frame, use_first_frame):
    self.path = path
    self.prefix = prefix
    self.use_best_frame = use_best_frame
    self.use_first_frame = use_first_frame

class DrivingVideo(PrettyPrinter):
  def __init__(self, path, fps, len, driving_kps):
    self.path = path
    self.fps = fps
    self.len = len
    self.driving_kps = driving_kps

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", default='testing.yaml', help="path to inputs")
    opt = parser.parse_args()
    print(f"Loading input from '{opt.input}'")
    with open(opt.input) as f:
        input = yaml.load(f, Loader=yaml.FullLoader)
    with open("classifications.yaml") as f:
        classifications = yaml.load(f, Loader=yaml.FullLoader)

    # Extract vars
    result_suffix = input["result_suffix"]
    if (result_suffix is None):
        result_suffix = ""
    intermediates_dir = input["intermediates"]
    output_dir = input["output"]
    preview = input["preview"]
    preview_length_seconds = input["preview_length_seconds"]
    source_hints = input["source_hints"]
    source_dirs = input["source_dirs"]
    driving_videos = input["driving_videos"]
    combined = input["combined"]
    add_relative = input["add_relative"]
    add_untransformed = input["add_untransformed"]
    add_relative_and_adapted = input["add_relative_and_adapted"]
    find_best_frame_seconds = input["find_best_frame_seconds"]
    use_best_frame = input["use_best_frame"]
    use_first_frame = input["use_first_frame"]
    debugging = input["debugging"]
    if (debugging is None):
        debugging = False
    if (not(use_best_frame) and not(use_first_frame)):
        raise Exception("At least one of 'use_best_frame' and 'use_first_frame' must be True")

    if (debugging):
        process = psutil.Process(os.getpid())
        print(f"Starting memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    keypoints_dir = "keypoints"

    # Set up dirs
    if (not(os.path.exists(intermediates_dir))):
        os.makedirs(intermediates_dir)
    if (not(os.path.exists(output_dir))):
        os.makedirs(output_dir)
    if (not(os.path.exists(keypoints_dir))):
        os.makedirs(keypoints_dir)

    source_img_configs = {}
    if (not(source_dirs is None)):
        for dir in source_dirs:
            for im in os.listdir(dir):
                print(f"Discovered source image: {im}")
                path = f"{dir}/{im}".replace("\\", "/")
                source_img_configs[path] = SourceConfig(path, "", use_best_frame, use_first_frame)
    if (not(source_hints is None)):
        for im_config in source_hints:
            path = im_config["path"].replace("\\", "/")
            if ("use_best_frame" in im_config):
                cfg_use_best_frame = im_config["use_best_frame"]
            else:
                cfg_use_best_frame = use_best_frame
            if ("use_first_frame" in im_config):
                cfg_use_first_frame = im_config["use_first_frame"]
            else:
                cfg_use_first_frame = use_first_frame
            if ("prefix" in im_config):
                cfg_prefix = im_config["prefix"]
            else:
                cfg_prefix = ""
            config = SourceConfig(path, cfg_prefix, cfg_use_best_frame, cfg_use_first_frame)
            print(f"Found source image hint for: {config.path}")
            source_img_configs[config.path] = config

    for im_config in source_img_configs:
        print(f"{source_img_configs[im_config]}")
    
    # Load checkpoint
    print("Loading checkpoints...")
    generator, kp_detector = load_checkpoints(config_path="config/vox-adv-256.yaml", checkpoint_path="models/vox-adv-cpk.pth.tar", cpu=False)
    if (debugging):
        process = psutil.Process(os.getpid())
        print(f"Memory after loading checkpoints: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                        device='cuda')

    if (debugging):
        process = psutil.Process(os.getpid())
        print(f"Memory after loading FaceAlignment: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print("Verifying driving videos...")
    driving_video_infos = []
    for path in driving_videos:
        print(f"Verifying video '{path}'...")
        reader = imageio.get_reader(path)
        fps = reader.get_meta_data()['fps']
        driving_video_width, driving_video_height = reader.get_meta_data()['size']
        if (driving_video_width != driving_video_height):
            sys.exit(f"Driving video width ({driving_video_width}) is not the same as driving video height ({driving_video_height})")
        print("Driving video verified.")
        driving_video_basename = os.path.basename(path)
        if (find_best_frame_seconds == -1):
            keypoints_filename = f"{keypoints_dir}/{os.path.basename(path)}_full.keypoints.txt.npy"
        else:
            keypoints_filename = f"{keypoints_dir}/{os.path.basename(path)}_{find_best_frame_seconds}.keypoints.txt.npy"
        if (not(os.path.exists(keypoints_filename))):
            if (find_best_frame_seconds == -1):
                print(f"Calculating keypoints for all of driving video..")
                drv_idx = 0
                driving_kps = []
                actual_len = len(reader)
                sys.stdout.write("\r    {:.1f}%   {} / {}   ".format(drv_idx / actual_len * 100.0, drv_idx, actual_len))
                for image in reader:
                    drv_idx = drv_idx + 1
                    im = resize(image, (256, 256), anti_aliasing=True)[..., :3]
                    kp_driving = fa.get_landmarks_from_image(255 * im)
                    if (kp_driving is None):
                        driving_kps.append(None)
                        continue
                    kp_driving = kp_driving[0]
                    driving_kps.append(kp_driving)
                    sys.stdout.write("\r    {:.1f}%   {} / {}   ".format(drv_idx / actual_len * 100.0, drv_idx, actual_len))
                sys.stdout.write("\r    100%\n")
            else:
                drv_idx = 0
                driving_kps = []
                actual_len = min(len(reader), find_best_frame_seconds * fps)
                print(f"Calculating keypoints for driving video's first {actual_len} seconds...")
                sys.stdout.write("\r    {:.1f}%   {} / {}   ".format(drv_idx / actual_len * 100.0, drv_idx, actual_len))
                for image in reader:
                    drv_idx = drv_idx + 1
                    im = resize(image, (256, 256), anti_aliasing=True)[..., :3]
                    kp_driving = fa.get_landmarks_from_image(255 * im)
                    if (kp_driving is None):
                        driving_kps.append(None)
                        continue
                    kp_driving = kp_driving[0]
                    driving_kps.append(kp_driving)
                    if ((drv_idx / fps) >= find_best_frame_seconds):
                        break
                    sys.stdout.write("\r    {:.1f}%   {} / {}   ".format(drv_idx / actual_len * 100.0, drv_idx, actual_len))
                sys.stdout.write("\r    100%\n")
            np.save(keypoints_filename, driving_kps)
        else:
            driving_kps = np.load(keypoints_filename)
        driving_video_infos.append(DrivingVideo(path, fps, len(reader), driving_kps))

    if (debugging):
        process = psutil.Process(os.getpid())
        print(f"Memory after loading KPs: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Check source images and driving video are both square
    source_image_data_name = "name"
    source_image_data_path = "path"
    source_image_data_im = "im"
    source_image_data_best_frame = "best_frame"
    source_image_data_best_frame_index = "best_frame_index"
    source_image_data_writers = "writers"
    source_image_data_intermediate_files = "intermediates"
    source_image_data_intermediate_dir = "intermediate_dir"
    source_image_data_out_files = "outs"
    source_image_data_first_frame = "first_frame"
    source_image_data_config = "config"
    source_image_data_diff = "normalized_diff"
    orig_filename_key = "intermediate_orig"
    rel_filename_key = "intermediate_rel"
    reladapt_filename_key = "intermediate_reladapt"
    combine_filename_key = "intermediate_combine"
    orig_first_filename_key = "intermediate_orig_first"
    rel_first_filename_key = "intermediate_rel_first"
    reladapt_first_filename_key = "intermediate_reladapt_first"
    combine_first_filename_key = "intermediate_combine_first"
    
    orig_out_filename_key = "orig"
    rel_out_filename_key = "rel"
    reladapt_out_filename_key = "reladapt"
    combine_out_filename_key = "combine"

    if (debugging):
        process = psutil.Process(os.getpid())
        print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    for drv_info in driving_video_infos:
        print(f"Processing '{drv_info.path}'...")
        all_source_images_data = []
        source_img_configs_idx = 0
        driving_video_basename = os.path.basename(drv_info.path)
        print("    Cleaning up old audio files...")
        if (os.path.exists(f"{intermediates_dir}/{driving_video_basename}_audio.mp3")):
            os.remove(f"{intermediates_dir}/{driving_video_basename}_audio.mp3")
        reader = imageio.get_reader(drv_info.path)
        kp_driving_cache = {}
        for src_key in source_img_configs.keys():
            im = source_img_configs[src_key].path
            prefix = source_img_configs[src_key].prefix
            base_name = os.path.basename(im)
            source_output_dir = f"{output_dir}\\{os.path.splitext(driving_video_basename)[0]}\\{os.path.splitext(base_name)[0]}\\{result_suffix}"
            source_intermediates_dir = f"{intermediates_dir}\\{os.path.splitext(driving_video_basename)[0]}\\{os.path.splitext(base_name)[0]}\\{result_suffix}"
            if (not(os.path.exists(f"{source_output_dir}"))):
                os.makedirs(f"{source_output_dir}")
            if (not(os.path.exists(f"{source_intermediates_dir}"))):
                os.makedirs(f"{source_intermediates_dir}")
            source_image_data = {}
            source_image = Image.open(im)
            source_image_width, source_image_height = source_image.size
            print(f"    Preprocessing source image #{source_img_configs_idx+1} '{im}'...")
            if (source_image_width != source_image_height):
                sys.exit(f"    Source image ({im}) width ({source_image_width}) is not the same as source image height ({source_image_height})")
            source_image = resize(imageio.imread(im), (256, 256), anti_aliasing=True)[..., :3]
            imageio.imsave(f"{source_output_dir}/source_resized.jpg", img_as_ubyte(source_image))
            source_image_data[source_image_data_im] = source_image
            source_image_data[source_image_data_name] = base_name
            source_image_data[source_image_data_path] = im
            if (source_img_configs[src_key].use_best_frame):
                # Find best frame for this source image
                kp_source = fa.get_landmarks_from_image(255 * source_image)[0]
                kp_source_normalized = normalize_kp_facealignment(kp_source)
                frame_num = 0
                fbf_idx = -1
                min_diff = float('inf')
                for kp_driving in drv_info.driving_kps:
                    fbf_idx = fbf_idx + 1
                    if (kp_driving is None):
                        continue
                    kp_driving_normalized = normalize_kp_facealignment(kp_driving)
                    diff = (np.abs(kp_source_normalized - kp_driving_normalized) ** 2).sum()
                    if (diff < min_diff):
                        min_diff = diff
                        frame_num = fbf_idx
                if (frame_num == -1):
                    frame_num = 0
                print(f"    Best frame: {frame_num}")
                print(f"    Normalized diff: {min_diff}")
                print(f"    Calculating best frame keypoints...")
                source_image_data[source_image_data_best_frame_index] = frame_num
                source_image_data[source_image_data_diff] = min_diff
                dv_idx = 0
                sys.stdout.write("\r        {:.1f}%   {} / {}   ".format(dv_idx / frame_num * 100.0, dv_idx, frame_num))
                for im in reader:
                    frame_key = None
                    filename = None
                    if (dv_idx == 0 and source_img_configs[src_key].use_first_frame):
                        frame_key = source_image_data_first_frame
                        filename = f"{source_output_dir}/first_frame.jpg"
                    if (dv_idx == frame_num and source_img_configs[src_key].use_best_frame):
                        frame_key = source_image_data_best_frame
                        filename = f"{source_output_dir}/best_frame.jpg"
                    if (not(filename is None)):
                        if (not(dv_idx in kp_driving_cache)):
                            driving_video_frame = resize(im, (256, 256), anti_aliasing=True)[..., :3]
                            driving = torch.tensor(np.array([driving_video_frame])[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
                            kp_driving_cache[dv_idx] = kp_detector(driving[:, :, 0])
                        source_image_data[frame_key] = kp_driving_cache[dv_idx]
                        imageio.imsave(filename, im)
                    if (dv_idx == frame_num):
                        break
                    dv_idx = dv_idx + 1
                    sys.stdout.write("\r        {:.1f}%   {} / {}             ".format(dv_idx / frame_num * 100.0, dv_idx, frame_num))
                sys.stdout.write("\r        100%                      \n")
            else:
                for im in reader:
                    driving_video_frame = resize(im, (256, 256), anti_aliasing=True)[..., :3]
                    driving = torch.tensor(np.array([driving_video_frame])[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
                    kp_driving_initial = kp_detector(driving[:, :, 0])
                    source_image_data[source_image_data_first_frame] = kp_driving_initial
                    source_image_data[source_image_data_best_frame] = kp_driving_initial
                    imageio.imsave(f"{source_output_dir}/first_frame.jpg", im)
                    break
            source_image_data[source_image_data_intermediate_files] = []
            source_image_data[source_image_data_out_files] = []
            if (source_img_configs[src_key].use_best_frame is True):
                source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/orig-intermediate.mp4")
                source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}orig_bestframe.mp4")
                source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/rel-intermediate.mp4")
                source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}rel_bestframe.mp4")
                source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/reladapt-intermediate.mp4")
                source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}reladapt_bestframe.mp4")
                if (combined is True):
                    source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/combine-intermediate.mp4")
                    source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}combined_bestframe.mp4")
            if (source_img_configs[src_key].use_first_frame is True):
                source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/orig-intermediate-first.mp4")
                source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}origfirst.mp4")
                source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/rel-intermediate-first.mp4")
                source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}rel-first.mp4")
                source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/reladapt-intermediate-first.mp4")
                source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}reladapt-first.mp4")
                if (combined is True):
                    source_image_data[source_image_data_intermediate_files].append(f"{source_intermediates_dir}/combine-intermediate-first.mp4")
                    source_image_data[source_image_data_out_files].append(f"{source_output_dir}/{prefix}combined-first.mp4")
            source_image_data[source_image_data_intermediate_dir] = f"{source_intermediates_dir}"
            source_image_data[source_image_data_config] = source_img_configs[src_key]
            all_source_images_data.append(source_image_data)
            # Clean up old intermediate files
            if (os.path.exists(f"{intermediates_dir}/{driving_video_basename}_audio.mp3")):
                os.remove(f"{intermediates_dir}/{driving_video_basename}_audio.mp3")
            for intermediateFile in source_image_data[source_image_data_intermediate_files]:
                if (os.path.exists(intermediateFile)):
                    os.remove(intermediateFile)
            current_intermediate_writers = {}
            if (source_img_configs[src_key].use_best_frame is True):
                current_intermediate_writers[orig_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/orig-intermediate.mp4", fps=fps)
                current_intermediate_writers[rel_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/rel-intermediate.mp4", fps=fps)
                current_intermediate_writers[reladapt_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/reladapt-intermediate.mp4", fps=fps)
            if (source_img_configs[src_key].use_first_frame is True):
                current_intermediate_writers[orig_first_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/orig-intermediate-first.mp4", fps=fps)
                current_intermediate_writers[rel_first_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/rel-intermediate-first.mp4", fps=fps)
                current_intermediate_writers[reladapt_first_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/reladapt-intermediate-first.mp4", fps=fps)
            if (combined is True):
                if (source_img_configs[src_key].use_best_frame is True):
                    current_intermediate_writers[combine_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/combine-intermediate.mp4", fps=fps)
                if (source_img_configs[src_key].use_first_frame is True):
                    current_intermediate_writers[combine_first_filename_key] = imageio.get_writer(f"{source_intermediates_dir}/combine-intermediate-first.mp4", fps=fps)
            source_image_data[source_image_data_writers] = current_intermediate_writers
            source_img_configs_idx = source_img_configs_idx+1
        try:
            # Process each frame in driving video and write to intermediate video
            print("    Creating animations...")
            if (preview):
                driving_video_len = preview_length_seconds * drv_info.fps
            else:
                driving_video_len = drv_info.len
            processed_frames = 0.0
            sys.stdout.write("            {:.1f}%   {} / {}   ".format(processed_frames / driving_video_len * 100.0, processed_frames, driving_video_len))
            start = time.perf_counter()
            with torch.no_grad():
                for im in reader:
                    driving_video_frame = resize(im, (256, 256), anti_aliasing=True)[..., :3]
                    driving = torch.tensor(np.array([driving_video_frame])[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
                    driving_frame = driving[:, :, 0].cuda()
                    kp_driving = kp_detector(driving_frame)
                    # Iterate through source images and generate next prediction
                    for idx, source_image_data in enumerate(all_source_images_data):
                        current_intermediate_writers = source_image_data[source_image_data_writers]
                        source_image = source_image_data[source_image_data_im]
                        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                        source = source.cuda()
                        kp_source = kp_detector(source)
                        combined_frames = [source_image]
                        combined_first_frames = [source_image]
                        if (add_untransformed is True):
                            if (source_image_data[source_image_data_config].use_best_frame is True):
                                # Rel=False, Adapt=False, Best Frame
                                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                                kp_driving_initial=source_image_data[source_image_data_best_frame], use_relative_movement=True,
                                                use_relative_jacobian=True, adapt_movement_scale=True)
                                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                                prediction_000 = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                                current_intermediate_writers[orig_filename_key].append_data(img_as_ubyte(prediction_000))
                                combined_frames.append(prediction_000)

                            if (source_image_data[source_image_data_config].use_first_frame is True):
                                # Rel=False, Adapt=False, First Frame
                                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                                kp_driving_initial=source_image_data[source_image_data_first_frame], use_relative_movement=True,
                                                use_relative_jacobian=True, adapt_movement_scale=True)
                                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                                prediction_000 = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                                current_intermediate_writers[orig_first_filename_key].append_data(img_as_ubyte(prediction_000))
                                combined_first_frames.append(prediction_000)
                        if (add_relative is True):
                            if (source_image_data[source_image_data_config].use_best_frame is True):
                                # Rel=True, Adapt=False, Best Frame
                                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                                kp_driving_initial=source_image_data[source_image_data_best_frame], use_relative_movement=True,
                                                use_relative_jacobian=True, adapt_movement_scale=False)
                                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                                prediction_100 = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                                current_intermediate_writers[rel_filename_key].append_data(img_as_ubyte(prediction_100))
                                combined_frames.append(prediction_100)

                            if (source_image_data[source_image_data_config].use_first_frame is True):
                                # Rel=True, Adapt=False, First Frame
                                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                                kp_driving_initial=source_image_data[source_image_data_first_frame], use_relative_movement=True,
                                                use_relative_jacobian=True, adapt_movement_scale=False)
                                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                                prediction_100 = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                                current_intermediate_writers[rel_first_filename_key].append_data(img_as_ubyte(prediction_100))
                                combined_first_frames.append(prediction_100)
                        if (add_relative_and_adapted is True):
                            if (source_image_data[source_image_data_config].use_best_frame is True):
                                # Rel=True, Adapt=True, Best Frame
                                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                                kp_driving_initial=source_image_data[source_image_data_best_frame], use_relative_movement=True,
                                                use_relative_jacobian=True, adapt_movement_scale=True)
                                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                                prediction_110 = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                                current_intermediate_writers[reladapt_filename_key].append_data(img_as_ubyte(prediction_110))
                                combined_frames.append(prediction_110)

                            if (source_image_data[source_image_data_config].use_first_frame is True):
                                # Rel=True, Adapt=True, Best Frame
                                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                                kp_driving_initial=source_image_data[source_image_data_first_frame], use_relative_movement=True,
                                                use_relative_jacobian=True, adapt_movement_scale=True)
                                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                                prediction_110 = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                                current_intermediate_writers[reladapt_first_filename_key].append_data(img_as_ubyte(prediction_110))
                                combined_first_frames.append(prediction_110)
                        # Combine all frames
                        if (combined is True):
                            if (source_image_data[source_image_data_config].use_best_frame is True):
                                combined_frames.append(driving_video_frame)
                                combined_image = np.concatenate(combined_frames, axis=1)
                                current_intermediate_writers[combine_filename_key].append_data(img_as_ubyte(combined_image))

                            if (source_image_data[source_image_data_config].use_first_frame is True):
                                combined_first_frames.append(driving_video_frame)
                                combined_image = np.concatenate(combined_first_frames, axis=1)
                                current_intermediate_writers[combine_first_filename_key].append_data(img_as_ubyte(combined_image))
                    processed_frames = processed_frames + 1
                    now = time.perf_counter()
                    frames_per_second = processed_frames / (now-start)
                    frames_left = driving_video_len - processed_frames
                    sys.stdout.write(f"\r        {processed_frames / driving_video_len * 100.0:.1f}%   {processed_frames} / {driving_video_len}   ETA = {frames_left / frames_per_second / 60:.2f}mins ({frames_per_second:.2f} F/s)")
                    if (preview is True and (processed_frames / fps) >= preview_length_seconds):
                        break
                
            sys.stdout.write("\r            100.0%   \n")
            # Close all intermediate writers
            for source_image_data in all_source_images_data:
                for file_key in source_image_data[source_image_data_writers]:
                    source_image_data[source_image_data_writers][file_key].close()

            if (preview is True):
                # Extract audio from driving video
                print("    Extracting audio from driving video...")
                subprocess.run(["ffmpeg", "-y", "-i", drv_info.path, "-vn", "-ar", "44100", "-f", "mp3", f"{intermediates_dir}/{driving_video_basename}_audio.mp3"], check=True, stdout=subprocess.DEVNULL)

                # Combine it with intermediate video
                print("    Finalizing all outputs...")
                for source_image_data in all_source_images_data:
                    for idx, intermediateFile in enumerate(source_image_data[source_image_data_intermediate_files]):
                        if (os.path.exists(intermediateFile)):
                            subprocess.run(["ffmpeg", "-y", "-i", f"{intermediates_dir}/{driving_video_basename}_audio.mp3", "-i", intermediateFile, "-af", f"atrim=0:{preview_length_seconds}", source_image_data[source_image_data_out_files][idx]], check=True, stdout=subprocess.DEVNULL)
            else:
                # Extract audio from driving video
                print("    Extracting audio from driving video...")
                subprocess.run(["ffmpeg", "-y", "-i", drv_info.path, "-vn", "-ar", "44100", "-f", "mp3", f"{intermediates_dir}/{driving_video_basename}_audio.mp3"], check=True, stdout=subprocess.DEVNULL)

                # Combine it with intermediate video
                print("    Finalizing all outputs...")
                for source_image_data in all_source_images_data:
                    for idx, intermediateFile in enumerate(source_image_data[source_image_data_intermediate_files]):
                        if (os.path.exists(intermediateFile)):
                            subprocess.run(["ffmpeg", "-y", "-i", f"{intermediates_dir}/{driving_video_basename}_audio.mp3", "-i", intermediateFile, source_image_data[source_image_data_out_files][idx]], check=True, stdout=subprocess.DEVNULL)
            if (preview is True):
                result_diffs = []
                print(f"Printing normalized diff from best frames...")
                sorted = sorted(all_source_images_data, key=lambda d:d[source_image_data_diff])
                for data in sorted:
                    classification = "Bad"
                    lowest_threshold = None
                    for c in classifications["classifications"]:
                        if (data[source_image_data_diff] <= c["threshold"] and (lowest_threshold is None or lowest_threshold > c["threshold"])):
                            classification = c["name"]
                            lowest_threshold = c["threshold"]
                    print(f"     {data[source_image_data_name]}: {data[source_image_data_diff]} ({classification})")
                generating_yaml_name = f'recommended_configs\{driving_video_basename}.yaml'
                if (not(os.path.exists("recommended_configs"))):
                    os.makedirs("recommended_configs")
                if (os.path.exists(generating_yaml_name)):
                    print(f"Cleaning up old generating yaml...")
                    os.remove(generating_yaml_name)
                print(f"Generating yaml can be found at: {generating_yaml_name}")
                input["preview"] = False
                input["source_dirs"] = None
                input["combined"] = False
                input["add_relative"] = False
                input["add_relative_and_adapted"] = True
                input["add_untransformed"] = False
                input["use_best_frame"] = True
                input["use_first_frame"] = False
                input["source_hints"] = [{"path": sorted[0][source_image_data_path]}]
                with open(generating_yaml_name, 'w') as file:
                    yaml.dump(input, file)
        finally:
            for source_image_data in all_source_images_data:
                for writer in source_image_data[source_image_data_writers]:
                    source_image_data[source_image_data_writers][writer].close()
            reader.close()
    print("Done!")
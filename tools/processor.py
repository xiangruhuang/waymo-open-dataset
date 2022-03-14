import argparse
import os, pickle
import numpy as np
from multiprocessing import Pool
import glob
import zlib
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import box_utils

def convert_range_image_to_point_cloud_labels(frame,
        range_images,
        segmentation_labels,
        ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
      range_image = range_images[c.name][ri_index]
      range_image_tensor = tf.reshape(
          tf.convert_to_tensor(range_image.data), range_image.shape.dims)
      range_image_mask = range_image_tensor[..., 0] > 0

      if c.name in segmentation_labels:
        assert c.name == dataset_pb2.LaserName.TOP
        sl = segmentation_labels[c.name][ri_index]
        sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
        sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        point_labels.append(sl_points_tensor.numpy())

    return point_labels

def decode_frame(frame, frame_id):
    """Extract points, segmentation labels and objects from frame.

    Args:
        frame: frame proto buffer
        frame_id: the index of this frame in the sequence

    Returns:
        lidar3d: lidar points
        seg3d: segmentation labels
        det3d: detection labels
    
    """
    (range_images, camera_projections, segmentation_labels,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)
        
    #Load points in vehicle frames. Sort by lidar cameras so that 
    # points from lidar.TOP goes first and aligns with segmentation labels
    # (Segmentation labels is occurs twice per second and is annotated only for 
    # lidar.TOP)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose)
    points_ri2, cp_points_ri2 = \
            frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections,
                range_image_top_pose, ri_index=1)
    points = [np.concatenate([p1, p2], axis=0) \
                for p1, p2 in zip(points, points_ri2)]
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    points = np.concatenate([points_all, points_all_ri2], axis=0)

    # load segmentation labels
    if frame.lasers[0].ri_return1.segmentation_label_compressed:
        assert frame.lasers[0].ri_return2.segmentation_label_compressed
        point_labels = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels)
        point_labels_ri2 = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels, ri_index=1)
        point_labels_all = np.concatenate(point_labels, axis=0)
        point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
        point_labels = np.concatenate([point_labels_all, point_labels_all_ri2],
                                      axis=0)
    else:
        point_labels = None

    # load objects
    objects = []
    for label in frame.laser_labels:
        box = box_utils.box_to_tensor(label.box).numpy()
        speed = np.array([label.metadata.speed_x, label.metadata.speed_y])
        accel = np.array([label.metadata.accel_x, label.metadata.accel_y])
        obj_class = label.type
        uid = label.id
        num_points = label.num_lidar_points_in_box
        
        obj = dict(
            box=box,
            speed=speed,
            accel=accel,
            obj_class=obj_class,
            uid=uid,
            num_points=num_points
        )
        objects.append(obj)

    frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
        scene_name=frame.context.name,
        location=frame.context.stats.location,
        time_of_day=frame.context.stats.time_of_day,
        timestamp=frame.timestamp_micros)

    lidar3d = dict(
        scene_name=frame.context.name,
        frame_name=frame_name,
        frame_id=frame_id,
        points=points,
    )

    if point_labels is None:
        seg3d = None
    else:
        seg3d = dict(
            scene_name=frame.context.name,
            frame_name=frame_name,
            frame_id=frame_id,
            point_labels=point_labels,
        )

    det3d = dict(
        scene_name=frame.context.name,
        frame_name=frame_name,
        frame_id=frame_id,
        objects=objects,
    )

    return lidar3d, seg3d, det3d


def convert_sequence(filename, idx, args):
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for frame_id, data in enumerate(dataset):

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        lidar3d, seg3d, det3d = decode_frame(frame, frame_id)

        for prefix, data in zip(
            ['lidar', 'seg', 'obj'], [lidar3d, seg3d, det3d]
        ):
            folder = os.path.join(
                         args.output_path,
                         prefix
                     )
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(
                       args.output_path,
                       prefix,
                       'seq_{}_frame_{}.pkl'.format(idx, frame_id)
                   )
            if data is not None:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
                    Run this script to convert .tfrecord file into
                        three type of .pkl files, containing
                        
                        1) lidar point clouds,
                        2) 3D point-wise segmentation labels,
                        3) objects (boxes and classes)

                    example:
                        python tools/processor.py <input_path> <output_path>
                    """,
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument('tfrecord_path', help='path to tfrecord files',
                        type=str)
    parser.add_argument('output_path', help='path to store converted files',
                        type=str)
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--chunksize', type=int, default=10)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print(args)
    
    tfrecord_files = sorted(glob.glob(f'{args.tfrecord_path}/*.tfrecord'))
    convert_sequence(tfrecord_files[0], 0, args)
    num_sequences = len(tfrecord_files)
    with Pool(args.num_processes) as pool: # change according to your cpu
        r = pool.map(
                lambda idx: convert_sequence(tfrecord_files[idx], idx, args),
                range(num_sequences),
                chunk_size=args.chunksize,
            )

if __name__ == '__main__':
    main()

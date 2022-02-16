from pathlib import Path

import numpy as np

# ========================================= #
# Set Path here
# EXAMPLE:
#   ROOT = '/path/to/HO3D_v2'

ROOT = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/wack/code/Users/fho3d"
# ========================================= #


if ROOT is None:
    raise ValueError("Set the ROOT var to the path to the Hand-Object 3D Pose Annotation datset"
                     "Data set can be found here: https://www.tugraz.at/index.php?id=40231 \n"
                     "NOTE:  This application uses the V2 data.")

# Change this path
evaluation_root_path = Path(ROOT, 'evaluation')
train_root_path = Path(ROOT, 'train')

# Load object mesh
reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

images_train = []
points2d_train = []
points3d_train = []

images_val = []
points2d_val = []
points3d_val = []

# Train
for subject in sorted(train_root_path.iterdir()):
    rgb = Path(subject, 'rgb')
    meta = Path(subject, 'meta')
    for rgb_file in sorted(rgb.iterdir()):

        meta_file = Path(meta, f'{rgb_file.stem}.pkl')
        data = np.load(meta_file, allow_pickle=True)

        cam_intr = data['camMat']
        hand3d = data['handJoints3D'][reorder_idx]

        obj_corners = data['objCorners3D']
        hand_object3d = np.concatenate([hand3d, obj_corners])
        hand_object3d = hand_object3d.dot(coordChangeMat.T)
        hand_object_proj = cam_intr.dot(hand_object3d.transpose()).transpose()
        hand_object2d = (hand_object_proj / hand_object_proj[:, 2:])[:, :2]

        if subject == 'MC6':
            images_val.append(rgb_file)
            points3d_val.append(hand_object3d)
            points2d_val.append(hand_object2d)
        else:
            images_train.append(rgb_file)
            points3d_train.append(hand_object3d)
            points2d_train.append(hand_object2d)

images_train = np.array(images_train)
points2d_train = np.array(points2d_train)
points3d_train = np.array(points3d_train)

images_val = np.array(images_val)
points2d_val = np.array(points2d_val)
points3d_val = np.array(points3d_val)

np.save('images-val.npy', images_val)
np.save('points2d-val.npy', points2d_val)
np.save('points3d-val.npy', points3d_val)

np.save('images-train.npy', images_train)
np.save('points2d-train.npy', points2d_train)
np.save('points3d-train.npy', points3d_train)

images_test = []
points2d_test = []
points3d_test = []

# Evaluation
for subject in sorted(evaluation_root_path.iterdir()):
    rgb = Path(subject, 'rgb')
    meta = Path(subject, 'meta')
    for rgb_file in sorted(rgb.iterdir()):
        meta_file = Path(meta, f'{rgb_file.stem}.pkl')
        data = np.load(meta_file, allow_pickle=True)

        cam_intr = data['camMat']
        hand3d = np.repeat(np.expand_dims(data['handJoints3D'], 0), 21, 0)
        hand3d = hand3d[reorder_idx]
        obj_corners = data['objCorners3D']
        hand_object3d = np.concatenate([hand3d, obj_corners])
        hand_object3d = hand_object3d.dot(coordChangeMat.T)
        hand_object_proj = cam_intr.dot(hand_object3d.transpose()).transpose()
        hand_object2d = (hand_object_proj / hand_object_proj[:, 2:])[:, :2]

        images_test.append(rgb_file)
        points3d_test.append(hand_object3d)
        points2d_test.append(hand_object2d)

images_test = np.array(images_test)
points2d_test = np.array(points2d_test)
points3d_test = np.array(points3d_test)

np.save('images-test.npy', images_test)
np.save('points2d-test.npy', points2d_test)
np.save('points3d-test.npy', points3d_test)

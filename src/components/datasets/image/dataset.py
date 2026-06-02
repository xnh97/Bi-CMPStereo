import os
from PIL import Image
import numpy as np
import yaml
import cv2
import torch.utils.data

class ImageDataset(torch.utils.data.Dataset):
    _PATH_DICT={
        'timestamp': 'timestamps.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = 0.0

    def __init__(self, root, location, disp_timestamps, root_calib=None):
        assert location in self._LOCATION
        self.location = location  # 对左/右目提取图像
        self.root = root
        
        self.warp_image_path = os.path.join(root, 'warp_' + self._PATH_DICT[self.location])
        os.makedirs(self.warp_image_path, exist_ok=True)

        self.warp_image_path_list = get_path_list(self.warp_image_path)
        if len(self.warp_image_path_list) < len(disp_timestamps):
            self.generate_warp_images(disp_timestamps, root_calib)
        
        self.warp_image_path_list = get_path_list(self.warp_image_path)
        assert len(self.warp_image_path_list) == len(disp_timestamps)

        self.timestamp_to_warp_image_path = {
            timestamp: filepath for timestamp, filepath in
                            zip(disp_timestamps, self.warp_image_path_list)
            }
        
    def generate_warp_images(self, disp_timestamps, root_calib):
        assert root_calib is not None
        image_timestamps = load_timestamp(os.path.join(self.root, self._PATH_DICT['timestamp']))
        
        image_path_list = get_path_list(os.path.join(self.root, self._PATH_DICT[self.location], 'rectified'))
        timestamp_to_image_path = {
                timestamp: filepath for timestamp, filepath in
                zip(image_timestamps, image_path_list)
            }
        print()
        timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(timestamp_to_image_path[timestamp]))[0])
            for timestamp in timestamp_to_image_path.keys()
        }

        self.homography = {}
        with open(os.path.join(root_calib, 'cam_to_cam.yaml')) as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)
        
        cam0_int = self.conf['intrinsics']['camRect0']['camera_matrix']
        Kr0 = np.array([[cam0_int[0], 0, cam0_int[2]], 
                        [0, cam0_int[1], cam0_int[3]], 
                        [0, 0, 1]])
        cam1_int = self.conf['intrinsics']['camRect1']['camera_matrix']
        Kr1 = np.array([[cam1_int[0], 0, cam1_int[2]], 
                        [0, cam1_int[1], cam1_int[3]], 
                        [0, 0, 1]])
        cam2_int = self.conf['intrinsics']['camRect2']['camera_matrix']
        Kr2 = np.array([[cam2_int[0], 0, cam2_int[2]], 
                        [0, cam2_int[1], cam2_int[3]], 
                        [0, 0, 1]])
        cam3_int = self.conf['intrinsics']['camRect3']['camera_matrix']
        Kr3 = np.array([[cam3_int[0], 0, cam3_int[2]], 
                        [0, cam3_int[1], cam3_int[3]], 
                        [0, 0, 1]])

        T32 = np.array(self.conf['extrinsics']['T_32'])
        T10 = np.array(self.conf['extrinsics']['T_10'])

        R_rect0 = np.array(self.conf['extrinsics']['R_rect0'])
        R_rect1 = np.array(self.conf['extrinsics']['R_rect1'])
        R_rect2 = np.array(self.conf['extrinsics']['R_rect2'])
        R_rect3 = np.array(self.conf['extrinsics']['R_rect3'])

        M1=np.matmul(Kr1,R_rect1)
        M2=np.matmul(M1,T10[:3,:3])
        M3=np.matmul(M2,np.linalg.inv(R_rect0))
        self.homography['left']=np.matmul(M3,np.linalg.inv(Kr0))

        M1=np.matmul(Kr3,R_rect3)
        M2=np.matmul(M1,T32[:3,:3])
        M3=np.matmul(M2,np.linalg.inv(R_rect2))
        self.homography['right']=np.matmul(M3,np.linalg.inv(Kr2))

        for i in disp_timestamps:
            image = load_image(timestamp_to_image_path[i])
            name = timestamp_to_image_path[i].split("/")[-1]

            if self.location == 'left':
                warp_image=cv2.warpPerspective(image[:,:,[2,1,0]], self.homography['left'], (1440, 1080),  flags=cv2.WARP_INVERSE_MAP)
            elif self.location == 'right':
                warp_image=cv2.warpPerspective(image[:,:,[2,1,0]], self.homography['right'], (1440, 1080))
            warp_image=warp_image[0:480, 0:640,:]
            cv2.imwrite(os.path.join(self.warp_image_path, name), warp_image)


    def __len__(self):
        return 0
    

    def __getitem__(self, timestamp):
        warp_image = load_image(self.timestamp_to_warp_image_path[timestamp])[:,:,[2,1,0]].astype("uint8")
        warp_image = 2 * (warp_image / 255.0) - 1.0
        return warp_image
    
    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        return batch

def load_timestamp(root):
    return np.loadtxt(root, dtype='int64')

def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]

def load_image(root):
    image = np.array(Image.open(root)).astype(np.uint8)
    return image
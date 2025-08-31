import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math
from tqdm import tqdm

import os
import sys
from dotenv import load_dotenv
load_dotenv()
shapenet_path = os.getenv("SHAPENET_DATASET_PATH")
dino_path = os.getenv("DINO_PROJECT_PATH")

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def rotation_z(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                [0.0, 1.0, 0.0],
                                [sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T

class ViPCDataLoader(Dataset):
    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884',
            'cabinet':'02933112',
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459',
            'firearm': '04090263',
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088',
            'watercraft':'04530566'
        }
        with open(filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
       
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        self.depth_path = os.path.join(data_path,'ShapeNetViPC-depth')



        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):
       
        key = self.key[idx]
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        depth_path = os.path.join(self.depth_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')

        #Inserted to correct a bug in the splitting for some lines
        if(len(ran_key.split('/')[-1])>3):
            print("bug")
            print(ran_key.split('/')[-1])
            fin = ran_key.split('/')[-1][-2:]
            interm = ran_key.split('/')[-1][:-2]
           
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')
            depth_path = os.path.join(self.depth_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')

        views = self.transform(Image.open(view_path))
        views = views[:3,:,:]

        depth = self.transform(Image.open(depth_path))

        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        # load the view metadata
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
       
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0)
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float(),depth.float()

    def __len__(self):
        return len(self.key)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    category = "table"
    ViPCDataset = ViPCDataLoader('train_list.txt',data_path=shapenet_path,status='test', category = category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    for image, gt, partial in tqdm(train_loader):
       
        print(image.shape)
       
        pass
   

# import os
# from torchvision import transforms
# import os.path
# from torch.utils.data import Dataset
# import torch
# from PIL import Image
# import numpy as np
# import pickle
# import random
# import math
# from tqdm import tqdm

# import os
# import sys
# from dotenv import load_dotenv
# # It's good practice to check if the .env file was loaded successfully
# # and handle cases where paths might be None.
# load_dotenv()
# shapenet_path = os.getenv("SHAPENET_DATASET_PATH")
# dino_path = os.getenv("DINO_PROJECT_PATH")

# if not shapenet_path or not dino_path:
#     print("Error: SHAPENET_DATASET_PATH or DINO_PROJECT_PATH not found in environment.")
#     print("Please check your .env file.")
#     sys.exit(1)


# def collate_fn(batch):
#     # This collate function is important as it filters out None items
#     # that can result from file-not-found errors or malformed keys.
#     batch = list(filter(lambda x: x is not None, batch))
#     if len(batch) == 0:
#         return None
#     return torch.utils.data.dataloader.default_collate(batch)


# def rotation_z(pts, theta):
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
#                                 [sin_theta, cos_theta, 0.0],
#                                 [0.0, 0.0, 1.0]])
#     return pts @ rotation_matrix.T


# def rotation_y(pts, theta):
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
#                                 [0.0, 1.0, 0.0],
#                                 [sin_theta, 0.0, cos_theta]])
#     return pts @ rotation_matrix.T


# def rotation_x(pts, theta):
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     rotation_matrix = np.array([[1.0, 0.0, 0.0],
#                                 [0.0, cos_theta, -sin_theta],
#                                 [0.0, sin_theta, cos_theta]])
#     return pts @ rotation_matrix.T

# class ViPCDataLoader(Dataset):
#     # --- CHANGED ---
#     # Removed the `view_align` parameter as it's no longer needed.
#     # The dataloader now strictly uses the items from the list file.
#     def __init__(self,filepath,data_path,status,pc_input_num=3500, category='all'):
#         super(ViPCDataLoader,self).__init__()
#         self.pc_input_num = pc_input_num
#         self.status = status
#         self.key = []
#         self.category = category
#         self.cat_map = {
#             'plane':'02691156', 'bench': '02828884', 'cabinet':'02933112',
#             'car':'02958343', 'chair':'03001627', 'monitor': '03211117',
#             'lamp':'03636649', 'speaker': '03691459', 'firearm': '04090263',
#             'couch':'04256520', 'table':'04379243', 'cellphone': '04401088',
#             'watercraft':'04530566'
#         }
       
#         # Read the list of data items
#         try:
#             with open(filepath,'r') as f:
#                 filelist = f.readlines()
#         except FileNotFoundError:
#             print(f"Error: The list file was not found at '{filepath}'")
#             filelist = []

#         self.imcomplete_path = os.path.join(data_path,'ShapeNet-ViPC-Partial')
#         self.gt_path = os.path.join(data_path,'ShapeNet-ViPC-GT')
#         self.rendering_path = os.path.join(data_path,'ShapeNet-ViPC-View')
#         self.depth_path = os.path.join(data_path,'ShapeNet-ViPC-depth')
#         # Filter the list based on the selected category
#         for key in filelist:
#             key = key.strip()
#             if not key:
#                 continue
           
#             current_cat_id = key.split('/')[0]
#             if category != 'all':
#                 if current_cat_id != self.cat_map.get(category):
#                     continue
#             self.key.append(key)

#         self.transform = transforms.Compose([
#             transforms.Resize(224),
#             transforms.ToTensor()
#         ])

#         print(f'{status} data num: {len(self.key)}')

#     # --- CHANGED ---
#     # This method has been rewritten to deterministically load items from the list
#     # and to be more robust against errors like missing files.
#     def __getitem__(self, idx):
       
#         key = self.key[idx]
#         parts = key.split('/')
#         if len(parts) != 3:
#             print(f"Warning: Malformed key '{key}' in filelist. Skipping.")
#             return None # This will be handled by our collate_fn

#         category_id, object_id, view_id = parts
       
#         # Construct all paths deterministically from the key
#         pc_part_path = os.path.join(self.imcomplete_path, category_id, object_id, view_id + '.dat')
#         pc_path = os.path.join(self.gt_path, category_id, object_id, view_id + '.dat')
#         view_path = os.path.join(self.rendering_path, category_id, object_id, 'rendering', view_id + '.png')
#         depth_path = os.path.join(self.depth_path, category_id, object_id, 'rendering', view_id + '.png')
#         # Use try-except to gracefully handle missing files
#         try:
#             views = self.transform(Image.open(view_path))
#             views = views[:3,:,:]
           
#             depth = self.transform(Image.open(depth_path))
#             with open(pc_path, 'rb') as f:
#                 pc = pickle.load(f).astype(np.float32)
           
#             with open(pc_part_path, 'rb') as f:
#                 pc_part = pickle.load(f).astype(np.float32)

#             # --- More robust metadata path construction ---
#             metadata_path = os.path.join(os.path.dirname(view_path), 'rendering_metadata.txt')
#             view_metadata = np.loadtxt(metadata_path)

#         except FileNotFoundError as e:
#             print(f"Warning: A file was not found for key '{key}'. Skipping. Details: {e}")
#             return None
#         except Exception as e:
#             print(f"An error occurred while loading data for key '{key}'. Skipping. Details: {e}")
#             return None

#         # In case some item point number is less than the required input number
#         if pc_part.shape[0] < self.pc_input_num:
#             pc_part = np.repeat(pc_part, (self.pc_input_num // pc_part.shape[0]) + 1, axis=0)[0:self.pc_input_num]

#         # Since the partial, GT, and image view are now always the same,
#         # the IDs are identical.
#         image_view_id = view_id
#         part_view_id = view_id
       
#         theta_part = math.radians(view_metadata[int(part_view_id), 0])
#         phi_part = math.radians(view_metadata[int(part_view_id), 1])

#         theta_img = math.radians(view_metadata[int(image_view_id), 0])
#         phi_img = math.radians(view_metadata[int(image_view_id), 1])

#         # This logic aligns the partial point cloud to its own canonical view
#         pc_part = rotation_y(rotation_x(pc_part, -phi_part), np.pi + theta_part)
#         pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

#         # Normalize partial point cloud and GT to the same scale
#         gt_mean = pc.mean(axis=0)
#         pc = pc - gt_mean
#         pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
#         pc = pc / pc_L_max

#         pc_part = pc_part - gt_mean
#         pc_part = pc_part / pc_L_max

#         return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float(), depth.float()

#     def __len__(self):
#         return len(self.key)


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     category = "table"
#     # Note: `view_align` argument is no longer needed
#     ViPCDataset = ViPCDataLoader('train_list.txt', data_path=shapenet_path, status='test', category=category)
   
#     # Using the custom collate_fn is important to filter out None values from skipped items
#     train_loader = DataLoader(ViPCDataset,
#                               batch_size=4,
#                               num_workers=2,
#                               shuffle=True,
#                               drop_last=True,
#                               collate_fn=collate_fn)
   
#     for batch_data in tqdm(train_loader):
#         # If the batch is None (all items in it had errors), skip it.
#         if batch_data is None:
#             continue
       
#         image, gt, partial = batch_data
#         print(f"Batch loaded successfully. Image shape: {image.shape}")
#         pass
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TestDataset(Dataset):
	def __init__(self):
		#self.imgs_path 
		directory = "/home-local/ConnectomeML/Data_Test/"
		
		subj_list = glob.glob(directory + "*")
		print(subj_list)
		self.data = []
		for subject_path in subj_list:
			subject = subject_path.split("/")[-1]
			img_path = glob.glob(directory + "/" + subject +"/*CONNECTOME_NUMSTREAM.csv")
			label_path = glob.glob(directory + "/" + subject + "/*MODULARITY.csv")
			site_path = glob.glob(directory + "/" + subject + "/*SITE.csv")
			#contrastivegroup_path = glob.glob(directory + "/" + subject + "/*CONTRASTIVE*.csv")
			
			self.data.append([img_path, label_path, site_path])
			print(subject)
			#print(img_path)
			#print(label_path)
			#print(site_path)
			#print(contrastivegroup_path)
			
		self.img_dim = (84, 84)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, label_path, site_path = self.data[idx]
		#print("Lengths:", len(label_path), len(img_path), len(site_path))
		#img = cv2.imread(img_path)
		#img = cv2.resize(img, self.img_dim)
		#print("Printing image path, ", type(img_path[0]))
		img = pd.read_csv(img_path[0], sep=',', header=None)
		img = img.to_numpy()
		img = img/10000
		img = np.float32(img)
		#print("Printing shape of connectome: ",np.shape(img))
		#print("Printing type of data:", type(img[1,1]))

		label = pd.read_csv(label_path[0], sep=',', header=None)
		label = label.to_numpy()
		label = np.float32(label)
		
		site = pd.read_csv(site_path[0], sep=',', header=None)
		site = site.to_numpy()
		site = np.float32(site)
		


		img_tensor = torch.from_numpy(img)
		label_tensor = torch.from_numpy(label)
		#img_tensor = img_tensor.permute(2, 0, 1)
		site_tensor = torch.tensor(site)
		#group_tensor = torch.tensor(group)
		return img_tensor, label_tensor, site_tensor, img_path

if __name__ == "__main__":
	dataset = CustomDataset()		
	data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
	for imgs, predictionlabels, sitelabels, img_paths in data_loader:
		print("Batch of images has shape: ", imgs.shape)
		print("Batch of labels has shape: ", sitelabels.shape)
		print("Batch of labels has shape: ", predictionlabels.shape)

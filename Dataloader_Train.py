import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

class CustomDataset(Dataset):
	def __init__(self):
		#self.imgs_path 
		directory = "/home-local/ConnectomeML/Data_contrastive/"
		
		subj_list = glob.glob(directory + "*")
		#print(subj_list)
		self.data = []
		for subject_path in subj_list:
			subject = subject_path.split("/")[-1]
			img_path = glob.glob(directory + "/" + subject +"/*CONNECTOME_NUMSTREAM.csv")
			#label_path = glob.glob(directory + "/" + subject + "/*CHARPATHLENGTH.csv")
			#img_path = glob.glob(directory + "/" + subject +"/*CONNECTOME_NUMSTREAM.csv")
			label_path = glob.glob(directory + "/" + subject + "/*MODULARITY.csv")
			age_path = glob.glob(directory + "/" + subject + "/*AGE.csv")
			site_path = glob.glob(directory + "/" + subject + "/*SITE.csv")
			#contrastivegroup_path = glob.glob(directory + "/" + subject + "/*CONTRASTIVE*.csv")
			
			self.data.append([img_path, label_path, site_path, age_path])
			#print(subject)
			#print(img_path)
			#print(label_path)
			#print(site_path)
			#print(contrastivegroup_path)
			
		self.img_dim = (84, 84)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, label_path, site_path, age_path = self.data[idx]
		#print("Lengths:", len(label_path), len(img_path), len(site_path))
		#img = cv2.imread(img_path)
		#img = cv2.resize(img, self.img_dim)
		#print("Printing image path, ", type(img_path[0]))
		img = pd.read_csv(img_path[0], sep=',', header=None)
		img = img.to_numpy()
		#img = img/256 
		#img = 1-img # reverse so big is small and small is big
		img = img/100000
		#img = img(img<0.95*np.max(img))
		img = np.float32(img)
		#print("Printing shape of connectome: ",np.shape(img))
		#print("Printing type of data:", type(img[1,1]))

		label = pd.read_csv(label_path[0], sep=',', header=None)
		label = label.to_numpy()
		label = np.float32(label)
		
		site = pd.read_csv(site_path[0], sep=',', header=None)
		site = site.to_numpy()
		site = np.float32(site)
		
		#group = pd.read_csv(group_path[0], sep=',', header=None)
		#group = group.to_numpy()
		#group = np.float32(group)

		age = pd.read_csv(age_path[0], sep=',', header=None)
		age = age.to_numpy()
		age = np.float32(age)


		img_tensor = torch.from_numpy(img)
		label_tensor = torch.from_numpy(label)
		#img_tensor = img_tensor.permute(2, 0, 1)
		site_tensor = torch.tensor(site)
		age_tensor = torch.tensor(age)
		return img_tensor, label_tensor, site_tensor, age_tensor, img_path
		
		#return img, label, site, group, img_path
	
def getTriplets(imgs, predictionlabels, sitelabels, contrastivegroups, img_paths):
        triplets = []
        print("Length",len(sitelabels))
        for i in np.arange(1,len(sitelabels)-1):
                print(i, imgs.shape, predictionlabels.shape, sitelabels.shape)
                paths = img_paths[0]
                #print(len(img_paths))
                anchor = [imgs[i], predictionlabels[i], sitelabels[i], paths[i]]
                # get current contrastive group
                samegroup = torch.reshape(contrastivegroups==contrastivegroups[i],(len(sitelabels),-1))
                othergroup = ~samegroup
                samegroup[i] = False # do not consider the anchor
                samegroup = torch.nonzero(samegroup)
                othergroup = torch.nonzero(othergroup)
                print("contrastive group of anchor",contrastivegroups[i])
                print(type(samegroup), samegroup.shape)
                print(type(othergroup), othergroup.shape)
                print(type(imgs), imgs.shape)
                
                
                pos_imgs = imgs[samegroup]
                pos_predictionlabels = predictionlabels[samegroup]
                pos_sitelabels = sitelabels[samegroup]
                #pos_imgpaths = img_paths[samegroup.detach()]
                
                neg_imgs = imgs[othergroup]
                neg_predictionlabels = predictionlabels[othergroup]
                neg_sitelabels = sitelabels[othergroup]
                #neg_imgpaths = img_paths[othergroups.detach()]            
                
                randposindex = random.randint(0, len(pos_sitelabels)-1)
                randnegindex = random.randint(0, len(neg_sitelabels)-1)
                
                positive = [imgs[randposindex], predictionlabels[randposindex], sitelabels[randposindex]]
                negative = [imgs[randnegindex], predictionlabels[randnegindex], sitelabels[randnegindex]]
                
                # get same bucket as this one, but random sample
                # get different bucket than this one, and random sample
                #triplets.append([anchor, positive, negative])
                anchorimgs.append(imgs[i])
                anchorpredictionlabels.append(predictionlabels[i])
                anchorsitelabels.append(sitelabels[i])
                anchorfilenames.append(paths[i])
                posimgs.append(imgs[randposindex])
                pospredictionlabels.append(predictionlabels[randposindex])
                possitelabels.append(sitelabels[randposindex])
                negimgs.append(imgs[randnegindex])
                negpredictionlabels.append(predictionlabels[randnegindex])
                negsitelabels.append(sitelabels[randnegindex])
               
                
        d = {'anchor_imgs': anchorimgs, 'anchor_predictionlabels': anchorpredictionlabels, 'anchor_sitelabels': anchorsitelabels, 'anchor_filenames': anchorfilenames, 'pos_imgs': posimgs, 'pos_predictionlabels': pospredictionlabels, 'pos_sitelabels': possitelabels, 'neg_imgs': negimgs, 'neg_predictionlabels': negpredictionlabels, 'neg_sitelabels':  negsitelabels}
        df = pd.DataFrame(data=d)
        return df  
    
if __name__ == "__main__":
		dataset = CustomDataset()		
		data_loader = DataLoader(dataset, batch_size=1500, shuffle=True)
		for imgs, predictionlabels, sitelabels, contrastivegroups, img_paths in data_loader:
				triplets = getTriplets(imgs, predictionlabels, sitelabels, contrastivegroups, img_paths)
	            
				print("print triplets element 1 len",len(triplets))
				print("print triplets element 1",(len(triplets['anchor_imgs'])))
				
				
				
				print("print triplets element anchor",(len(anchor)))
				print("print triplets element anchor",(len(positive)))
				print("print triplets element anchor",(len(negative)))
				print("Batch of images has shape: ", imgs.shape)
				print("Batch of labels has shape: ", sitelabels.shape)
				print("Batch of labels has shape: ", predictionlabels.shape)

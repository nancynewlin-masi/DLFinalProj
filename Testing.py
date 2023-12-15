import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
from torch.utils.data import Dataset, DataLoader
from Dataloader_contrastive import CustomDataset
from Dataloader_Test_modularity import TestDataset
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from Architecture import AE, Conditional_VAE
import losses
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

CONNECTOME_SIZE = 84*84
seed = 42
batch_size = 2000
test_dataset = TestDataset()
train_dataset = CustomDataset()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)




experimentname = []
predictionloss_site1 = []
predictionloss_site2 = []
age_loss = []
mutualinformation = []
rfacc = []
delta=1
for beta in [0,0.1,1,10,100]:
        for fold in [1]:
        	
                criterion_site1 = nn.MSELoss()
                criterion_site2 = nn.MSELoss()
                loss_fn = losses.BetaVAE_Loss(beta)

                #PATH ='/home-local/ConnectomeML/Models/Model_Y_fold_1_beta_1000_delta_1_epoch_400.pt'
                #PATH ='/home-local/ConnectomeML/Models/Model_Y_fold_1_beta_1000_delta_100_epoch_400.pt'
                PATH = '/home-local/ConnectomeML/Models/Model_G_mod_age_fold_{}_beta_{}_delta_{}_epoch_200.pt'.format(fold, beta, delta)
                #PATH = '/home-local/ConnectomeML/Models/Model_E_charpath_fold_1_beta_{}_delta_{}_epoch_450.pt'.format(beta, delta)
	        
                print(PATH)
                model = Conditional_VAE(in_dim=CONNECTOME_SIZE,c_dim=1, z_dim=100).to(DEVICE)

                if not os.path.isfile(PATH):
                        print("SKIPPING")
                        break
                    
                model.load_state_dict(torch.load(PATH))    
                
                model.eval()

                for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, img_paths in train_loader:
                        print(batch_predictionlabels)
                        batch_features = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE)
                        batch_predictionlabels = batch_predictionlabels.view(-1,1).to(DEVICE)
                        
                        batch_sitelabels = batch_sitelabels.view(-1,1).to(DEVICE)
                        clf = RandomForestClassifier(max_depth=2, random_state=0)
                        x_hat, z, z_mu, z_log_sigma_sq, model_predictions_site1, model_predictions_site2, model_predictions_age = model.forward_train(batch_features,batch_sitelabels)
                        y = batch_sitelabels.detach().cpu()
                        X = z.detach().cpu()
                        clf.fit(X, y)
                
                test_examples = None
                plt.figure(figsize=(20, 4))
                index = 0
                number = len(test_loader)

                site1predictions = np.zeros((len(test_loader),1))
                site2predictions = np.zeros((len(test_loader),1))
                pred_labels = np.zeros((len(test_loader),1))
                site_labels = np.zeros((len(test_loader),1))
                predictions = np.zeros((len(test_loader),1))
                ageerrors = np.zeros((len(test_loader),1))
                filenames = []
                seed_value = 42

                with torch.no_grad():

                        for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels , _ in test_loader:

                                # Send features, prediction labels, and site labels to device and set dimensions
                                batch_features = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE)
                                batch_predictionlabels = batch_predictionlabels.view(-1,1).to(DEVICE)
                                batch_sitelabels = batch_sitelabels.view(-1,1).to(DEVICE)
                                batch_agelabels = batch_agelabels.view(-1,1).to(DEVICE)

                                # Put data through forward pass
                                x_hat, z, z_mu, z_log_sigma_sq, model_predictions_site1, model_predictions_site2, model_predictions_age = model.forward_train(batch_features,batch_sitelabels)
                                
                                #print(x_hat.shape, z.shape, z_mu.shape, z_log_sigma_sq.shape)
                                # Reconstruction and KL divergence loss terms
                                recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)
                                
                                # Weight KL loss with beta, set parameter at top
                                kl_loss = beta * kl_loss
                                
                                # Prediction Loss term
                                # We don't want to model to learn from predictions from another site. Thus, we will make sure the loss is zero for those elements of the batch
                            
                                site1_indicator = (batch_sitelabels == 1).to(torch.float32)
                                site2_indicator = (batch_sitelabels == 2).to(torch.float32)

                                # Errors by site
                                model_prediction_error_site1 = site1_indicator * (model_predictions_site1 - batch_predictionlabels)
                                model_prediction_error_site2 = site2_indicator * (model_predictions_site2 - batch_predictionlabels)

                                prediction_error = model_prediction_error_site1 + model_prediction_error_site2

                                # MSE prediction loss
                                prediction_error = torch.mean(torch.square(prediction_error))
                                age_error = torch.mean(torch.square(batch_agelabels-model_predictions_age))
                                #prediction_loss_site1 = criterion_site1(batch_predictionlabels, model_predictions_site1)
                                #prediction_loss_site2 = criterion_site2(batch_predictionlabels, model_predictions_site2)
                                
                                # Total Loss
                                #loss = kl_loss + alpha * recon_loss + delta * prediction_error + age_error

                                # Compute loss gradients and take a step with the Adam optimizer

                                # Add the mini-batch training loss to epoch loss
                                
                                print("Training predictions:", batch_sitelabels[0:5] ,batch_predictionlabels[0:5], model_predictions_site1[0:5], model_predictions_site2[0:5])  
                                print("Age predictions:", batch_agelabels[0:5], model_predictions_age[0:5])
                            
                                print("RF accuracy",clf.score(z.detach().cpu(), batch_sitelabels.detach().cpu()))
                                rfacc.append(clf.score(z.detach().cpu(), batch_sitelabels.detach().cpu()))
             
                                
                                #validation_MI_site1 = mutual_info_regression(model_predictions_site1.detach().cpu().reshape(-1,1), batch_sitelabels.detach().cpu())
                                #validation_MI_site2 = mutual_info_regression(model_predictions_site2.detach().cpu().reshape(-1,1), batch_sitelabels.detach().cpu())
                                            
                                #print("Z shape",z.detach().cpu().shape)
                                #print("Batch site labels",batch_sitelabels.detach().cpu().shape)
                                #MI_z_site = mutual_info_regression(z.detach().cpu(), batch_sitelabels.detach().cpu())
                                #print(MI_z_site)
                                #print(MI_z_site.shape)
                                #print(recon_loss.item())
                                #print(prediction_error.item())
                                #print(validation_MI_site1)
                                #print(validation_MI_site2)
                                #    print("Prediction site 1:", prediction_site1, "Prediction site 2:", prediction_site2)
                                model_prediction_error_site1 = torch.mean(torch.square(model_prediction_error_site1))
                                model_prediction_error_site2 = torch.mean(torch.square(model_prediction_error_site2))
                                experimentname.append("beta_{}_delta_{}_fold_{}".format(beta, delta, fold))
                                predictionloss_site1.append(model_prediction_error_site1.item())
                                predictionloss_site2.append(model_prediction_error_site2.item())
                                age_loss.append(age_error.item())
                                #mutualinformation.append(MI_z_site)
                                site1predictions = model_predictions_site1.detach().cpu()
                                site2predictions = model_predictions_site2.detach().cpu()
                                """
                                if index<=4:
                                    ax = plt.subplot(2, 5, index + 1)
                                    plt.imshow(batch_features[index].cpu().numpy().reshape(84,84))
                                    ax.get_xaxis().set_visible(False)
                                    ax.get_yaxis().set_visible(False)
                                    ax = plt.subplot(2, 5, index + 1 + 5)
                                    plt.imshow(x_hat[index].cpu().numpy().reshape(84,84))        
                                    ax.get_xaxis().set_visible(False)
                                    ax.get_yaxis().set_visible(False)
                                    #print("Prediction:",model_predictions_site1, model_predictions_site2,"Label:",batch_predictionlabels, "Site", batch_sitelabels)
                                           
                                   
                                    
                                    
                                    plt.savefig(r'tutorial_output_connectome.png')
                                    plt.show()
                                    """
                                
                                index = index+1
                                for sample in np.arange(0,len(batch_sitelabels)):
                                        connectome = x_hat[sample].cpu().numpy().reshape(84,84)
                                        np.savetxt("connectome_sample_{}_fold_{}_beta_{}_delta_{}_modularityage.csv".format(sample, fold, beta, delta), connectome, delimiter=",")

                                np.savetxt('/home-local/ConnectomeML/Models/predictionsite1_{}_{}_{}_modularityage_forpaper.csv'.format(fold, beta, delta), site1predictions, delimiter=',')        
                                np.savetxt('/home-local/ConnectomeML/Models/predictionsite2_{}_{}_{}_modularityage_forpaper.csv'.format(fold, beta, delta), site2predictions, delimiter=',')
                                np.savetxt('/home-local/ConnectomeML/Models/predictionage_{}_{}_{}_modularityage_forpaper.csv'.format(fold, beta, delta), model_predictions_age.detach().cpu(), delimiter=',')           
                                np.savetxt('/home-local/ConnectomeML/Models/predictionlabels_modularity_forpaper.csv', pred_labels, delimiter=',')   
                                np.savetxt('/home-local/ConnectomeML/Models/sitelabels_modularity_forpaper.csv', site_labels, delimiter=',')  
                                #np.savetxt('/home-local/ConnectomeML/Models/predictions.csv', predictions, delimiter=',')  
                                #np.savetxt('/home-local/ConnectomeML/Models/filenames.csv', filenames, delimiter=',')        
                                with open("/home-local/ConnectomeML/Models/filenames.txt", 'w') as f:
                                        for s in filenames:
                                                f.write(str(s) + '\n')
                
df = pd.DataFrame({'Experiment': experimentname, 'PredictionLoss Site 1': predictionloss_site1, 'PredictionLoss Site 2': predictionloss_site2, 'RFPrediction': rfacc, 'Prediction Loss Age': age_loss})
df.to_csv('/home-local/ConnectomeML/report.csv')

#print(site1predictions)
#print(site2predictions)

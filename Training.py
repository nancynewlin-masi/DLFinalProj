import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
from torch.utils.data import Dataset, DataLoader
from Dataloader_contrastive import CustomDataset
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from Architecture import AE, Conditional_VAE
import losses
from sklearn.feature_selection import mutual_info_regression


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

CONNECTOME_SIZE = 84*84
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.utils.tensorboard import SummaryWriter
tensorboard_dir = '/home-local/ConnectomeML/Tensorboard_vae_mlp_5-46_8-8/'
writer = SummaryWriter(tensorboard_dir)

delta = 100
alpha = 100
batch_size = 1050
epochs = 500
learning_rate = 1e-3
k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = CustomDataset()		

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
for delta in [1]:
    for beta in [0, 1, 10, 100]:
        ##  use gpu if available
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

            print('Fold {}'.format(fold + 1))

            # Create data loaders for training and validation using random selections
            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)

            # create a model from Arch (conditional variational autoencoder)
            # load it to the specified device, either gpu or cpu
            model = Conditional_VAE(in_dim=CONNECTOME_SIZE,c_dim=1, z_dim=100).to(DEVICE)
            #model.train()

            # create an optimizer object
            # Adam optimizer with learning rate 1e-3
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Define losses
            criterion_site1 = nn.MSELoss()
            criterion_site2 = nn.MSELoss()
            loss_fn = losses.BetaVAE_Loss(beta)

            for epoch in range(epochs):

                # Set all losses to zero
                loss = 0
                training_loss_total = 0
                training_loss_reconstruction = 0
                training_loss_kldivergence = 0
                training_loss_prediction_site1 = 0
                training_loss_prediction_site2 = 0
                countofsite1 = 0
                countofsite2 = 0
                
                for batch_features, batch_predictionlabels, batch_sitelabels, _, _ in train_loader:

                    # Send features, prediction labels, and site labels to device and set dimensions
                    batch_features = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE)
                    batch_predictionlabels = batch_predictionlabels.view(-1,1).to(DEVICE)
                    batch_sitelabels = batch_sitelabels.view(-1,1).to(DEVICE)

                    optimizer.zero_grad()

                    # Put data through forward pass
                    x_hat, z, z_mu, z_log_sigma_sq, model_predictions_site1, model_predictions_site2 = model.forward_train(batch_features,batch_sitelabels)
                    
                    print(x_hat.shape, z.shape, z_mu.shape, z_log_sigma_sq.shape)
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

                    #prediction_loss_site1 = criterion_site1(batch_predictionlabels, model_predictions_site1)
                    #prediction_loss_site2 = criterion_site2(batch_predictionlabels, model_predictions_site2)
                    
                    # Total Loss
                    loss = kl_loss + alpha * recon_loss + delta * prediction_error

                    # Compute loss gradients and take a step with the Adam optimizer
                    loss.backward()
                    optimizer.step()

                    # Add the mini-batch training loss to epoch loss
                    training_loss_total             += loss.item()      
                    training_loss_reconstruction    += recon_loss.item()
                    training_loss_kldivergence      += kl_loss.item()
                    countofsite1                    += countofsite1 + torch.sum(batch_sitelabels==1)
                    countofsite2                    += countofsite2 + torch.sum(batch_sitelabels==2)
                    training_loss_prediction_site1  += model_prediction_error_site1.square().sum().item()
                    training_loss_prediction_site2  += model_prediction_error_site2.square().sum().item()

                    print("Training predictions:", batch_sitelabels[0:5] ,batch_predictionlabels[0:5], model_predictions_site1[0:5], model_predictions_site2[0:5])  

                    
                print("Count of site 1", countofsite1, "site 2", countofsite2)

                # compute the epoch training loss
                training_loss_total = training_loss_total / len(train_loader)
                training_loss_reconstruction = training_loss_reconstruction / len(train_loader)
                training_loss_kldivergence = training_loss_kldivergence /  len(train_loader)
                
                training_loss_prediction_site1 = training_loss_prediction_site1 / countofsite1
                training_loss_prediction_site2 = training_loss_prediction_site2 / countofsite2

                    
                validation_loss = 0
                validation_loss_reconstruction = 0
                validation_loss_prediction_site1 = 0
                validation_loss_prediction_site2 = 0
                validation_loss_total = 0
                validation_loss_kl = 0
                countofsite1 = 0
                countofsite2 = 0

                for batch_features, batch_predictionlabels, batch_sitelabels, _, _ in validation_loader:
                    batch_features = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE)
                    batch_predictionlabels = batch_predictionlabels.view(-1,1).to(DEVICE)
                    batch_sitelabels = batch_sitelabels.view(-1,1).to(DEVICE)

                    x_hat, z, z_mu, z_log_sigma_sq, model_predictions_site1, model_predictions_site2 = model.forward_train(batch_features,batch_sitelabels)
                    recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)            
                    
                    kl_loss = kl_loss * beta

                    site1_indicator = (batch_sitelabels == 1).to(torch.float32)
                    site2_indicator = (batch_sitelabels == 2).to(torch.float32)

                    # Errors by site
                    model_prediction_error_site1 = site1_indicator * (model_predictions_site1 - batch_predictionlabels)
                    model_prediction_error_site2 = site2_indicator * (model_predictions_site2 - batch_predictionlabels)


                    validation_loss_reconstruction += recon_loss.item()
                    validation_loss_kl += kl_loss.item()
                    validation_loss_prediction_site1 += model_prediction_error_site1.square().sum().item()
                    validation_loss_prediction_site2 += model_prediction_error_site2.square().sum().item()
                    #print("Validation prediction loss",validation_loss_prediction_site1, validation_loss_prediction_site2)
                    countofsite1 = countofsite1 + torch.sum(batch_sitelabels==1)
                    countofsite2 = countofsite2 + torch.sum(batch_sitelabels==2)  
                    validation_MI_site1 = mutual_info_regression(model_predictions_site1.detach().cpu(), batch_sitelabels.detach().cpu())
                    validation_MI_site2 = mutual_info_regression(model_predictions_site2.detach().cpu(), batch_sitelabels.detach().cpu())
                    
                    print("Mutual info score",mutual_info_regression(model_predictions_site1.detach().cpu(), batch_sitelabels.detach().cpu()))
                    print("Mutual info score",mutual_info_regression(model_predictions_site2.detach().cpu(), batch_sitelabels.detach().cpu()))
    
                    print("Predictions:", batch_sitelabels[0:5] ,batch_predictionlabels[0:5], model_predictions_site1[0:5], model_predictions_site2[0:5])          

                #print("Count of sites: " ,countofsite1, countofsite2)
                validation_loss_reconstruction = validation_loss_reconstruction / len(validation_loader)
                
                print("Validation prediction loss", validation_loss_prediction_site1, validation_loss_prediction_site2)
                validation_loss_prediction_site1 = validation_loss_prediction_site1 / countofsite1
                validation_loss_prediction_site2 = validation_loss_prediction_site2 / countofsite2
                validation_loss_kl = validation_loss_kl / len(validation_loader)
                
                #model.eval()
                writer.add_scalars('Total Loss Beta {} Delta {} '.format(beta, delta),   {'Training Loss Total Fold {}'.format(fold + 1): training_loss_total}, epoch)
                writer.add_scalars('Prediction Loss Site 1 Beta {} Delta {}'.format(beta, delta),   {'Training Loss Prediction Site 1 Fold {}'.format(fold + 1): training_loss_prediction_site1, 'Validation Loss Prediction Site 1 Fold {}'.format(fold + 1): validation_loss_prediction_site1},epoch) 
                writer.add_scalars('Prediction Loss Site 2 Beta {} Delta {}'.format(beta, delta),   {'Training Loss Prediction Site 2 Fold {}'.format(fold + 1): training_loss_prediction_site2, 'Validation Loss Prediction Site 2 Fold {}'.format(fold + 1): validation_loss_prediction_site2}, epoch)
                writer.add_scalars('Reconstruction Loss Beta {} Delta {}'.format(beta, delta),   {'Training Loss Reconstruction Fold {}'.format(fold + 1): training_loss_reconstruction, 'Validation Loss Reconstruction Fold {}'.format(fold + 1): validation_loss_reconstruction}, epoch)
                writer.add_scalars('KL Loss Beta {} Delta {}'.format(beta, delta),   {'Training Loss KL Fold {}'.format(fold + 1): training_loss_kldivergence, 'Validation Loss KL Fold {}'.format(fold+1): validation_loss_kl}, epoch)
                writer.add_scalars('Mutual Information Beta {} Delta {}'.format(beta, delta),   {'Validation Mutual information Site 1 Fold {}'.format(fold + 1): validation_MI_site1, 'Validation Mutual Inftomation site 2 Fold {}'.format(fold+1): validation_MI_site2}, epoch)
                
                print(epoch, training_loss_total)
                if(epoch%50==0):
                    PATH = '/home-local/ConnectomeML/Models/Model_D_mod_fold_{}_beta_{}_delta_{}_epoch_{}.pt'.format(fold+1, beta, delta,epoch)
                    torch.save(model.state_dict(), PATH)

            PATH = '/home-local/ConnectomeML/Models/Model_D_mod_fold_{}_beta_{}_delta_{}.pt'.format(fold+1, beta, delta)
            torch.save(model.state_dict(), PATH)
            

        #/home-local/ConnectomeML/Models/Model_T_fold_3_beta_0_delta_1_epoch_600.pt
    

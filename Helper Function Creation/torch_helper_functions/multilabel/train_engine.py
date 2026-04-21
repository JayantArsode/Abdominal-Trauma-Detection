"""
Helper class for training and validation steps in a convolutional neural network for mutilabel classification
Also it will save all model progress in weight and bias platform.

## Attributes:
  None
## Methods:
  `train_step`: Performs a single training step.
  `validation_step`: Performs a single validation step.
  `train`: Trains the model for a specified number of epochs.
"""
from torch import cuda,onnx,sigmoid,softmax,round,inference_mode,randperm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Tuple,Dict,List
from collections import defaultdict
import wandb
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy,MulticlassAccuracy
from torch_helper_functions import utils
import pandas as pd
import os
import numpy as np

def shuffle_minibatch(X, y_bowel, y_extra, y_kidney, y_liver, y_spleen):
    """
    Shuffle the minibatch of inputs and labels.

    Args:
        X (torch.Tensor): Input images.
        y_bowel (torch.Tensor): Labels for bowel task.
        y_extra (torch.Tensor): Labels for extra task.
        y_kidney (torch.Tensor): Labels for kidney task.
        y_liver (torch.Tensor): Labels for liver task.
        y_spleen (torch.Tensor): Labels for spleen task.

    Returns:
        tuple: Tuple containing shuffled input images and corresponding shuffled labels.
    """
    batch_size = X.size(0)
    idx = randperm(batch_size)

    X_shuffled = X[idx]
    y_bowel_shuffled = y_bowel[idx]
    y_extra_shuffled = y_extra[idx]
    y_kidney_shuffled = y_kidney[idx]
    y_liver_shuffled = y_liver[idx]
    y_spleen_shuffled = y_spleen[idx]

    return X_shuffled, y_bowel_shuffled, y_extra_shuffled, y_kidney_shuffled, y_liver_shuffled, y_spleen_shuffled

def train_step_cutmix(model: nn.Module,
               dataloader: DataLoader,
               criterion_bowel: nn.Module,
               criterion_extra: nn.Module,
               criterion_kidney: nn.Module,
               criterion_liver: nn.Module,
               criterion_spleen: nn.Module,
               binary_acc
               ,multi_acc
               ,optimizer: optim.Optimizer,
               CUTMIX_ALPHA):
    """ Perform a single training step with CutMix data augmentation.
    Parameters:
    model (nn.Module): The neural network model.
    dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    criterion_bowel (nn.Module): Loss function for bowel task.
    criterion_extra (nn.Module): Loss function for extra task.
    criterion_liver (nn.Module): Loss function for liver task.
    criterion_kidney (nn.Module): Loss function for kidney task.
    criterion_spleen (nn.Module): Loss function for spleen task.
    binary_acc: Function to compute binary accuracy.
    multi_acc: Function to compute multi-class accuracy.
    optimizer (optim.Optimizer): Optimizer for gradient descent.
    EPOCHS (int): Total number of training epochs.
    CUTMIX_ALPHA (float): Alpha parameter for CutMix.
    device (str): Device to use for training (e.g., 'cpu' or 'cuda').
    Returns:
    Tuple: Tuple containing training loss, accuracy for bowel, extra, liver, kidney, spleen, f1 score for bowel, extra, liver, kidney, spleen.
    """
    # Setting up device
    device = "cuda" if cuda.is_available() else "cpu"
    # Create train metrics variables
    train_loss, train_acc_bowel, train_acc_extra, train_acc_liver, train_acc_kidney, train_acc_spleen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Creating training loop
    model.train()
    total_steps = len(dataloader)
    for i, (X, y_bowel, y_extra, y_kidney, y_liver, y_spleen) in tqdm(enumerate(dataloader), desc="Training", leave=False, total=len(dataloader)):
      X, y_bowel, y_extra, y_kidney, y_liver, y_spleen = X.to(device), y_bowel.to(device), y_extra.to(device), y_kidney.to(device), y_liver.to(device), y_spleen.to(device)
      _, C, H, W = X.size()
      # CutMix data augmentation
      cutmix_decision = np.random.rand()
      if cutmix_decision > 0.50:
        # Cutmix: https://arxiv.org/pdf/1905.04899.pdf
        X_shuffled, y_bowel_shuffled, y_extra_shuffled, y_kidney_shuffled, y_liver_shuffled, y_spleen_shuffled = shuffle_minibatch(X, y_bowel, y_extra, y_kidney, y_liver, y_spleen)
        lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        X[:, :, bbx1:bbx2, bby1:bby2] = X_shuffled[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)

      # 1. Forward pass
      bowel_out, extra_out, kidney_out, liver_out, spleen_out = model(X)

      # 2. Calculate loss/acc
      if cutmix_decision > 0.50:
        loss_bowel = criterion_bowel(bowel_out, y_bowel) * lam + criterion_bowel(bowel_out, y_bowel_shuffled) * (1. - lam)
        loss_extra = criterion_extra(extra_out, y_extra) * lam + criterion_extra(extra_out, y_extra_shuffled) * (1. - lam)
        loss_kidney = criterion_kidney(kidney_out, y_kidney) * lam + criterion_kidney(kidney_out, y_kidney_shuffled) * (1. - lam)
        loss_liver = criterion_liver(liver_out, y_liver) * lam + criterion_liver(liver_out, y_liver_shuffled) * (1. - lam)
        loss_spleen = criterion_spleen(spleen_out, y_spleen) * lam + criterion_spleen(spleen_out, y_spleen_shuffled) * (1. - lam)
      else:
        loss_bowel = criterion_bowel(bowel_out, y_bowel)
        loss_extra = criterion_extra(extra_out, y_extra)
        loss_kidney = criterion_kidney(kidney_out, y_kidney)
        loss_liver = criterion_liver(liver_out, y_liver)
        loss_spleen = criterion_spleen(spleen_out, y_spleen)

      loss = loss_bowel + loss_extra + loss_kidney + loss_liver + loss_spleen
      train_loss += loss.item()

      # Converting binary labels to true
      bowel_out = round(sigmoid(bowel_out))
      extra_out = round(sigmoid(extra_out))
      kidney_out = softmax(kidney_out, dim=1).argmax(dim=1)
      liver_out = softmax(liver_out, dim=1).argmax(dim=1)
      spleen_out = softmax(spleen_out, dim=1).argmax(dim=1)

      # Getting true labels for multiclass classification
      y_kidney = y_kidney.argmax(dim=1)
      y_liver = y_liver.argmax(dim=1)
      y_spleen = y_spleen.argmax(dim=1)

      # Compute accuracy for each task
      train_acc_bowel += binary_acc(bowel_out, y_bowel).item()
      train_acc_extra += binary_acc(extra_out, y_extra).item()
      train_acc_kidney += multi_acc(kidney_out, y_kidney).item()
      train_acc_liver += multi_acc(liver_out, y_liver).item()
      train_acc_spleen += multi_acc(spleen_out, y_spleen).item()

      # 3. Zero grad
      optimizer.zero_grad()

      # 4. Backward pass
      loss.backward()

      # 5. Optimizer step
      optimizer.step()
    num_batches = len(dataloader)
    train_loss /= num_batches
    train_acc_bowel /= num_batches
    train_acc_extra /= num_batches
    train_acc_kidney /= num_batches
    train_acc_liver /= num_batches
    train_acc_spleen /= num_batches

    return train_loss, train_acc_bowel, train_acc_extra, train_acc_kidney, train_acc_liver, train_acc_spleen

def train_step(model: nn.Module,
               dataloader: DataLoader,
               criterion_bowel: nn.Module,
               criterion_extra: nn.Module,
               criterion_kidney: nn.Module,
               criterion_liver: nn.Module,
               criterion_spleen: nn.Module,
               binary_acc,
               multi_acc,
               optimizer: optim.Optimizer):
    """
    Perform a single training step.

    Parameters:
        model (nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion_bowel (nn.Module): Loss function for bowel task.
        criterion_extra (nn.Module): Loss function for extra task.
        criterion_liver (nn.Module): Loss function for liver task.
        criterion_kidney (nn.Module): Loss function for kidney task.
        criterion_spleen (nn.Module): Loss function for spleen task.
        optimizer (optim.Optimizer): Optimizer for gradient descent.

    Returns:
        Tuple: Tuple containing training loss, accuracy for bowel, extra, liver, kidney, spleen,
               f1 score for bowel, extra, liver, kidney, spleen.
    """
    device = "cuda" if cuda.is_available() else "cpu"
    # Create train metrics variables
    train_loss, train_acc_bowel, train_acc_extra, train_acc_liver, train_acc_kidney, train_acc_spleen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Creating training loop
    model.train()
    for batch, (X, y_bowel, y_extra, y_kidney, y_liver, y_spleen) in tqdm(enumerate(dataloader), desc="Training", leave=False, total=len(dataloader)):
        X, y_bowel, y_extra, y_kidney, y_liver, y_spleen = X.to(device), y_bowel.to(device), y_extra.to(device), y_kidney.to(device), y_liver.to(device), y_spleen.to(device)
        # 1. Forward pass
        bowel_out, extra_out, kidney_out, liver_out, spleen_out = model(X)

        # 2. Calculate loss/acc
        loss_bowel = criterion_bowel(bowel_out, y_bowel)
        loss_extra = criterion_extra(extra_out, y_extra)
        loss_kidney = criterion_kidney(kidney_out, y_kidney)
        loss_liver = criterion_liver(liver_out, y_liver)
        loss_spleen = criterion_spleen(spleen_out, y_spleen)
        loss = loss_bowel + loss_extra + loss_kidney + loss_liver + loss_spleen
        train_loss += loss.item()

        # Converting binary labels to true
        bowel_out=round(sigmoid(bowel_out))
        extra_out=round(sigmoid(extra_out))
        kidney_out=softmax(kidney_out,dim=1).argmax(dim=1)
        liver_out=softmax(liver_out,dim=1).argmax(dim=1)
        spleen_out=softmax(spleen_out,dim=1).argmax(dim=1)
        # Getting true labels for muticlass classification
        y_kidney=y_kidney.argmax(dim=1)
        y_liver=y_liver.argmax(dim=1)
        y_spleen=y_spleen.argmax(dim=1)

        # Compute accuracy for each task
        train_acc_bowel += binary_acc(bowel_out, y_bowel).item()
        train_acc_extra += binary_acc(extra_out, y_extra).item()
        train_acc_kidney += multi_acc(kidney_out, y_kidney).item()
        train_acc_liver += multi_acc(liver_out, y_liver).item()
        train_acc_spleen += multi_acc(spleen_out, y_spleen).item()

        # 3. Zero grad
        optimizer.zero_grad()
        # 4. Backward pass
        loss.backward()
        # 5. Optimizer step
        optimizer.step()

    # Compute the average metrics
    num_batches = len(dataloader)
    train_loss /= num_batches
    train_acc_bowel /= num_batches
    train_acc_extra /= num_batches
    train_acc_kidney /= num_batches
    train_acc_liver /= num_batches
    train_acc_spleen /= num_batches

    return train_loss, train_acc_bowel, train_acc_extra, train_acc_kidney, train_acc_liver, train_acc_spleen


def validation_step(model: nn.Module,
                    dataloader: DataLoader,
                    criterion_bowel: nn.Module,
                    criterion_extra: nn.Module,
                    criterion_kidney: nn.Module,
                    criterion_liver: nn.Module,
                    criterion_spleen: nn.Module,
                    binary_acc,
                    multi_acc):
    """
    Perform a single validation step.

    Parameters:
        model (nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion_bowel (nn.Module): Loss function for bowel task.
        criterion_extra (nn.Module): Loss function for extra task.
        criterion_liver (nn.Module): Loss function for liver task.
        criterion_kidney (nn.Module): Loss function for kidney task.
        criterion_spleen (nn.Module): Loss function for spleen task.

    Returns:
        tuple: Tuple containing validation loss, accuracy for each task, and F1 score for each task.
    """
    device = "cuda" if cuda.is_available() else "cpu"
    # Create validation metrics variables
    val_loss, val_acc_bowel, val_acc_extra, val_acc_liver, val_acc_kidney, val_acc_spleen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Set model to evaluation mode
    model.eval()

    # Iterate over validation dataloader
    with inference_mode():
        for batch_idx, (X, y_bowel, y_extra, y_kidney, y_liver, y_spleen) in tqdm(enumerate(dataloader), desc="Validation", leave=False, total=len(dataloader)):
            # Move data to appropriate device
            X, y_bowel, y_extra, y_kidney, y_liver, y_spleen = X.to(device), y_bowel.to(device), y_extra.to(device), y_kidney.to(device), y_liver.to(device), y_spleen.to(device)

            # Forward pass
            bowel_out, extra_out, kidney_out, liver_out, spleen_out = model(X)

            # Calculate loss
            loss_bowel = criterion_bowel(bowel_out, y_bowel)
            loss_extra = criterion_extra(extra_out, y_extra)
            loss_kidney = criterion_kidney(kidney_out, y_kidney)
            loss_liver = criterion_liver(liver_out, y_liver)
            loss_spleen = criterion_spleen(spleen_out, y_spleen)

            # Accumulate loss
            val_loss += (loss_bowel.item() + loss_extra.item() + loss_liver.item() + loss_kidney.item() + loss_spleen.item())

            # Converting binary labels to true
            bowel_out=round(sigmoid(bowel_out))
            extra_out=round(sigmoid(extra_out))
            kidney_out=softmax(kidney_out,dim=1).argmax(dim=1)
            liver_out=softmax(liver_out,dim=1).argmax(dim=1)
            spleen_out=softmax(spleen_out,dim=1).argmax(dim=1)
            # Getting true labels for muticlass classification
            y_kidney=y_kidney.argmax(dim=1)
            y_liver=y_liver.argmax(dim=1)
            y_spleen=y_spleen.argmax(dim=1)

            # Compute accuracy for each task
            val_acc_bowel += binary_acc(bowel_out, y_bowel).item()
            val_acc_extra += binary_acc(extra_out, y_extra).item()
            val_acc_kidney += multi_acc(kidney_out, y_kidney).item()
            val_acc_liver += multi_acc(liver_out, y_liver).item()
            val_acc_spleen += multi_acc(spleen_out, y_spleen).item()

    # Compute average loss, accuracy, and F1 score
    num_batches = len(dataloader)
    val_loss /= (num_batches)
    val_acc_bowel /= num_batches
    val_acc_extra /= num_batches
    val_acc_kidney /= num_batches
    val_acc_liver /= num_batches
    val_acc_spleen /= num_batches

    return val_loss, val_acc_bowel, val_acc_extra, val_acc_kidney, val_acc_liver, val_acc_spleen

from tabulate import tabulate
# Define a function to print the DataFrame after each epoch
def print_model_train_results(results):
    df_results = pd.DataFrame(results)
    # Set 'Type' as the index
    df_results.set_index("Results", inplace=True)
    print(tabulate(df_results, headers='keys', tablefmt='fancy_grid'),"\n")

def train(model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: optim.Optimizer,
          criterion_bowel: nn.Module,
          criterion_extra: nn.Module,
          criterion_kidney: nn.Module,
          criterion_liver: nn.Module,
          criterion_spleen: nn.Module,
          wandb_init_params: Dict = {"project": "Demo_Project",
                                     "experiment": "Demo_Experiment",
                                     "hyperparameters": {
                                         "learning_rate": 0.0001,
                                         "epochs": 5,
                                         "batch_size": 32
                                     }},
          epochs: int = 5,
          lr_scheduler: optim.lr_scheduler = None,
          early_stopping: Dict = {"patience": 5},
          CUTMIX_ALPHA=None) -> Dict:
    """
    Train the model for a specified number of epochs.

    Parameters:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        optimizer (optim.Optimizer): Optimizer for gradient descent.
        criterion_bowel (nn.Module): Loss function for the bowel task.
        criterion_extra (nn.Module): Loss function for the extra task.
        criterion_kidney (nn.Module): Loss function for the kidney task.
        criterion_liver (nn.Module): Loss function for the liver task.
        criterion_spleen (nn.Module): Loss function for the spleen task.
        accuracy_fn (nn.Module): Accuracy metric.
        wandb_init_params (Dict): Parameters for initializing Wandb.
        epochs (int): Number of training epochs. Default is 5.
        lr_scheduler (optim.lr_scheduler): Learning rate scheduler. Default is None.
        early_stopping (Dict): Dictionary containing early stopping parameters. Default is {"patience": 5}.

    Returns:
        Dict: A dictionary containing training and validation metrics.
    """

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)
    images,temp_bowel, temp_extra, temp_kidney, temp_liver, temp_spleen = next(iter(val_dataloader))

    # Initialize Wandb
    wandb.init(project=wandb_init_params["project"], name=wandb_init_params["experiment"],
               config={"hyperparameters":wandb_init_params["hyperparameters"],
                       "criterions":{
                           "bowel":criterion_bowel,
                           "extravasation":criterion_extra,
                           "kidney":criterion_kidney,
                           "liver":criterion_liver,
                           "spleen":criterion_spleen
                       },
                       "metrics":{
                           "bowel,extravasation":{
                               "Accuracy":BinaryAccuracy
                           },
                           "kideny,liver,spleen":{
                               "num_classes": "3",
                               "Accuracy":MulticlassAccuracy
                           }
                       }})
    wandb.watch(model)

    # Getting accuracy and f1score
    binary_acc = BinaryAccuracy().to(device)
    multi_acc = MulticlassAccuracy(num_classes=3).to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement = 0

    results = defaultdict(list)
    results["model_name"]=wandb_init_params["experiment"]
    if CUTMIX_ALPHA is not None:
      print("Training With CutMix Augmentation")
    for epoch in tqdm(range(1, epochs + 1),desc="Epoch"):
        # Training
        if CUTMIX_ALPHA is None:
          train_loss, train_acc_bowel, train_acc_extra, train_acc_kidney, train_acc_liver,train_acc_spleen= train_step(
              model, train_dataloader,
              criterion_bowel, criterion_extra,
              criterion_kidney, criterion_liver,
              criterion_spleen,
              binary_acc,
              multi_acc,
              optimizer)
        else:
          train_loss, train_acc_bowel, train_acc_extra, train_acc_kidney, train_acc_liver,train_acc_spleen= train_step_cutmix(
               model, train_dataloader,
               criterion_bowel, criterion_extra,
               criterion_kidney, criterion_liver,
               criterion_spleen,
               binary_acc,
               multi_acc,
               optimizer,
               CUTMIX_ALPHA=CUTMIX_ALPHA)
        # Validation
        val_loss, val_acc_bowel, val_acc_extra, val_acc_kidney, val_acc_liver, val_acc_spleen = validation_step(
            model, val_dataloader,
            criterion_bowel, criterion_extra,
            criterion_kidney, criterion_liver,
            criterion_spleen,
            binary_acc,
            multi_acc)

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)
            print(f"lr:{optimizer.param_groups[0]['lr']}")

        wandb.log({
            "train_loss": train_loss,
            "train_acc_bowel": train_acc_bowel,
            "train_acc_extra": train_acc_extra,
            "train_acc_kidney": train_acc_kidney,
            "train_acc_liver": train_acc_liver,
            "train_acc_spleen": train_acc_spleen,
            "val_loss": val_loss,
            "val_acc_bowel": val_acc_bowel,
            "val_acc_extra": val_acc_extra,
            "val_acc_kidney": val_acc_kidney,
            "val_acc_liver": val_acc_liver,
            "val_acc_spleen": val_acc_spleen,
        })

        if lr_scheduler is not None:
            wandb.log({"learning_rate":optimizer.param_groups[0]['lr']})

        results["train_loss"].append(train_loss)
        results["train_acc_bowel"].append(train_acc_bowel)
        results["train_acc_extra"].append(train_acc_extra)
        results["train_acc_kidney"].append(train_acc_kidney)
        results["train_acc_liver"].append(train_acc_liver)
        results["train_acc_spleen"].append(train_acc_spleen)
        results["val_loss"].append(val_loss)
        results["val_acc_bowel"].append(val_acc_bowel)
        results["val_acc_extra"].append(val_acc_extra)
        results["val_acc_kidney"].append(val_acc_kidney)
        results["val_acc_liver"].append(val_acc_liver)
        results["val_acc_spleen"].append(val_acc_spleen)

        if lr_scheduler is not None:
            results["learning_rate"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch}:")
        print_model_train_results({
            "Results":["Train","Validation"],
            "loss":[train_loss,val_loss],
            "bowel": [train_acc_bowel,val_acc_bowel],
            "extravation": [train_acc_extra,val_acc_extra],
            "kidney": [train_acc_kidney,val_acc_kidney],
            "liver": [train_acc_liver,val_acc_liver],
            "spleen": [train_acc_spleen,val_acc_spleen]
            })

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement = 0
            print(f"Saving new best model")
            # Delete previous best epoch model if it exists
            if best_epoch > 0:
              previous_model_path = f"{wandb_init_params['experiment']}/epoch_{best_epoch - 1}.pth"
              if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
            utils.save_model(model, f"{wandb_init_params['experiment']}/epoch_{epoch}", f"{wandb_init_params['experiment']}.pth")
        else:
            no_improvement += 1

        if no_improvement >= early_stopping["patience"]:
            wandb.log({
                "train_loss": train_loss,
                "train_acc_bowel": train_acc_bowel,
                "train_acc_extra": train_acc_extra,
                "train_acc_kidney": train_acc_kidney,
                "train_acc_liver": train_acc_liver,
                "train_acc_spleen": train_acc_spleen,
                "val_loss": val_loss,
                "val_acc_bowel": val_acc_bowel,
                "val_acc_extra": val_acc_extra,
                "val_acc_kidney": val_acc_kidney,
                "val_acc_liver": val_acc_liver,
                "val_acc_spleen": val_acc_spleen,
            })
            if lr_scheduler is not None:
                wandb.log({"learning_rate":optimizer.param_groups[0]['lr']})

            results["train_loss"].append(train_loss)
            results["train_acc_bowel"].append(train_acc_bowel)
            results["train_acc_extra"].append(train_acc_extra)
            results["train_acc_kidney"].append(train_acc_kidney)
            results["train_acc_liver"].append(train_acc_liver)
            results["train_acc_spleen"].append(train_acc_spleen)
            results["val_loss"].append(val_loss)
            results["val_acc_bowel"].append(val_acc_bowel)
            results["val_acc_extra"].append(val_acc_extra)
            results["val_acc_kidney"].append(val_acc_kidney)
            results["val_acc_liver"].append(val_acc_liver)
            results["val_acc_spleen"].append(val_acc_spleen)

            if lr_scheduler is not None:
                results["learning_rate"].append(optimizer.param_groups[0]['lr'])

            wandb.log({"early_stopping_epoch": epoch})
            wandb.log({"early_stopping_reason": "No improvement in validation loss"})
            print(f"Early stopping at epoch {epoch}")
            print(f"Saving last epoch model")
            utils.save_model(model, f"{wandb_init_params['experiment']}/epoch_{epoch}", f"{wandb_init_params['experiment']}.pth")
            break

    print(f"Saving last epoch model")
    utils.save_model(model, f"{wandb_init_params['experiment']}/epoch_{epoch}", f"{wandb_init_params['experiment']}.pth")
    print(f"Best validation loss: {best_val_loss} at epoch {best_epoch}")
    model.eval()
    # Save the model in the exchangeable ONNX format
    onnx.export(model, images.to(device), f"{wandb_init_params['experiment']}.onnx")
    wandb.save(f"{wandb_init_params['experiment']}.onnx")
    wandb.finish()
    return results

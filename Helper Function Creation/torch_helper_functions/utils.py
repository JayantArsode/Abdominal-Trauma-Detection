"""
Files contains utility functions for PyTorch model training.
"""
from pathlib import Path
import torch
from typing import Dict,List
import pandas as pd
import torch
from torch import nn,sigmoid,softmax,cuda,inference_mode
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy,MulticlassAccuracy,BinaryF1Score,MulticlassF1Score
from tabulate import tabulate
from tqdm.auto import tqdm

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """
  #### Saves a PyTorch model to a target directory.

  ## Parameters:
    `model`: A target PyTorch model to save.
    `target_dir`: A directory for saving the model to.
    `model_name`: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  ## Return:
    None: This only saves the state dict(model weights) of
          the model that can be then used load in same model architecture.
  ## Example:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def calculate_class_weights(dataframe:pd.DataFrame,
                            label_columns:List[str]):
    """
    #### Calculate normalized class weights based on the frequencies of each class in the label columns of the dataframe.

    ## Args:
      `dataframe (pd.DataFrame)`: The dataframe containing the data.
      `label_columns (list)`: A list of column names containing labels.

    ## Returns:
     `dict`: A dictionary containing the normalized class weights.
     `list`: A list containing weights of classes for loss function.
    ## Example:
    Suppose we have a dataframe 'df' with the following structure:
    ```
          label_column
    0      class1
    1      class2
    2      class1
    3      class3
    4      class2
    ```
    We can calculate the class weights using:
    ```
    class_weights_dict,class_weights = calculate_class_weights(df, ['label_column'])
    ```
    This will return a dictionary like:
    ```
    {'class1': 1.0, 'class2': 2.0, 'class3': 2.0}
    ```
    Also class weight tensor like list for loss_fn.
    [1.0,2.0,2.0]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_frequencies = {}

    # Calculate class frequencies
    for label in label_columns:
        class_frequencies[label] = dataframe[label].sum()

    # Calculate total samples
    total_samples = sum(class_frequencies.values())

    # Calculate class weights
    class_weights = {cls: total_samples / freq for cls, freq in class_frequencies.items()}

    # Normalize class weights
    max_weight = max(class_weights.values())
    class_weights_normalized = {cls: max_weight / freq for cls, freq in class_frequencies.items()}

    # Convert to PyTorch tensor
    weights_tensor = torch.tensor(list(class_weights_normalized.values()),device=device).float()
    return class_weights_normalized, weights_tensor

from tabulate import tabulate
# Define a function to print the DataFrame after each epoch
def print_model_train_results(results):
    df_results = pd.DataFrame(results)
    # Set 'Type' as the index
    df_results.set_index("Results", inplace=True)
    print(tabulate(df_results, headers='keys', tablefmt='fancy_grid'),"\n")

def model_evaluation(model: nn.Module,
                    dataloader: DataLoader,
                    criterion_bowel: nn.Module,
                    criterion_extra: nn.Module,
                    criterion_kidney: nn.Module,
                    criterion_liver: nn.Module,
                    criterion_spleen: nn.Module):
    """
    Perform a model evaluation on testing data.

    Parameters:
        model (nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        criterion_bowel (nn.Module): Loss function for bowel task.
        criterion_extra (nn.Module): Loss function for extra task.
        criterion_liver (nn.Module): Loss function for liver task.
        criterion_kidney (nn.Module): Loss function for kidney task.
        criterion_spleen (nn.Module): Loss function for spleen task.

    Returns:
        No print model evaluation on testing data
    """
    device = "cuda" if cuda.is_available() else "cpu"

    # Getting accuracy and f1score
    binary_acc = BinaryAccuracy().to(device)
    multi_acc = MulticlassAccuracy(num_classes=3).to(device)
    binary_f1 = BinaryF1Score().to(device)
    multi_f1 = MulticlassF1Score(num_classes=3).to(device)

    # Create testing metrics variables
    test_loss, test_acc_bowel, test_acc_extra, test_acc_kidney, test_acc_liver, test_acc_spleen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    test_f1_bowel, test_f1_extra, test_f1_liver, test_f1_kidney, test_f1_spleen = 0.0, 0.0, 0.0, 0.0, 0.0

    # Set model to evaluation mode
    model.eval()

    # Iterate over testing dataloader
    with inference_mode():
        for batch_idx, (X, y_bowel, y_extra, y_kidney, y_liver, y_spleen) in tqdm(enumerate(dataloader), desc="Testing", total=len(dataloader)):
            # Move data to appropriate device
            X, y_bowel, y_extra, y_kidney,y_liver, y_spleen = X.to(device), y_bowel.to(device), y_extra.to(device), y_kidney.to(device), y_liver.to(device), y_spleen.to(device)

            # Forward pass
            bowel_out, extra_out, kidney_out,liver_out, spleen_out = model(X)

            y_bowel = y_bowel.view(-1,1)
            y_extra = y_extra.view(-1,1)
            # Calculate loss
            loss_bowel = criterion_bowel(bowel_out, y_bowel)
            loss_extra = criterion_extra(extra_out, y_extra)
            loss_kidney = criterion_kidney(kidney_out, y_kidney)
            loss_liver = criterion_liver(liver_out, y_liver)
            loss_spleen = criterion_spleen(spleen_out, y_spleen)

            # Accumulate loss
            test_loss += (loss_bowel.item() + loss_extra.item() + loss_kidney.item() + loss_liver.item() + loss_spleen.item())
            y_kidney=y_kidney.argmax(dim=1)
            y_liver=y_liver.argmax(dim=1)
            y_spleen=y_spleen.argmax(dim=1)
            # Compute accuracy for each task
            test_acc_bowel += binary_acc(sigmoid(bowel_out), y_bowel).item()
            test_acc_extra += binary_acc(sigmoid(extra_out), y_extra).item()
            test_acc_kidney += multi_acc(softmax(kidney_out,dim=1).argmax(dim=1), y_kidney).item()
            test_acc_liver += multi_acc(softmax(liver_out,dim=1).argmax(dim=1), y_liver).item()
            test_acc_spleen += multi_acc(softmax(spleen_out,dim=1).argmax(dim=1), y_spleen).item()

            # Compute F1 score for each task
            test_f1_bowel += binary_f1(sigmoid(bowel_out), y_bowel).item()
            test_f1_extra += binary_f1(sigmoid(extra_out), y_extra).item()
            test_f1_kidney += multi_f1(softmax(kidney_out,dim=1).argmax(dim=1), y_kidney).item()
            test_f1_liver += multi_f1(softmax(liver_out,dim=1).argmax(dim=1), y_liver).item()
            test_f1_spleen += multi_f1(softmax(spleen_out,dim=1).argmax(dim=1), y_spleen).item()

    # Compute average loss, accuracy, and F1 score
    num_batches = len(dataloader)
    test_loss /= (num_batches)
    test_acc_bowel /= num_batches
    test_acc_extra /= num_batches
    test_acc_kidney /= num_batches
    test_acc_liver /= num_batches
    test_acc_spleen /= num_batches
    test_f1_bowel /= num_batches
    test_f1_extra /= num_batches
    test_f1_kidney /= num_batches
    test_f1_liver /= num_batches
    test_f1_spleen /= num_batches

    print(f"Test loss is: {test_loss:.4f}")
    print_model_train_results({
    "Results":["Accuracy","F1_Score"],
    "bowel": [test_acc_bowel,test_f1_bowel],
    "extravation": [test_acc_extra,test_f1_extra],
    "kidney": [test_acc_kidney,test_f1_kidney],
    "liver": [test_acc_liver,test_f1_liver],
    "spleen": [test_acc_spleen,test_f1_spleen]
    })

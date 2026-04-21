"""
This contains utility realted to plotting model information.

## Functions:
  `plot_loss_accuracy_curve`:plot model training and evaluation loss and accuracy curves
"""
import matplotlib.pyplot as plt
from typing import Dict,List
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import seaborn as sns

def generate_confusion_matrix(model, dataloader):
    model.eval()
    all_preds = {task: [] for task in ['Bowel', 'Extra', 'Kidney', 'Liver', 'Spleen']}
    all_labels = {task: [] for task in ['Bowel', 'Extra', 'Kidney', 'Liver', 'Spleen']}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for images, y_bowel, y_extra, y_kidney, y_liver, y_spleen in dataloader:
            images = images.to(device)

            # Forward pass
            bowel_out, extra_out, kidney_out, liver_out, spleen_out = model(images)

            # Convert logits to predictions
            bowel_preds = (torch.sigmoid(bowel_out) > 0.5).int().cpu().numpy().flatten()
            extra_preds = (torch.sigmoid(extra_out) > 0.5).int().cpu().numpy().flatten()
            kidney_preds = np.argmax(F.softmax(kidney_out, dim=1).cpu().numpy(), axis=1).flatten()
            liver_preds = np.argmax(F.softmax(liver_out, dim=1).cpu().numpy(), axis=1).flatten()
            spleen_preds = np.argmax(F.softmax(spleen_out, dim=1).cpu().numpy(), axis=1).flatten()

            # Append predictions to the dictionary
            all_preds['Bowel'].append(bowel_preds)
            all_preds['Extra'].append(extra_preds)
            all_preds['Kidney'].append(kidney_preds)
            all_preds['Liver'].append(liver_preds)
            all_preds['Spleen'].append(spleen_preds)

            # Append true labels to the dictionary (already on CPU)
            all_labels['Bowel'].append(y_bowel.numpy())
            all_labels['Extra'].append(y_extra.numpy())
            all_labels['Kidney'].append(torch.argmax(y_kidney, dim=1).numpy())
            all_labels['Liver'].append(torch.argmax(y_liver, dim=1).numpy())
            all_labels['Spleen'].append(torch.argmax(y_spleen, dim=1).numpy())

    # Concatenate the lists within each task
    for task in ['Bowel', 'Extra', 'Kidney', 'Liver', 'Spleen']:
        all_preds[task] = np.concatenate(all_preds[task])
        all_labels[task] = np.concatenate(all_labels[task])

    # Create 2x2 subplot for confusion matrices
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Create confusion matrix for each task
    for i, task in enumerate(['Bowel', 'Extra', 'Kidney', 'Liver', 'Spleen']):
        cm = confusion_matrix(all_labels[task], all_preds[task])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix for {task}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide the remaining subplot
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_model_history_curves(model_history: Dict):
    """
    Plots the training history including loss and metrics for each class.

    Parameters:
    - model_history (dict): A dictionary containing training history data.
                      It should have keys like 'train_loss', 'val_loss',
                      'train_acc_classname', 'val_acc_classname', etc.

    Returns:
    - None
    """
    epochs = range(1, len(model_history["train_loss"]) + 1)
    # Plot train and validation losses for each class
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    title_position = (0.5, 1)  # Adjust the position as needed
    fig.suptitle(f"{model_history.get('model_name', 'Model')} Training History", fontsize=18,
                 position=title_position, fontweight='bold')
    axs = axs.ravel()
    handles = []
    labels = []
    train_line, = axs[0].plot(epochs, model_history['train_loss'], 'r', label='Train Loss')
    val_line, = axs[0].plot(epochs, model_history['val_loss'], 'b', label='Validation Loss')
    axs[0].set_title('Train and Validation Loss - All Classes')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    handles.extend([train_line, val_line])
    labels.extend([train_line.get_label(), val_line.get_label()])
    classes = ['bowel', 'extra', 'kidney','liver', 'spleen']

    colors = ['violet', 'orange']
    metrics = ['acc']

    for i, class_name in enumerate(classes):
        train_line, = axs[i + 1].plot(epochs, model_history[f'train_acc_{class_name}'], color=colors[0], linestyle='-', label=f'Train Accuracy')
        val_line, = axs[i + 1].plot(epochs, model_history[f'val_acc_{class_name}'], color=colors[1], linestyle='--', label=f'Validation Accuracy')
        if i == 0:
          handles.extend([train_line, val_line])
          labels.extend([train_line.get_label(), val_line.get_label()])

        axs[i + 1].set_title(f'Train and Validation Metrics - {class_name.capitalize()}')
        axs[i + 1].set_xlabel('Epochs')
        axs[i + 1].set_ylabel('Metrics')
        axs[i + 1].grid(True)

    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

    plt.tight_layout()
    plt.show()

# def plot_images_with_predictions(test_dataloader:torch.utils.data.DataLoader,
#                                  model:torch.nn.Module,
#                                  label_names:List[str]=None):
#     """
#     #### Plots images from the test dataloader along with predicted labels and probabilities.

#     ## Parameters:
#       `test_dataloader (torch.utils.data.DataLoader)`: Dataloader for the test dataset.
#       `model (torch.nn.Module)`: The trained model for making predictions.
#       `label_names (list)`: List of label names. Default is None.

#     ## Returns:
#       None

#     ## Example:
#       plot_images_with_predictions(test_dataloader, model, label_names=['bowel_injury', 'extravasation_injury', ...])
#     """
#     # Setting device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # Set model to evaluation mode
#     model.eval()
#     # Get one batch of test data
#     images, labels = next(iter(test_dataloader))
#     images = images.to(device)

#     # Forward pass
#     with torch.inference_mode():
#         outputs = model(images)
#         probabilities = torch.sigmoid(outputs)

#     # Convert probabilities and labels to numpy arrays
#     probabilities = probabilities.cpu().numpy()
#     labels = labels.numpy()

#     # Plot images with predicted labels and probabilities
#     fig, axes = plt.subplots(2, 5, figsize=(15, 8))
#     fig.suptitle('Images with Predicted Labels and Probabilities', fontsize=16)
#     axes = axes.flatten()

#     for i in range(10):
#         image = images[i].cpu().numpy().transpose((1, 2, 0))
#         image = np.clip(image, 0, 1)  # Clip image values to [0, 1] range
#         axes[i].imshow(image)
#         axes[i].axis('off')

#         # Get labels and probabilities
#         label_indices = np.arange(len(probabilities[i]))
#         probs = probabilities[i]

#         # Sort labels based on probabilities
#         sorted_indices = np.argsort(-probs)
#         sorted_probs = probs[sorted_indices]
#         sorted_labels = label_indices[sorted_indices]

#         # Get label names if provided
#         if label_names:
#             label_info = [f'{label_names[j]}: {sorted_probs[j]:.2f}' for j in sorted_labels]
#         else:
#             label_info = [f'Label {sorted_labels[j]}: {sorted_probs[j]:.2f}' for j in range(len(sorted_probs))]

#         # Set title with labels and probabilities
#         title = '\n'.join(label_info)
#         axes[i].set_title(title)

#     plt.tight_layout()
#     plt.show()

# def plot_roc_curves(test_dataloader, model, label_names=None):
#     """
#     #### Plots ROC curves for each label in the multi-label classification task.

#     ## Parameters:
#       `test_dataloader (torch.utils.data.DataLoader)`: Dataloader for the test dataset.
#       `model (torch.nn.Module)`: The trained model for evaluation.
#       `label_names (list)`: List of label names. Default is None.

#     ## Returns:
#        None

#     ## Example:
#         plot_roc_curves(test_dataloader, model, label_names=['Label 0', 'Label 1', ...])
#     """
#     # Setting device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # Set model to evaluation mode
#     model.eval()

#     true_labels = []
#     predicted_probabilities = []

#     with torch.inference_mode():
#         for images, labels in test_dataloader:
#             images = images.to(device)

#             outputs = model(images)
#             probabilities = torch.sigmoid(outputs).cpu().numpy()

#             true_labels.extend(labels.numpy())
#             predicted_probabilities.extend(probabilities)

#     # Convert true labels and predicted probabilities to numpy arrays
#     true_labels = np.array(true_labels)
#     predicted_probabilities = np.array(predicted_probabilities)

#     # Get label names if provided
#     target_names = label_names if label_names else [f'Label {i}' for i in range(true_labels.shape[1])]

#     # Compute and plot ROC curve for each label
#     plt.figure(figsize=(10, 7))
#     for label_idx in range(true_labels.shape[1]):
#         fpr, tpr, _ = roc_curve(true_labels[:, label_idx], predicted_probabilities[:, label_idx])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'{target_names[label_idx]} (AUC = {roc_auc:.2f})')

#     plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves for Multi-label Classification')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

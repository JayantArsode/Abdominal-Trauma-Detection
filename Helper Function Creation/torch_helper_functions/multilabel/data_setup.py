"""
Contains functionality for creating PyTorch DataLoader's for
image classification data and related utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import pandas as pd

def find_classes(label_columns: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    #### Identify classes from the given list of label columns and assign numerical labels.

    ## Parameters:
      `label_columns (List[str])`: List of class labels.

    ## Returns:
      `Tuple[List[str], Dict[str, List[int]]]`: A tuple containing a list of class names
          and a dictionary mapping class names to numerical labels.

    ## Exampel:
       label_columns = ["dog","cat"]
       classes,class_to_idx = find_classes(label_colums)
    """
    # 1. Get the class names by scanning the target directory
    classes = label_columns

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {label_columns}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {}
    for index, class_name in enumerate(label_columns):
        temp = [0] * len(label_columns)
        temp[index] = 1
        class_to_idx[class_name] = temp
    return classes, class_to_idx

class Images_From_DataFrame(Dataset):
    """
    #### Dataset class for loading images and labels from a DataFrame.

    ## Parameters:
      `df (pd.DataFrame)`: DataFrame containing image paths and labels.
      `image_path_column (str)`: Name of the column containing image paths.
      `label_columns (List[str])`: List of column names containing labels.
      `transform (Optional[transforms.Compose])`: Optional transforms to be applied to images.

    ## Returns:
      `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing the image tensor and its label tensor.

    ## Example:
      train_data=Images_From_DataFrame(df,'image_path',["dog","cat"],transform)
    """
    def __init__(self, df: pd.DataFrame, image_path_column: str, label_columns: List[str], transform:transforms.Compose=None,channels:int=1):
        self.paths = df[image_path_column].apply(Path).tolist()
        self.df = df
        self.label_columns = label_columns
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(label_columns)
        self.channels=channels

    def load_image(self, index: int):
        """Load image from the given index."""
        image_path = self.paths[index]
        if self.channels==3:
          return Image.open(image_path).convert('RGB')
        return Image.open(image_path)

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the image and its label at the given index."""
        img = self.load_image(index)
        label = self.df[self.label_columns].iloc[index].values.tolist()
        if self.transform:
            label=torch.tensor(label, dtype=torch.float32)
            y_bowel, y_extra, y_kidney, y_liver, y_spleen=label[0:1],label[1:2],label[2:5],label[5:8],label[8:]
            return self.transform(img),y_bowel, y_extra, y_kidney, y_liver, y_spleen
        else:
            label=torch.tensor(label, dtype=torch.float32)
            y_bowel, y_extra, y_kidney, y_liver, y_spleen=label[0:1],label[1:2],label[2:5],label[5:8],label[8:]
            return img, y_bowel, y_extra, y_kidney, y_liver, y_spleen

    def __repr__(self):
        """Return a string representation of the dataset."""
        return (f"Dataset Images_From_DataFrame\n\
                  Number of datapoints: {len(self.paths)}\n\
                  StandardTransform\nTransform: {self.transform}")

def create_dataloaders_from_dataframe(
    df: pd.DataFrame,
    image_path_column: str,
    label_columns: List[str],
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int=32,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = os.cpu_count(),
    channels:int=1,
    collate_fn=None) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    #### Create DataLoader instances for training, validation, and testing from a DataFrame.

    ## Parameters
      `df (pd.DataFrame)`: DataFrame containing image paths and labels.
      `image_path_column (str)`: Name of the column containing image paths.
      `label_columns (List[str])`: List of column names containing labels.
      `transform (transforms.Compose)`: Transforms to be applied to images.
      `batch_size (int)`: Batch size for DataLoader.
      `validation_split (float)`: Percentage of data to use for validation. Default is 0.2.
      `test_split (float)`: Percentage of data to use for testing. Default is 0.1.
      `num_workers (int)`: Number of subprocesses to use for data loading. Default is CPU count.

    ## Returns:
        `Tuple[DataLoader, DataLoader, DataLoader, List[str]]`: A tuple containing
          DataLoaders for training, validation, and testing, and a list of class names.

    ## Example:
       train_dataloader, val_dataloader, test_dataloader, classes=create_dataloaders_from_dataframe(df,
                                                                                                    'image_path',
                                                                                                    ["dog","cat"],
                                                                                                    transform.32,
                                                                                                    0.2,
                                                                                                    0.1,
                                                                                                    0)
    """
    patient_ids = df['patient_id'].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)

    # Create masks to filter the data
    train_mask = df['patient_id'].isin(train_ids)
    test_mask = df['patient_id'].isin(test_ids)

    # Split the data into train, test, and validation sets
    train_data = df[train_mask]
    test_data = df[test_mask]
    
    test_data,val_data=train_test_split(test_data,test_size=0.5,random_state=42)

    print(f"Same Patients of Training Data In Validation Data are: {val_data['patient_id'].isin(train_data['patient_id']).astype(int).sum()}")
    print(f"Same Patients of Training Data In Testing Data are: {test_data['patient_id'].isin(train_data['patient_id']).astype(int).sum()}")
    # Create datasets and dataloaders for train, validation, and test data
    train_dataset = Images_From_DataFrame(df=train_data,
                                           image_path_column=image_path_column,
                                           label_columns=label_columns,
                                           transform=train_transform,
                                           channels=channels)
    val_dataset = Images_From_DataFrame(df=val_data,
                                         image_path_column=image_path_column,
                                         label_columns=label_columns,
                                         transform=val_transform,
                                         channels=channels)
    test_dataset = Images_From_DataFrame(df=test_data,
                                          image_path_column=image_path_column,
                                          label_columns=label_columns,
                                          transform=test_transform,
                                          channels=channels)
    # Get classes
    classes = train_dataset.classes
    # Create DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=True,
                                   collate_fn=collate_fn)  # Shuffle training data
    val_dataloader = DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)  # No need to shuffle validation data
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False)  # No need to shuffle test data
    print(f"Each 1 instance in dataloader={batch_size} data points.")
    print(f"Train DataLoader contains: {len(train_dataloader)} instance = {train_data.shape[0]} data points.")
    print(f"Validation DataLoader contains: {len(val_dataloader)} instance = {val_data.shape[0]} data points.")
    print(f"Test DataLoader contains: {len(test_dataloader)} instance = {test_data.shape[0]} data points.")
    return train_dataloader, val_dataloader, test_dataloader, classes

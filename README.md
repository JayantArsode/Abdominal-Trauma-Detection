# Abdominal Trauma Detection

Abdominal Trauma Detection is a deep learning project for identifying trauma-related findings in abdominal CT scan images. This repository contains the notebook workflow, helper modules, cleaned metadata files, sample test data, and experiment notebooks used for the project.

The project is based on the **RSNA 2023 Abdominal Trauma Detection** task and treats the problem as a **multi-task classification pipeline**.

## What The Project Predicts

The models in this repository predict labels for:

- Bowel: healthy or injury
- Extravasation: healthy or injury
- Kidney: healthy, low, or high
- Liver: healthy, low, or high
- Spleen: healthy, low, or high

Training setup:

- binary classification for bowel and extravasation
- multiclass classification for kidney, liver, and spleen

## What Is Included In This Repository

This repository already includes:

- notebook workflow for data cleaning, experiments, and inference
- reusable PyTorch helper modules
- cleaned metadata files in `Data Cleaning And Analysis/Valid Images Data/`
- sample inference data in `test.csv` and `test_images/`

This repository does **not** include:

- the full original Kaggle training dataset
- trained model checkpoint files such as `.pth` or `.onnx`

That means a new user can understand the workflow, inspect the notebooks, review the cleaned data files, and prepare the code locally, but full training and full inference with trained models require extra data and model weights.

## Models In The Project

- ATD Simple
- ATD EfficientNetB1
- ATD EfficientNetB1 CutMixUp
- ATD ResNet50
- ATD ResNet50 CutMixUp

Main design choices:

- custom CNN baseline for grayscale CT images
- transfer learning with EfficientNetB1 and ResNet50
- separate prediction heads for different tasks
- class weights for imbalanced targets
- CutMix augmentation in advanced experiments

## Repository Contents

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- test.csv
|-- test_images/
|-- Data Cleaning And Analysis/
|   |-- Data_Cleaning _Analysis.ipynb
|   `-- Valid Images Data/
|       |-- patients_with_abnormalities.csv
|       |-- train_images_valid.csv
|       `-- valid_image_paths.npy
|-- Experiments ATD(Abdominal Trauma Detection)/
|   |-- Experiment 1 (ATD_Simple).ipynb
|   |-- Experiment 2 (ATD_EfficientNetB1_Model).ipynb
|   |-- Experiment 3 (ATD_EfficientNetB1_CutMixUp_Model).ipynb
|   |-- Experiment 4 (ATD_ResNet50_Model).ipynb
|   |-- Experiment 5 (ATD_ResNet50_CutMixUp_Model).ipynb
|   `-- Inference of Models Trained.ipynb
`-- Helper Function Creation/
    |-- helper_functions_creation.ipynb
    `-- torch_helper_functions/
        |-- utils.py
        |-- plotting_utils.py
        `-- multilabel/
            |-- data_cleaner.py
            |-- data_setup.py
            `-- train_engine.py
```

## What A New User Should Do First

If you are opening this repository for the first time, use this order:

1. Install dependencies.
2. Open the repository in Jupyter.
3. Read the data cleaning notebook to understand the dataset format.
4. Review the helper modules to understand the training code.
5. Open the experiment notebooks to see how each model was trained.
6. Open the inference notebook only after you have trained weights or your own checkpoint files.

## Local Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Notes:

- Python 3.10 or newer is recommended
- if you want GPU support, install the correct PyTorch build for your CUDA version from the official PyTorch site
- `requirements.txt` is enough for a basic local notebook setup

### 3. Start Jupyter

```bash
jupyter notebook
```

Run Jupyter from the repository root so file paths work more predictably.

## How To Run The Project Locally

There are three realistic ways to use this repository.

### Option 1: Understand the project structure

Use this if you want to study the workflow and code.

Open these files in order:

1. `Data Cleaning And Analysis/Data_Cleaning _Analysis.ipynb`
2. `Helper Function Creation/helper_functions_creation.ipynb`
3. `Helper Function Creation/torch_helper_functions/`
4. `Experiments ATD(Abdominal Trauma Detection)/`

### Option 2: Reuse the helper code in your own training setup

Use this if you want to train with your own local copy of the RSNA dataset.

Typical order:

1. prepare your dataset paths locally
2. use `multilabel/data_cleaner.py` and `multilabel/data_setup.py`
3. run one of the experiment notebooks
4. train and save model checkpoints
5. run inference with your saved weights

### Option 3: Run inference on the included sample data

Use this if you already have trained model weights.

Steps:

1. open `Experiments ATD(Abdominal Trauma Detection)/Inference of Models Trained.ipynb`
2. replace any old `/content/...` paths with local paths
3. load your trained `.pth` weights
4. use `test.csv` and `test_images/`
5. run the prediction and evaluation cells

Important:

- this repository includes sample test images and metadata
- it does **not** include trained weights, so inference will not fully run until you provide or train model checkpoints

## Notebook Guide

### `Data Cleaning And Analysis/Data_Cleaning _Analysis.ipynb`

Use this notebook to:

- clean dataset records
- validate image paths
- inspect class distributions and abnormalities
- generate the cleaned files used in later experiments

Main outputs already present in the repository:

- `train_images_valid.csv`
- `patients_with_abnormalities.csv`
- `valid_image_paths.npy`

### `Helper Function Creation/helper_functions_creation.ipynb`

This notebook was used to generate the helper files inside `torch_helper_functions/`.

For a new user:

- you usually do **not** need to run this notebook first
- the generated helper files are already present in the repository
- you only need this notebook if you want to understand or recreate how the helper package was built

### Experiment notebooks

Located in `Experiments ATD(Abdominal Trauma Detection)/`.

- `Experiment 1 (ATD_Simple).ipynb`: baseline custom CNN
- `Experiment 2 (ATD_EfficientNetB1_Model).ipynb`: EfficientNetB1 transfer learning
- `Experiment 3 (ATD_EfficientNetB1_CutMixUp_Model).ipynb`: EfficientNetB1 with CutMix
- `Experiment 4 (ATD_ResNet50_Model).ipynb`: ResNet50 transfer learning
- `Experiment 5 (ATD_ResNet50_CutMixUp_Model).ipynb`: ResNet50 with CutMix
- `Inference of Models Trained.ipynb`: inference and evaluation

## Helper Modules

Reusable helper code lives in `Helper Function Creation/torch_helper_functions/`.

### `utils.py`

Used for:

- saving trained PyTorch models
- computing class weights
- evaluating trained models
- printing training summaries

### `plotting_utils.py`

Used for:

- plotting training and validation curves
- generating confusion matrices

### `multilabel/data_cleaner.py`

Used for:

- preprocessing dataframe image paths
- validating image files
- saving valid image-path arrays

### `multilabel/data_setup.py`

Used for:

- creating dataset objects from pandas dataframes
- splitting data by patient ID
- creating train, validation, and test dataloaders

### `multilabel/train_engine.py`

Used for:

- training loops
- validation loops
- optional CutMix training
- WandB logging
- checkpoint saving
- ONNX export

## Running The Included Sample Data

This repository includes:

- `test.csv`
- `test_images/`

`test.csv` contains rows like:

- patient ID
- injury labels
- series ID
- image path

The image paths currently use Colab-style prefixes such as `/content/test_images/...`.

Before using them locally, convert them to local paths. Example:

```python
df["image_path"] = df["image_path"].str.replace(
    "/content/test_images",
    "test_images",
    regex=False,
)
```

After that, the rows should point to images inside the local `test_images/` folder.

## What You Need For Full Training

To reproduce full training locally, you will need:

- access to the full RSNA dataset or a prepared local copy
- correct local paths inside the notebooks
- enough storage for CT scan images
- enough compute to train EfficientNetB1 and ResNet50 models
- your own checkpoint output directory

Without the training dataset, a new user can still inspect the workflow but cannot fully reproduce training.

## Common Local Issues

### 1. Notebook paths still point to Colab

Some notebooks were originally executed in Colab. Replace `/content/...` paths with local repository paths before running cells.

### 2. Helper imports fail

If a notebook cannot find `torch_helper_functions`, add the helper directory manually:

```python
import sys
sys.path.append(r"Helper Function Creation")
```

### 3. WandB prompts during training

The training code uses `wandb`. Configure or disable WandB before running training notebooks if you do not want online logging.

### 4. Inference notebook cannot find model weights

This is expected if you have not trained the models yet. The repository does not currently include saved `.pth` or `.onnx` checkpoints.

## Results Summary

Overall findings from the project:

- the baseline model is weaker on some injury categories
- transfer-learning models perform better than the baseline
- CutMix-based experiments improve performance further
- performance on fully unseen external data remains a challenge

## Credits

Contributor:

- Jayant Arsode

Acknowledgement:

- Hemant Tekade helped in the project work

## License

See `LICENSE` for license information.

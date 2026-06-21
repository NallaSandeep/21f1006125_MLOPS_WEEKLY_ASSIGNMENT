# Setting Up the IRIS ML Pipeline on Vertex AI

## Overview

Build an end-to-end machine learning pipeline for the IRIS classifier on Google Cloud's Vertex AI platform, using Google Cloud Storage for data and artifact management.

### Objectives

* Set up and navigate the Google Cloud Platform and Vertex AI Workbench.
* Use Google Cloud Storage for ML data and artifact management.
* Build and execute an end-to-end IRIS classification pipeline on cloud infrastructure.
* Organize model artifacts by execution timestamp for traceability.
* Separate training and inference into distinct, reproducible scripts.

### Data
The Iris dataset (iris.csv), containing 150 samples, is stored in the data folder and is used for training and validating the machine learning model. The dataset includes four input features—sepal_length, sepal_width, petal_length, and petal_width—along with the target species label (Setosa, Versicolor, or Virginica) used for classification.

### Dataset splits
IRIS dataset is divided into training (60%) and testing (40%) subsets to enable model training and performance evaluation on unseen data. Stratified sampling is used to preserve the original distribution of iris species across both datasets, ensuring a balanced and representative split for reliable model assessment.

### Source code (`week_1_graded_assignment.ipynb`)

This Jupyter Notebook contains the end-to-end implementation for building an Iris species classification model, including data loading, model training, artifact generation, and inference. The trained model is saved as a versioned artifact and uploaded to a Google Cloud Storage (GCS) bucket, where each execution creates a separate timestamp-based directory to maintain model history. The notebook also demonstrates how to retrieve a trained model from GCS, load it for inference, and perform predictions on new input data, enabling model versioning, traceability, and reuse of previously trained models.

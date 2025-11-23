# End_to_End_Interactive_ML_Pipeline_for_DGE_-Dummy_Dataset-
End_to_End_Interactive_ML_Pipeline_for_DGE_[Dummy_Dataset]
# End-to-End Interactive ML Pipeline for Differential Gene Expression (DGE)

## Overview
This repository contains a robust **Machine Learning pipeline** designed to analyze Differential Gene Expression (DGE) data. The notebook takes raw gene expression metrics (Log Fold Change, P-values, Base Mean) and trains multiple classification models to distinguish between **Upregulated** and **Downregulated** genes.

A key feature of this pipeline is its **interactivity** and **robustness**—if no dataset is provided, it automatically generates a synthetic biological dataset with embedded signals to demonstrate the workflow.

## Key Features
* **Automatic Data Generation:** Automatically generates synthetic gene expression data (Gaussian distributions for LogFC, Exponential for BaseMean) if an input CSV is not found.
* **Interactive EDA:** Uses **Plotly** to create interactive Volcano Plots with hover capabilities.
* **Multi-Model Training:** Trains and evaluates four distinct classifiers:
    * Logistic Regression
    * Support Vector Machine (RBF Kernel)
    * Random Forest
    * XGBoost (if available)
* **Advanced Evaluation:** Includes interactive ROC Curves and Model Comparison bar charts.
* **Dimensionality Reduction:** Performs **PCA (2D and 3D)** to visualize sample clustering.
* **Feature Selection:** Utilizes **LASSO (L1 Regularization)** to identify biologically significant features.

## Prerequisites
To run this notebook, you need Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn plotly xgboost

Workflow Steps

The pipeline executes the following steps in order:

    Imports & Setup: Loads necessary libraries and checks for GPU/Acceleration libraries.

    Data Loading/Generation:

        Attempts to load gene_expression_data.csv.

        Fallback: Generates synthetic data (N=200) with artificially introduced upregulated/downregulated signals.

    Exploratory Data Analysis (EDA):

        Generates an Interactive Volcano Plot (log2​FC vs −log10​Pvalue).

        Classifies genes as Upregulated, Downregulated, or Not Significant based on thresholds (p<0.05, ∣logFC∣>1).

    Preprocessing:

        Target encoding: 1 (Upregulated) vs 0 (Downregulated).

        Stratified Train-Test split (80/20).

        Standard Scaling (StandardScaler) of features.

    Model Training:

        Fits Logistic Regression, SVM, Random Forest, and XGBoost.

        Computes Accuracy and Cross-Validation scores.

    Evaluation:

        Plots interactive ROC Curves (AUC calculation).

        Compares models via grouped bar charts.

    Dimensionality Reduction:

        Principal Component Analysis (PCA) to project data into 2D and 3D space.

        Visualizes how well the classes separate based on expression profiles.

    Feature Selection:

        Applies LASSO Logistic Regression to find non-zero coefficients.

        Visualizes expression distributions of the selected top features using box plots.

Usage

    Clone this repository.

    (Optional) Place your own dataset named gene_expression_data.csv in the root directory. It requires columns: name, logFC, pvalue, baseMean.

    Open the notebook:
    Bash
jupyter notebook "End_to_End_Interactive_ML_Pipeline_for_DGE_[Dummy_Dataset].ipynb"
Run all cells to execute the pipeline.

Outputs

The notebook produces interactive HTML-based plots (via Plotly) directly within the interface:

    Volcano Plot

    Model Accuracy Comparison

    ROC Curves

    PCA Scatter Plots (2D & 3D)

    Feature Importance Bar Chart & Box Plots

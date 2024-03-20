# Class-specific-PCA-with-synthetic-data

This is a code repository for "Advancing Frontline Clinical Diagnosis: Integrating
Class-Specific PCA and Synthetic Spectroscopy Data
for Improved Pancreatic Cancer Detection"

This repository includes experiments with various analysis pipelines including class specific PCA, LDA, and SVM techniques using spectrum samples synthesised from individual mean spectrum obtained from different material ATR spectra samples to simulate real patient spectroscopy data. 
To generate synthetic datasets, run Sample_Synthesis.ipynb and configurate the desired synthetic dataset output.
To build classifiers using the proposed pipeline, run Synth-Experiment1-Groups-PCs.ipynb which includes an end-to-end analysis process for this experimentation. Results can be observed in notebook outputs (cross-validated accuracy etc) and/or figure outputs according to different configurations.

The code only contains the synthetic data experiment part due to privacy policies regarding patient information.

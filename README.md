# OralCare AI --- Oral Disease Classification

An image-classification project that uses deep learning to classify oral
disease images into six categories. The project compares a custom
Convolutional Neural Network (CNN) with transfer-learning models and
provides a Gradio interface for trying the selected model.

> **Medical disclaimer:** This project is for educational and research
> purposes only. Predictions are not a diagnosis and must not replace
> assessment by a qualified dental professional.

## Classes

The model predicts one of the following classes:

1.  Calculus
2.  Caries
3.  Gingivitis
4.  Hypodontia
5.  Mouth Ulcer
6.  Tooth Discoloration

## Project Structure

  ---------------------------------------------------------------------------------------------------
  Path / File                                                     Description
  --------------------------------------------------------------- -----------------------------------
  `evaluation_results/`                                           Stores CSV files containing
                                                                  per-model evaluation reports and a
                                                                  comparison of model performance.

  `evaluation_results/custom_cnn_classification_report.csv`       Classification metrics for the
                                                                  custom CNN, typically including
                                                                  precision, recall, F1-score, and
                                                                  support for each class.

  `evaluation_results/efficientnetb0_classification_report.csv`   Classification report for the
                                                                  selected EfficientNetB0 model.

  `evaluation_results/mobilenetv2_classification_report.csv`      Classification report for
                                                                  MobileNetV2.

  `evaluation_results/test_model_comparison.csv`                  Side-by-side comparison of the
                                                                  evaluated models on the test set.

  `models/`                                                       Contains saved Keras model files.

  `models/custom_cnn_best.keras`                                  Saved custom CNN model.

  `models/efficientnetb0_fixed_best.keras`                        Corrected EfficientNetB0 model used
                                                                  by the application.

  `models/mobilenetv2_best.keras`                                 Saved MobileNetV2 model.

  `process/`                                                      Contains the notebook used for the
                                                                  project workflow.

  `process/Oral_Disease_Classification.ipynb`                     Jupyter notebook for data
                                                                  preparation, model training,
                                                                  evaluation, and related
                                                                  experiments.

  `README.md`                                                     Project overview, setup
                                                                  instructions, file descriptions,
                                                                  and known challenges.

  `app.py`                                                        Gradio application that loads the
                                                                  selected model and predicts a class
                                                                  for an uploaded image.

  `requirements.txt`                                              Python dependencies required to run
                                                                  the application.
  ---------------------------------------------------------------------------------------------------

## Models

Three model approaches were evaluated:

-   **Custom CNN** --- a convolutional neural network trained from
    scratch.
-   **MobileNetV2** --- a pretrained architecture adapted for the six
    target classes.
-   **EfficientNetB0** --- a pretrained architecture adapted for the six
    target classes.

The application is configured to load
`models/efficientnetb0_fixed_best.keras`. The CSV files in
`evaluation_results/` document the model comparison and classification
reports. Refer to those files for the exact reported metrics; results
can vary depending on the dataset split and evaluation setup.

## Run the Application Locally

### 1. Requirements

-   Python 3.10 or another Python version supported by the installed
    TensorFlow release.
-   The files listed above, including
    `models/efficientnetb0_fixed_best.keras`.

### 2. Install dependencies

``` bash
python -m venv .venv
```

Activate the environment:

**Windows**

``` bash
.venv\\Scripts\\activate
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Install packages:

``` bash
pip install -r requirements.txt
```

### 3. Launch the app

``` bash
python app.py
```

Open the local Gradio URL printed in the terminal, upload an image, and
review the predicted class and scores.

> If `app.py` uses `demo.launch(share=True)`, Gradio may generate a
> temporary public link. That link is not permanent and should not be
> treated as a production deployment.

## Challenges and Lessons Learned

  -----------------------------------------------------------------------------------
  Challenge               Why it mattered         Approach / lesson
  ----------------------- ----------------------- -----------------------------------
  Organizing image data   Inconsistent folders or Keep class names and label mappings
  across six classes      labels can lead to      consistent throughout preparation,
                          incorrect training and  training, evaluation, and
                          evaluation.             deployment.

  Preparing and checking  Image dimensions, color Convert uploaded images to RGB,
  image data              channels, and           resize them to `224 × 224`, and
                          preprocessing must      apply the same scaling used during
                          match the model's       training.
                          expected input.         

  Comparing different     Accuracy alone may hide Compare the custom CNN,
  model architectures     poor performance on     MobileNetV2, and EfficientNetB0
                          individual classes.     using the test comparison CSV and
                                                  per-class classification reports.

  Uneven performance      A model can perform     Review per-class precision, recall,
  across classes          well overall while      and F1-score, not just overall
                          missing examples from a accuracy. The reported
                          particular class.       EfficientNetB0 evaluation showed
                                                  weaker recall for the Calculus
                                                  class, so this remains an area for
                                                  improvement.

  Choosing and loading    Multiple checkpoints    Configure the application to load
  the correct model file  can make it easy to     `efficientnetb0_fixed_best.keras`
                          load an outdated model  and keep the filename and path
                          by mistake.             consistent with the repository
                                                  structure.

  Deploying the model in  The application needs   Use Gradio for a simple
  an interactive          to load the saved model image-upload and prediction
  interface               and accept              interface, and test the app after
                          user-uploaded images    confirming that the model path and
                          reliably.               dependencies are correct.

  Sharing a demo from a   A notebook-hosted       Treat shared Gradio links as
  notebook environment    public link may expire  temporary demos; use a persistent
                          when the session ends.  hosting service if long-term
                                                  availability is required.

  Distinguishing model    Image predictions can   Include a clear disclaimer and
  output from medical     be wrong and are not    present the application as an
  advice                  clinical diagnoses.     educational prototype rather than a
                                                  diagnostic tool.
  -----------------------------------------------------------------------------------

## Limitations and Future Improvements

-   Improve performance on classes with lower recall, especially
    Calculus.
-   Inspect the confusion matrix and per-class metrics to identify
    commonly confused classes.
-   Evaluate on a larger, independently collected dataset and check for
    data leakage between splits.
-   Document the exact training configuration, dataset source, split
    strategy, and reproducibility settings.
-   Consider a persistent deployment platform if a stable public demo is
    needed.
-   Do not use the model to make clinical decisions without appropriate
    validation and qualified professional oversight.

## Repository Notes

-   The application expects the selected model at
    `models/efficientnetb0_fixed_best.keras`.
-   The training notebook is located at
    `process/Oral_Disease_Classification.ipynb`.
-   Evaluation CSV files are located in `evaluation_results/`.
-   This repository structure does not include the original image
    dataset. Dataset access and usage rights should be checked
    separately before redistribution.

## Disclaimer

OralCare AI is an educational deep-learning project. Its predictions may
be inaccurate and are not a substitute for professional dental
examination, diagnosis, or treatment.

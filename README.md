# OralCare AI
## AI-Powered Oral Disease Image Classification

**Developed by Hassan Saleh**

OralCare AI is an AI-powered image classification application designed
to classify oral images into six categories using deep learning.

## Features
- Upload oral images for classification.
- Display the predicted category.
- View model scores for all six categories.
- Medical-inspired user interface.

## Supported Categories
1. Calculus
2. Caries
3. Gingivitis
4. Hypodontia
5. Mouth Ulcer
6. Tooth Discoloration

## Models Evaluated
- Custom CNN
- MobileNetV2
- EfficientNetB0

## Test Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Custom CNN | 92.44% | 88.43% |
| MobileNetV2 | 89.88% | 87.42% |
| EfficientNetB0 | 92.31% | 88.68% |

The test set contains 1,521 images. EfficientNetB0 is the current
deployment candidate based on its macro F1-score.

## Technology Stack
- Python
- TensorFlow / Keras
- Gradio
- NumPy
- Pillow

## Project Structure

    Sprint2_Oral_Disease_Classification/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── models/
    │   └── efficientnetb0_fixed_best.keras
    └── evaluation_results/

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    python app.py

Make sure the trained model exists at:

    models/efficientnetb0_fixed_best.keras

## Medical Disclaimer

OralCare AI is intended for informational purposes and portfolio
demonstration. Its predictions are not a medical diagnosis and are
not a substitute for professional dental examination, diagnosis,
or treatment. Consult a qualified dental professional for medical advice.

## Author

Hassan Saleh

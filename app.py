
import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "efficientnetb0_fixed_best.keras"
)

CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth_Ulcer",
    "Tooth_Discoloration"
]

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)


def predict_oral_disease(image):
    if image is None:
        return None, {}, "Please upload an image first."

    image = image.convert("RGB").resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    results = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
    score = float(np.max(predictions)) * 100

    summary = (
        f"### Predicted Category: {predicted_class}\n\n"
        f"**Model Score:** {score:.2f}%\n\n"
        "This score represents the model output and does not "
        "confirm the presence of a medical condition."
    )

    return image, results, summary


medical_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate"
)

custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    background: #f4f8fb !important;
}

#hero {
    background: linear-gradient(135deg, #103b5c, #087e8b);
    padding: 32px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 22px;
}

#hero h1 {
    font-size: 34px;
    color: white !important;
}

#hero p {
    color: #e0f7fa !important;
    font-size: 16px;
}

.section-card {
    background: white;
    border: 1px solid #dce8ef;
    border-radius: 16px;
    padding: 18px;
}

#analyze-btn {
    background: #087e8b;
    color: white;
    border-radius: 10px;
    font-weight: 700;
}

#medical-note {
    background: #fff8e7;
    border-left: 5px solid #e8ad35;
    padding: 15px;
    border-radius: 10px;
    color: #000000 !important;
    margin-top: 20px;
}

#medical-note strong,
.gradio-container h3 {
    color: #000000 !important;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 20px;
}
"""


with gr.Blocks(
    title="OralCare AI | Oral Image Classification"
) as demo:

    gr.HTML("""
    <div id="hero">
        <h1>🦷 OralCare AI</h1>
        <p>AI-Powered Oral Disease Image Classification</p>
        <p>
            Intelligent image classification across six oral
            condition categories.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📤 Upload Oral Image")

            input_image = gr.Image(
                type="pil",
                label="Oral Image",
                height=330
            )

            gr.Markdown(
                "Upload a clear image relevant to one of the "
                "six categories supported by the model."
            )

            analyze_btn = gr.Button(
                "Analyze Image",
                elem_id="analyze-btn",
                variant="primary"
            )

            clear_btn = gr.ClearButton(value="Clear")

        with gr.Column():
            gr.Markdown("## 🔬 Analysis Results")

            output_image = gr.Image(
                label="Submitted Image",
                height=250,
                interactive=False
            )

            output_summary = gr.Markdown(
                "Upload an image and select **Analyze Image** "
                "to view the model output."
            )

    with gr.Row():
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 📊 Class Scores")

            output_scores = gr.Label(
                num_top_classes=6,
                label="Model Output"
            )

    analyze_btn.click(
        fn=predict_oral_disease,
        inputs=input_image,
        outputs=[output_image, output_scores, output_summary]
    )

    clear_btn.add(
        [input_image, output_image, output_scores, output_summary]
    )

    gr.HTML("""
    <div id="medical-note">
        <strong>⚕️ Important Medical Notice</strong><br>
        OralCare AI uses artificial intelligence to classify
        oral images into six categories. Its predictions are
        intended for informational purposes only and are not
        a substitute for professional dental diagnosis or
        medical advice. Consult a qualified dental professional
        for accurate assessment and treatment recommendations.
    </div>

    <div class="footer">
        OralCare AI · Developed by Hassan Saleh
    </div>
    """)


if __name__ == "__main__":
    demo.launch(share=True)

from flask import Flask, request, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import io
import base64
import torch
import openai
import os

app = Flask(__name__)

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast = True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Include your OpenAI API key
openai.api_key = ""

def generate_makeup_routine(caption, selected_products):
    product_list = ", ".join(selected_products) if selected_products else "no makeup products"
    prompt = (
        f"Create a detailed, step-by-step makeup tutorial based on this image caption: '{caption}'. "
        f"Only use the following available makeup products: {product_list}. "
        f"Do not include any items that are not on the list. Be creative but stick strictly to what's available."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")

    # Get selected makeup products from the form
    selected_products = request.form.getlist('products')

    # Generate image caption
    inputs = processor(image, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Generate makeup routine from caption and selected products
    raw_routine = generate_makeup_routine(caption, selected_products)
    steps = [step.strip() for step in raw_routine.split('\n') if step.strip()]

    # Adjust and display image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return render_template('display.html', image_data=encoded_image, caption=caption, steps=steps)


if __name__ == '__main__':
    app.run(debug=True)

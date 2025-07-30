from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image

model_name= 'anuashok/ocr-captcha-v3' 
#'microsoft/trocr-base-printed'
# Load model and processor
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Load image
# image = Image.open('/content/captcha_images/captcha_20250729_224520.png').convert("RGB")
# Load and preprocess image for display
image = Image.open('/Users/prashanth/Desktop/captcha_images/captcha_20250730_112712.png').convert("RGBA")
# Create white background
background = Image.new("RGBA", image.size, (255, 255, 255))
combined = Image.alpha_composite(background, image).convert("RGB")

# Prepare image
pixel_values = processor(combined, return_tensors="pt").pixel_values

# Generate text
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

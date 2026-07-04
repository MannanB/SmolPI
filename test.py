from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

# Load images
image1 = load_image("https://huggingface.co/spaces/HuggingFaceTB/SmolVLM/resolve/main/example_images/rococo.jpg")
image2 = load_image("https://huggingface.co/spaces/HuggingFaceTB/SmolVLM/resolve/main/example_images/rococo_1.jpg")

batch_messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Can you describe the two images?"}
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Compare these two images."}
            ],
        }
    ],
]

prompts = [
    processor.apply_chat_template(m, add_generation_prompt=True)
    for m in batch_messages
]

inputs = processor(
    text=prompts,
    images=[
        [image1, image2],   # images for sample 1
        [image1, image2],   # images for sample 2
    ],
    padding=True,
    return_tensors="pt",
)

print(inputs)
print(type(inputs))
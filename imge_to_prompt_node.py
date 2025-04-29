from io import BytesIO
from google import genai
from google.genai import types, Client
import numpy as np
from PIL import Image

from defaults import DEFAULT_PROMPT


class GeminiImageToPrompt:
    def __init__(self):
        self.output_dir = "output"
        self.type = "output"
        self._client: Client|None = None
        self._current_api_key = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["gemini-1.5-pro-latest", "gemini-2.0-flash-exp"],),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True
                }),
            },
            "optional": {
                 "prompt_template": ("STRING", {
                    "default": DEFAULT_PROMPT,
                    "multiline": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process_image"
    CATEGORY = "Gemini/Image"

    def get_client(self, api_key):
        """Get or create Gemini client"""
        if self._client and self._current_api_key != api_key:
            self._client = None

        if not self._client:
            self._client = genai.Client(api_key=api_key)
            self._current_api_key = api_key

        return self._client
    
    @staticmethod
    def convert_image_to_bytes(image, format="PNG"):
        # Convert PyTorch tensor to byte content
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        if image.shape[0] == 3:  # If image is in CHW format
            image = np.transpose(image, (1, 2, 0))
        pil_image = Image.fromarray(image)

        # Convert PIL Image to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue()

    @staticmethod
    def describe_image(prompt_template, image_bytes, client, model):
        """Process image using OpenAI API format"""
        try:
            image = types.Part.from_bytes(
                data=image_bytes, mime_type="image/jpeg"
            )
            response = client.models.generate_content(
                model=model,
                contents=[prompt_template, image],
            )
            return response.text
        except Exception as e:
            print(f"API error: {str(e)}")
            return f"Error: API request failed - {str(e)}"

    def process_image(self, image, model, api_url, api_key, prompt_template):
        try:
            # Convert the first image in bytes
            if len(image.shape) == 4:
                image = image[0]
            image_bytes = self.convert_image_to_bytes(image)

            # Process using Gemini api
            client = self.get_client(api_key)
            description = self.describe_image(prompt_template, image_bytes, client, model)

            return (description,)

        except Exception as e:
            print(f"Error in image description: {str(e)}")
            return (f"Error: Failed to generate image description. {str(e)}",)

    def __del__(self):
        """Cleanup"""
        pass

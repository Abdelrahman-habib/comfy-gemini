from .imge_to_prompt_node import GeminiImageToPrompt
from .enhance_prompt import GeminiEnhancePrompt # Add this import

NODE_CLASS_MAPPINGS = {
    "GeminiImageToPrompt": GeminiImageToPrompt,
    "GeminiEnhancePrompt": GeminiEnhancePrompt # Add this mapping
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageToPrompt": "Gemini Image To Prompt",
    "GeminiEnhancePrompt": "Gemini Enhance Prompt" # Add this display name
}

__version__ = "1.0.0"

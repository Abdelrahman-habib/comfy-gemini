import os
from google import genai
from google.genai import Client, types
from .defaults import ENHANCE_PROMPT_SYSTEM_PROMPT

class GeminiEnhancePrompt:
    def __init__(self):
        self._client: Client | None = None
        self._current_api_key = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["gemini-1.5-pro-latest", "gemini-2.0-flash-exp"],),
            },
            "optional": {
                 "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get('GEMINI_API_KEY', ''), # Check env var for default
                    "password": True
                }),
                 "system_prompt": ("STRING", {"multiline": True, "default": ENHANCE_PROMPT_SYSTEM_PROMPT}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "Gemini/Text"

    def get_client(self, api_key=None):
        """Get or create Gemini client, checking input and environment variable."""

        key_to_use = api_key if api_key else os.environ.get('GEMINI_API_KEY')

        if not key_to_use:
             # Handle missing API key after checking both sources
            raise ValueError("API key is required. Provide it in the node input or set the GEMINI_API_KEY environment variable.")

        # Check if client needs reset (key changed)
        if self._client and self._current_api_key != key_to_use:
            self._client = None

        # Create client if needed
        if not self._client:
            self._client = genai.Client(api_key=key_to_use)
            self._current_api_key = key_to_use

        return self._client

    def enhance(self, prompt, model, api_key=None, system_prompt=ENHANCE_PROMPT_SYSTEM_PROMPT): # api_key is now optional here too
        try:

            client = self.get_client(api_key)

            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),
                contents=[prompt],
            )

            if response.text and isinstance(response.text, str):
                enhanced_prompt = response.text.strip()
            else:
                print("Warning: Gemini API did not return valid text. Returning original prompt.")
                enhanced_prompt = prompt # Fallback to original prompt

            return (enhanced_prompt,)
        except Exception as e:
            print(f"Error calling Gemini API in Enhance Prompt: {e}")
            # Return the original prompt if enhancement fails
            return (prompt,)

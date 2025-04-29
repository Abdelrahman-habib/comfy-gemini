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
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "password": True
                }),
                "model": (["gemini-1.5-pro-latest", "gemini-2.0-flash-exp"],), # Renamed to model
            },
            "optional": {
                 "system_prompt": ("STRING", {"multiline": True, "default": ENHANCE_PROMPT_SYSTEM_PROMPT}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "Gemini/Text"

    def get_client(self, api_key):
        """Get or create Gemini client"""
        if not api_key:
             # Handle missing API key early
            raise ValueError("API key is required for Gemini Enhance Prompt node")

        if self._client and self._current_api_key != api_key:
            self._client = None

        if not self._client:
            self._client = genai.Client(api_key=api_key)
            self._current_api_key = api_key

        return self._client

    def enhance(self, prompt, api_key, model, system_prompt=ENHANCE_PROMPT_SYSTEM_PROMPT):
        try:
            client = self.get_client(api_key)

            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),
                contents=[prompt],
            )
            # Check if response.text is valid before stripping
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

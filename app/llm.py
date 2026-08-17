import time
from google.genai import Client
from config.env import api_key, llm_model

def mask(text: str | None, limit: int = 1500) -> str:
    if not text:
        return "None yet."
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[...{len(text) - limit} chars omitted — full content in state]"

class LLM:
    def __init__(self):
        self.client = Client(api_key=api_key)
        self.model = llm_model

    def generate(self, prompt: str, retries: int = 3):
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(model=self.model,
                                                               contents=prompt)
                return response.text
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[LLM] attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    def summarize(self, content: str, max_words: int = 200):
        prompt = f"""
        Summarize the following content in under {max_words} words.
        
        - Preserve all key facts, findings, conclusions, and named entities.
        - Do not add interpretation or commentary — only compress.
        - If the content is already short, return it unchanged.

        Content:{content}
        """
        return self.generate(prompt)
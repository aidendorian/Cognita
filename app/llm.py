import time
from abc import ABC, abstractmethod
from google.genai import Client
from google.genai.types import GenerateContentConfig
from config.env import api_key, llm_model, llm_backend

def mask(text: str | None, limit: int = 1500) -> str:
    if not text:
        return "None yet."
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[...{len(text) - limit} chars omitted — full content in state]"

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_output_tokens: int = 8192, **kwargs) -> str:
        ...

class GeminiBackend(LLMBackend):
    def __init__(self):
        self._client = Client(api_key=api_key)
        self._model = llm_model

    def generate(self, prompt: str, max_output_tokens: int = 8192, **kwargs) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=GenerateContentConfig(
                max_output_tokens=max_output_tokens,
            ),
        )
        text = response.text
        if text is None:
            candidates = getattr(response, 'candidates', [])
            if candidates:
                reason = getattr(candidates[0], 'finish_reason', 'unknown')
                safety = getattr(candidates[0], 'safety_ratings', [])
                raise ValueError(f"Gemini returned empty response. finish_reason={reason}, safety={safety}")
            raise ValueError("Gemini returned empty response")
        return text

class LlamaServerBackend(LLMBackend):
    """
    server first:
        ./build/bin/llama-server -hf unsloth/Qwen3.5-4B-GGUF:UD-Q5_K_XL -ngl 99 -c 32000
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1"):
        from openai import OpenAI
        self._client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
        )

    def generate(self, prompt: str, max_output_tokens: int = 8192, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model="local",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=None, # None because Reasoning Traces also counts as output tokens, keeping it free to generate locally.
            extra_body={
                "n_ctx": 32000,
                "cache_prompt": True,
            }
        )
        text = response.choices[0].message.content
        if text is None:
            raise ValueError("Local model returned empty response")
        return text

class LLM:
    def __init__(self):
        target = llm_backend

        if target == "hosted":
            print("Gemini")
            self._backend: LLMBackend = GeminiBackend()
        elif target == "local":
            print("")
            self._backend = LlamaServerBackend()
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {target!r} — must be 'hosted' or 'local'")

    def generate(self, prompt: str, retries: int = 3, max_output_tokens: int = 8192) -> str: #type: ignore
        """Call the LLM with exponential backoff on transient failures."""
        for attempt in range(retries):
            try:
                return self._backend.generate(prompt, max_output_tokens=max_output_tokens)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[LLM] attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    def summarize(self, content: str, max_words: int = 500) -> str:
        prompt = f"""
        Summarize the following content in under {max_words} words.
        - Preserve all key facts, findings, conclusions, and named entities.
        - Do not add interpretation or commentary — only compress.
        - If the content is already short, return it unchanged.

        Content:
        {content}
        """
        
        return self.generate(prompt, max_output_tokens=2000)
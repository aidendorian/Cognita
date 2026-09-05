import time
from abc import ABC, abstractmethod
from google.genai import Client
from google.genai.types import GenerateContentConfig
from config.env import api_key, llm_model, llm_backend, open_router_api_key, open_router_model
from observability.tracing import trace_llm_call
from openai import OpenAI
import httpx
import logging

logger = logging.getLogger("ResearchAgent")

RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.NetworkError,
)

RETRYABLE_HTTP_CODES = {429, 500, 502, 503}

def mask(text: str | None, limit: int = 1500) -> str:
    if not text:
        return "None yet."
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[...{len(text) - limit} chars omitted — full content in state]"

def mask_tail(text: str | None, limit: int = 1500) -> str:
    if not text:
        return "None yet."
    if len(text) <= limit:
        return text
    return f"[...{len(text) - limit} earlier chars omitted...]\n" + text[-limit:]

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
        
        trace_llm_call(
            name="gemini.generate",
            prompt=prompt,
            response=text,
            model=self._model,
            metadata={"max_output_tokens": max_output_tokens}
        )
        
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
        
        trace_llm_call(
            name="llama.generate",
            prompt=prompt,
            response=text,
            model="local",
            metadata={"max_output_tokens": max_output_tokens}
        )
        
        return text

class OpenRouterBackend(LLMBackend):
    def __init__(self):
        
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open_router_api_key)
        self._model = open_router_model

    def generate(self, prompt: str, max_output_tokens: int = 8192, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_output_tokens,
        )

        if not response.choices:
            logger.error("LLM response: %r", response)
            raise RuntimeError(f"LLM returned no choices: {response!r}")

        content = response.choices[0].message.content

        if content is None:
            logger.error("LLM choice has no content: %r", response)
            raise RuntimeError(f"LLM returned no message content: {response!r}")

        trace_llm_call(
            name="openrouter.generate",
            prompt=prompt,
            response=content,
            model=self._model,
            metadata={
                "max_output_tokens": max_output_tokens,
            },
        )
        return content

class LLM:
    def __init__(self):
        target = llm_backend

        if target == "gemini":
            print("Gemini")
            self._backend: LLMBackend = GeminiBackend()

        elif target == "openrouter":
            print("OpenRouter")
            self._backend = OpenRouterBackend()

        elif target == "local":
            print("Llama.cpp")
            self._backend = LlamaServerBackend()

        else:
            raise ValueError(f"Unknown LLM_BACKEND: {target} — must be 'hosted', 'openrouter', or 'local'")

    def generate(self, prompt: str, retries: int = 3, max_output_tokens: int = 8192) -> str:  # type: ignore
        for attempt in range(retries):
            try:
                return self._backend.generate(prompt, max_output_tokens=max_output_tokens)
            except Exception as e:
                is_retryable = isinstance(e, RETRYABLE_EXCEPTIONS)
                if not is_retryable:
                    status_code = (
                        getattr(e, "status_code", None)
                        or getattr(e, "code", None)
                        or getattr(e, "response", None) and getattr(e.response, "status_code", None)  # type: ignore[union-attr]
                    )
                    if isinstance(status_code, int) and status_code in RETRYABLE_HTTP_CODES:
                        is_retryable = True

                if not is_retryable or attempt == retries - 1:
                    raise
                wait = 15 * attempt
                print(f"[LLM] attempt {attempt + 1} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
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
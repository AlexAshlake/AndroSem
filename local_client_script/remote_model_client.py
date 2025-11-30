# remote_model_client.py
# -*- coding: utf-8 -*-
import requests


class RemoteTokenizer:
    """
    Remote tokenizer proxy:
    - `encode()` calls the server's `/count_tokens` function, using a genuine Qwen tokenizer.
    - The returned token ID is [0, 1, ..., n-1] because we only care about the length.
    """

    def __init__(self, base_url: str, use_lora: bool = False, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.use_lora = use_lora
        self.timeout = timeout

    def encode(self, text: str, add_special_tokens: bool = False):
        url = f"{self.base_url}/count_tokens"
        payload = {
            "text": text,
            "use_lora": self.use_lora,
            "add_special_tokens": add_special_tokens,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        n_tokens = int(data.get("tokens", 0))
        return list(range(n_tokens))


class RemoteLLMClient:
    """
    Remote LLM client:
    - generate(prompt) → Calls /generate
    - get_tokenizer() → Returns RemoteTokenizer (using the server's real tokenizer)
    """

    def __init__(self, base_url: str, use_lora: bool = False, timeout: int = 120):
        """
        :param base_url: e.g., "http://127.0.0.1:8000"
        :param use_lora: True: Use the LoRA classification model; False: Use the base model
        :param timeout: HTTP request timeout (in seconds)
        """
        self.base_url = base_url.rstrip("/")
        self.use_lora = use_lora
        self.timeout = timeout
        self._tok = None

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/generate"
        payload = {
            "prompt": prompt,
            "use_lora": self.use_lora,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("output", "").strip()

    def get_tokenizer(self):
        """
        For use with TokenCounter: Returns an object with an encode() method.
        The encode() function internally calls the remote /count_tokens, using a real tokenizer.
        """
        if self._tok is None:
            self._tok = RemoteTokenizer(self.base_url, self.use_lora, self.timeout)
        return self._tok

    def unload(self):
        """
        Instruct the server to unload the corresponding model (base or LoRA).
        Local objects can still be used; the next `generate` will prompt the server to reload them.
        """
        url = f"{self.base_url}/unload"
        payload = {"use_lora": self.use_lora}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            print(f"[RemoteLLMClient] unload failed: {e}")
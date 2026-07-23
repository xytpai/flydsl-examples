import os
import sys
import httpx
from abc import ABC, abstractmethod
from typing import Iterator
from urllib.parse import urlsplit
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")


def create_router_client(proxy_url: str | None) -> httpx.Client | None:
    if not proxy_url:
        return None

    scheme = urlsplit(proxy_url).scheme.lower()
    if scheme not in {"http", "https", "socks5"}:
        raise ValueError(
            "ROUTER_PROXY must use http://, https://, or socks5://"
        )

    try:
        return httpx.Client(
            proxy=proxy_url,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
    except ImportError as exc:
        if scheme == "socks5":
            raise RuntimeError(
                'SOCKS proxy support is missing; install it with: pip install "httpx[socks]"'
            ) from exc
        raise


def proxy_label(proxy_url: str) -> str:
    parsed = urlsplit(proxy_url)
    host = parsed.hostname or "<invalid-host>"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


class AgentBackend(ABC):
    def __init__(self):
        self.initialize()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def stream_response(self, content: str, max_tokens: int) -> Iterator[str]:
        pass


class OpenaiBackend(AgentBackend):
    def initialize(self):
        base_url = os.environ.get("BASE_URL")
        api_key = os.environ.get("API_KEY")
        router_proxy = os.environ.get("ROUTER_PROXY")
        client_options = {}

        if router_proxy:
            client_options["http_client"] = create_router_client(router_proxy)
            print(
                f"==== SYSTEM ==== ROUTER_PROXY:{proxy_label(router_proxy)}\n",
                flush=True,
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy",
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
            **client_options,
        )
        self.model = MODEL_NAME

    def stream_response(self, content: str, max_tokens: int = 65536) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
            stream=True,
        )
        for event in stream:
            if len(event.choices) > 0:
                delta = event.choices[0].delta
                if delta and delta.content and len(delta.content) > 0:
                    yield delta.content


def get_backend():
    global MODEL_NAME
    print(f"==== SYSTEM ==== MODEL_NAME:{MODEL_NAME}\n", flush=True)
    if "gpt" in MODEL_NAME.lower():
        return OpenaiBackend()
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")


if __name__ == "__main__":
    backend = get_backend()
    text = sys.argv[1]
    for chunk in backend.stream_response(text, 65536):
        print(chunk, end="", flush=True)
    print("", flush=True)

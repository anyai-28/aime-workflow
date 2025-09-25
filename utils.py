import time
import litellm
from litellm import RateLimitError


def call_llm_with_retry(
    model: str, messages: list, temperature: float, response_format: dict = None, max_retries: int = 3, **kwargs
):
    """指数バックオフ付きでLLM APIを呼び出す"""
    delay = 5
    for attempt in range(max_retries):
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **kwargs,
            }
            if response_format:
                params["response_format"] = response_format

            response = litellm.completion(**params)
            return response
        except RateLimitError:
            print(f"レートリミットエラー。{delay}秒待って再試行します... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # バックオフ時間を増やす
        except Exception as e:
            print(f"予期せぬAPIエラー: {e}")
            raise
    raise Exception("APIリクエストが最大再試行回数に達しました。")

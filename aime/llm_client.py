"""
LLM API呼び出しを一元管理するクライアント
Langfuse統合とエラーハンドリングを含む
"""
import time
from typing import Dict, List, Optional, Any
import litellm
from litellm import RateLimitError
from langfuse import observe
from aime.config import config


class LLMClient:
    """LLM API呼び出しを統一管理するクライアント"""

    def __init__(self):
        """LLMクライアントを初期化"""
        # litellm設定
        litellm.api_key = config.openai_api_key

    @observe(name="llm-completion")
    def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        LLM APIを呼び出してレスポンスを取得

        Args:
            messages: 会話履歴のメッセージリスト
            model: 使用するモデル名（デフォルト: config.openai_model）
            temperature: 生成温度（デフォルト: config.default_temperature）
            response_format: レスポンス形式指定
            max_retries: 最大再試行回数（デフォルト: config.max_retries）
            **kwargs: その他のパラメータ

        Returns:
            LLM APIのレスポンス

        Raises:
            Exception: 最大再試行回数に達した場合
        """
        model = model or config.openai_model
        temperature = temperature or config.default_temperature
        max_retries = max_retries or config.max_retries

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
                if attempt < max_retries - 1:
                    print(f"レートリミットエラー。{delay}秒待って再試行します... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

            except Exception as e:
                print(f"予期せぬAPIエラー: {e}")
                raise

        raise Exception("APIリクエストが最大再試行回数に達しました。")

    def completion_mini(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        軽量タスク用のminiモデルでAPIを呼び出し

        Args:
            messages: 会話履歴のメッセージリスト
            temperature: 生成温度
            **kwargs: その他のパラメータ

        Returns:
            LLM APIのレスポンス
        """
        return self.completion(
            messages=messages,
            model=config.openai_mini_model,
            temperature=temperature,
            **kwargs
        )


# グローバルLLMクライアントインスタンス
llm_client = LLMClient()
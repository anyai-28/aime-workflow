"""
設定管理モジュール
OpenAI API、Langfuse設定を統一管理
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class AimeConfig:
    """Aimeフレームワークの設定クラス"""

    # OpenAI設定
    openai_api_key: str = ""
    openai_model: str = "openai/gpt-4o"
    openai_mini_model: str = "openai/gpt-4o-mini"

    # Langfuse設定
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Google API設定
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

    # システム設定
    max_parallel_actors: int = 4
    max_retries: int = 3
    default_temperature: float = 0.3
    actor_max_turns: int = 5

    # ディレクトリ設定
    results_dir: str = "task_results"
    progress_file: str = "progress.md"
    final_report_file: str = "final_report.md"

    def __post_init__(self):
        """環境変数から設定を読み込み"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", self.langfuse_secret_key)
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key)
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID", self.google_cse_id)
        if host := os.getenv("LANGFUSE_HOST"):
            self.langfuse_host = host


# グローバル設定インスタンス
config = AimeConfig()

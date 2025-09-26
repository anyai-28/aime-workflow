import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from litellm import success_callback
from planner import DynamicPlanner


def main():
    """メイン実行関数"""
    # .envファイルから環境変数を読み込む
    load_dotenv()

    langfuse_callback_handler = CallbackHandler()
    success_callback.append(langfuse_callback_handler)
    
    # AimeのDynamic Plannerを初期化
    planner = DynamicPlanner()

    # ユーザーからのリクエスト
    user_request = (
        "東京での1泊2日の完璧な観光プランを作成してください。移動手段と予算の見積もりもお願いします。"
    )

    # Aimeフレームワークを実行
    planner.run(user_request)


if __name__ == "__main__":
    main()

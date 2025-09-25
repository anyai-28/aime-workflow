import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from litellm import success_callback
from planner import DynamicPlanner


def main():
    # .envファイルから環境変数を読み込む
    load_dotenv()

    langfuse_callback_handler = CallbackHandler()
    success_callback.append(langfuse_callback_handler)
    # AimeのDynamic Plannerを初期化 (同時実行アクター数を指定)
    planner = DynamicPlanner(max_parallel_actors=4)  # client引数を削除

    # ユーザーからのリクエスト
    # user_request = "AIエージェントの最新動向について調査し、その結果をブログ記事としてまとめてください。"
    user_request = (
        # "東京での1泊2日の完璧な観光プランを作成してください。出発地点は名古屋とし、移動手段と予算の見積もりもお願いします。"
        "東京での1泊2日の完璧な観光プランを作成してください。移動手段と予算の見積もりもお願いします。"
    )

    # Aimeフレームワークを実行
    planner.run(user_request)


if __name__ == "__main__":
    main()

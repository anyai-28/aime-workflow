import os
from dotenv import load_dotenv
from openai import OpenAI
from planner import DynamicPlanner


def main():
    # .envファイルから環境変数を読み込む
    load_dotenv()

    # OpenAIクライアントの初期化
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEYが設定されていません。")
    client = OpenAI(api_key=api_key)

    # AimeのDynamic Plannerを初期化 (同時実行アクター数を指定)
    planner = DynamicPlanner(openai_client=client, max_parallel_actors=4)

    # ユーザーからのリクエスト
    # user_request = "AIエージェントの最新動向について調査し、その結果をブログ記事としてまとめてください。"
    user_request = (
        "東京での1泊2日の完璧な観光プランを作成してください。出発地点は名古屋とし、移動手段と予算の見積もりもお願いします。"
    )

    # Aimeフレームワークを実行
    planner.run(user_request)


if __name__ == "__main__":
    main()

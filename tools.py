from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

from config import config


def web_search(query: str) -> str:
    """
    指定されたクエリでWeb検索を行い、上位の結果を返すツール。
    """
    print(f"  [TOOL] web_search: クエリ='{query}'")
    if query.startswith('"') and query.endswith('"'):
        query = query[1:-1]

    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return (
                "\n".join([f"Title: {res['title']}\nBody: {res['body']}" for res in results])
                if results
                else "検索結果が見つかりませんでした。"
            )
    except Exception as e:
        return f"検索中にエラーが発生しました: {e}"


def google_search(query: str, max_retries: int = 3) -> str:  # リトライ回数を引数に追加
    """
    指定されたクエリでGoogle検索を行い、上位の結果を返すツール。
    一時的なエラーの場合、数回リトライする。
    """
    print(f"  [TOOL] google_search: クエリ='{query}'")

    # ▼▼▼ クエリから不要な引用符を削除する処理を追加 ▼▼▼
    # LLMが生成しがちな前後のダブルクォーテーションを削除
    if query.startswith('"') and query.endswith('"'):
        query = query[1:-1]

    api_key = config.google_api_key
    cse_id = config.google_cse_id

    if not api_key or not cse_id:
        return "エラー: GOOGLE_API_KEY または GOOGLE_CSE_ID が環境変数に設定されていません。"

    # ▼▼▼ リトライ処理のループを追加 ▼▼▼
    for attempt in range(max_retries):
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=query, cx=cse_id, num=3).execute()

            if "items" in res and res["items"]:
                results = []
                for item in res["items"]:
                    title = item.get("title", "N/A")
                    snippet = item.get("snippet", "N/A")
                    link = item.get("link", "N/A")
                    results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}")
                return "\n---\n".join(results)
            else:
                # 検索結果が0件の場合はリトライ不要なので、ここでループを抜ける
                return "検索結果が見つかりませんでした。"

        except HttpError as e:
            # サーバーエラー(5xx)やレート制限(429)の場合のみリトライ
            if e.resp.status in [429, 500, 503]:
                print(f"  [WARN] Google Search APIエラー: {e.resp.status}。リトライします... ({attempt + 1}/{max_retries})")
                time.sleep(2**attempt)  # 1, 2, 4秒... と待機時間を増やす
            else:
                # その他のクライアントエラー(4xx)はリトライしても無駄なので即時失敗させる
                return f"Google API HTTPエラー（リトライ不可）: {e.resp.status} {e.reason}"
        except Exception as e:
            return f"予期せぬエラーが発生しました: {e}"

    return f"最大リトライ回数({max_retries}回)に達しました。検索に失敗しました。"


# Actorがタスク完了を宣言するための特別なツール
def finish(final_answer: str) -> str:
    """
    サブタスクの最終的な回答を返すためのツール。これを呼び出すとタスクが完了する。
    """
    return final_answer


def reflect(reflection: str) -> str:
    """
    外部ツールを使わずに、自身の思考や計画を整理するために使用する。
    次の行動計画を立てたり、状況を分析したりする際に呼び出す。
    このツールの観察結果は、次の思考のインプットとして利用できる。
    """
    print(f"  [TOOL] reflect: 思考内容='{reflection}'")
    return f"思考内容を記録しました: '{reflection}'"

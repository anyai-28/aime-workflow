from duckduckgo_search import DDGS


def web_search(query: str) -> str:
    """
    指定されたクエリでWeb検索を行い、上位の結果を返すツール。
    """
    print(f"  [TOOL] web_search: クエリ='{query}'")
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


# Actorがタスク完了を宣言するための特別なツール
def finish(final_answer: str) -> str:
    """
    サブタスクの最終的な回答を返すためのツール。これを呼び出すとタスクが完了する。
    """
    return final_answer

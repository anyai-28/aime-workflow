import json
import time
import os
from openai import OpenAI
from progress_manager import ProgressManagementModule
from factory import ActorFactory
from concurrent.futures import ThreadPoolExecutor

from utils import call_llm_with_retry


class DynamicPlanner:
    def __init__(self, openai_client: OpenAI, max_parallel_actors: int = 4):
        self.client = openai_client
        self.progress_manager = ProgressManagementModule()
        self.factory = ActorFactory(openai_client, self.progress_manager)
        self.results_dir = "task_results"
        self.max_parallel_actors = max_parallel_actors
        self.main_goal = ""

    def _decompose_task(self, main_goal: str) -> list[dict]:
        """LLMを使い、依存関係を含むサブタスクのリストに分解する"""
        prompt = f"""
ユーザーの複雑なリクエストを、実行可能なサブタスクのリストに細分化してください。
各サブタスクには一意のIDを0から振り、他のタスクに依存する場合はそのIDをリストで指定してください。
依存関係がなければ空リスト `[]` とします。タスクは論理的な順序で定義してください。
出力は以下のJSON形式の配列でなければなりません:
`[{{"id": 0, "description": "タスク1", "dependencies": []}}, {{"id": 1, "description": "タスク2", "dependencies": [0]}}]`

リクエスト: 「{main_goal}」
分解されたサブタスクのJSON:
"""
        response = call_llm_with_retry(  # <-- 変更なし (呼び出し元は同じ)
            client=self.client,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(response.choices[0].message.content)
            # "tasks"キーなどでラップされている場合に対応
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # 辞書の場合、値にリストが含まれていればそれを返す
                for key, value in data.items():
                    if not key == "dependencies":
                        if isinstance(value, list):
                            print(f"  [Planner] JSONオブジェクトからキー '{key}' のリストを抽出しました。")
                            return value
            raise ValueError("JSONが期待されるリスト形式ではありません。")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"タスク分解のJSONパースに失敗: {e}")
            return []

    def _refine_plan(self, trigger_reason: str):
        """現在の進捗と失敗理由に基づき、計画を動的に修正する"""
        print(f"\n[!!!] Planner: {trigger_reason} のため、計画の再評価と修正を開始します...")

        progress_context = self.progress_manager.get_progress_summary()

        prompt = f"""
あなたはプロジェクトマネージャーAIです。以下の初期目標と現在の進捗状況、そして再計画のトリガーとなった理由を考慮して、残りの計画を最適化してください。
タスクの追加、変更、削除が可能です。出力は以前と同じJSON形式（IDと依存関係を含む）で、"completed"ステータスのタスクはそのまま含めてください。
失敗したタスクは、代替案のタスクを新たに追加するか、修正して再試行できるようにしてください。

# 初期目標
{self.main_goal}

# 現在の計画と進捗
{progress_context}

# 再計画の理由
{trigger_reason}

# 修正後の新しい計画（JSON配列）:
"""
        response = call_llm_with_retry(
            client=self.client,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            new_plan = json.loads(response.choices[0].message.content)
            if isinstance(new_plan, dict) and "tasks" in new_plan:
                new_plan = new_plan["tasks"]

            print("--- 新しい計画が生成されました ---")
            self.progress_manager.update_tasks(new_plan)
            print("--- タスクリストが新しい計画で更新されました ---")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"計画修正のJSONパースに失敗しました: {e}")

    def _execute_task_wrapper(self, task: dict):
        """Actorの生成と実行をラップし、並列処理で呼び出せるようにする"""
        try:
            # Phase 2-1: Actorのインスタンス化
            print(f"  ▶ Actor Factory: タスク '{task['description']}' のActorを生成中...")
            actor = self.factory.create_actor(task)

            # Phase 2-2: Actorの実行
            print(f"  ▶ Dynamic Actor: タスク '{task['description']}' の実行を開始します...")
            result = actor.run()

            # 結果の保存
            result_filename = f"task_{task['id']}_result.md"
            result_filepath = os.path.join(self.results_dir, result_filename)
            with open(result_filepath, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"  ▶ タスク結果を '{result_filepath}' に保存しました。")

            return task["id"], result
        except Exception as e:
            error_message = f"タスク {task['id']} の実行中に予期せぬエラーが発生しました: {e}"
            print(f"  ▶ [ERROR] {error_message}")
            return task["id"], error_message

    def run(self, main_goal: str):
        self.main_goal = main_goal
        print(f"=== Aimeフレームワーク実行開始: {self.main_goal} ===")
        os.makedirs(self.results_dir, exist_ok=True)

        # Phase 1: タスク分解
        print("\n[Phase 1/4] Dynamic Planner: タスク分解を開始します...")
        subtasks = self._decompose_task(self.main_goal)
        if not subtasks:
            print("タスクの分解に失敗したため、処理を終了します。")
            return
        self.progress_manager.initialize_tasks(subtasks)

        # Phase 2: 実行ループ (依存関係を考慮した並列処理)
        print(f"\n[Phase 2/4] Planner: サブタスクの実行ループを開始します (最大ワーカー数: {self.max_parallel_actors})...")

        with ThreadPoolExecutor(max_workers=self.max_parallel_actors) as executor:
            active_futures = {}
            while not self.progress_manager.are_all_tasks_done():
                executable_tasks = self.progress_manager.get_executable_tasks()

                # 実行可能なタスクをワーカーに投入
                while len(active_futures) < self.max_parallel_actors and executable_tasks:
                    task_to_run = executable_tasks.pop(0)
                    if task_to_run["id"] not in [f.result()[0] for f in active_futures if f.done()] and task_to_run[
                        "id"
                    ] not in [active_futures[f] for f in active_futures if not f.done()]:
                        self.progress_manager.update_task_status(task_to_run["id"], "in_progress")
                        future = executor.submit(self._execute_task_wrapper, task_to_run)
                        active_futures[future] = task_to_run["id"]

                # 完了したタスクを処理
                if not active_futures:
                    if not self.progress_manager.get_executable_tasks() and not self.progress_manager.are_all_tasks_done():
                        print(
                            "[WARN] 実行可能なタスクがありませんが、まだ完了していないタスクがあります。デッドロックの可能性があります。"
                        )
                        # ここでデッドロック解消のための再計画をトリガーすることも可能
                        self._refine_plan("デッドロックの可能性: 実行可能なタスクがありません。")

                    time.sleep(2)
                    continue

                done_futures = []
                for future in list(active_futures.keys()):
                    if future.done():
                        task_id = active_futures[future]
                        try:
                            _task_id, result = future.result()
                            task_failed = (
                                "実行中に予期せぬエラーが発生しました" in result or "最大ターン数に達しました" in result
                            )

                            if task_failed:
                                status = "failed"
                                self.progress_manager.update_task_status(task_id, status, result)
                                self._refine_plan(
                                    f"タスク {task_id} ('{self.progress_manager.tasks[task_id]['description']}') が失敗しました。"
                                )
                            else:
                                status = "completed"
                                self.progress_manager.update_task_status(task_id, status, result)

                        except Exception as exc:
                            print(f"  ▶ [ERROR] タスク {task_id} の実行で致命的な例外: {exc}")
                            self.progress_manager.update_task_status(task_id, "failed", str(exc))
                            self._refine_plan(f"タスク {task_id} が致命的な例外で失敗しました。")

                        done_futures.append(future)

                for future in done_futures:
                    del active_futures[future]

                time.sleep(1)

        print("\n[Phase 2/4] 全てのサブタスクの実行が完了しました。")

        # Phase 3: 最終報告書の作成
        print("\n[Phase 3/4] Planner: 全てのタスクが完了しました。最終報告書を作成します...")
        final_report = self._generate_final_report(main_goal)

        # Phase 4: 最終報告書の出力
        print("\n[Phase 4/4] Planner: 最終報告書をファイルに出力します...")
        report_filepath = "final_report.md"
        try:
            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"--- 最終報告書を {report_filepath} に出力しました ---")
        except IOError as e:
            print(f"ファイルへの書き込みに失敗しました: {e}")

        print("\n=== Aimeフレームワークの全処理が完了しました ===")
        print("\n--- 最終報告書 ---")
        print(final_report)
        print("--------------------")

    def _generate_final_report(self, main_goal: str):
        """全てのサブタスクの結果を統合して最終報告書を作成する"""
        results_context = ""
        for task in self.progress_manager.tasks:
            results_context += f"## サブタスク「{task['description']}」\n\n**ステータス:** {task['status']}\n**結果:**\n{task.get('result', 'N/A')}\n\n---\n"

        prompt = f"""
以下のサブタスクの実行結果を統合し、ユーザーの最初の要求に対する最終的な回答を、Markdown形式でプロフェッショナルに作成してください。

# 最初の要求
「{main_goal}」

# 各サブタスクの実行結果
{results_context}

# 最終報告書 (Markdown形式)
"""
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.2
        )
        return response.choices[0].message.content

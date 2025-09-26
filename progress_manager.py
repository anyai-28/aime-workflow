import os
from threading import RLock


class ProgressManagementModule:
    """
    システム全体のタスク進捗を管理する中央モジュール。
    進捗をMarkdownファイルにも出力する。
    """

    def __init__(self, filepath="progress.md"):
        self.tasks = []
        self._lock = RLock()
        self.filepath = filepath
        # 初期化時に空ファイルを作成
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("# Aime Framework Task Progress\n\n")

    def _write_progress_to_file(self):
        """現在の進捗状況をMarkdownファイルに書き込む"""
        # このメソッドはロックを取得済みのコンテキストから呼ばれることを想定
        md_content = "# Aime Framework Task Progress\n\n"
        for task in self.tasks:
            if task["status"] == "completed":
                marker = "✅"
            elif task["status"] == "in_progress":
                marker = "[-]"
            elif task["status"] == "failed":
                marker = "❌"
            else:  # pending
                marker = "[ ]"
            md_content += f"- {marker} {task['description']}\n"

            # ログの表示を追加
            if task.get("logs"):
                for log_entry in task["logs"]:
                    md_content += f"  - 📝 Log: {log_entry}\n"

            # if task["status"] == "completed" and task["result"]:
            #     md_content += f"  - **Result:** {str(task['result'])[:150]}...\n"
            if task["status"] == "failed" and task["result"]:
                md_content += f"  - **failed:** {str(task['result'])[:100]}...\n"

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

    def initialize_tasks(self, tasks_with_deps: list[dict]):
        """依存関係を含むタスクリストを初期化する"""
        with self._lock:
            self.tasks = []
            for task_data in tasks_with_deps:
                self.tasks.append(
                    {
                        "id": task_data["id"],
                        "description": task_data["description"],
                        "dependencies": task_data.get("dependencies", []),
                        "status": "pending",
                        "result": None,
                        "logs": [],
                    }
                )
            print("--- Progress Manager: 依存関係を含むタスクリストを初期化しました ---")
            self.display_progress()
            self._write_progress_to_file()

    def update_tasks(self, new_plan: list[dict]):
        """LLMによって再生成された新しい計画でタスクリストを更新する"""
        with self._lock:
            new_tasks = []
            existing_ids = {t["id"] for t in self.tasks}
            completed_tasks = {t["id"]: t for t in self.tasks if t["status"] == "completed"}

            for task_data in new_plan:
                task_id = task_data["id"]
                if task_id in completed_tasks:
                    new_tasks.append(completed_tasks[task_id])  # 完了済みタスクは維持
                else:
                    new_tasks.append(
                        {
                            "id": task_id,
                            "description": task_data["description"],
                            "dependencies": task_data.get("dependencies", []),
                            "status": "pending",  # 失敗したタスクも再度pendingに戻す
                            "result": None,
                            "logs": [],
                        }
                    )
            self.tasks = new_tasks
            self.display_progress()
            self._write_progress_to_file()

    def update_task_status(self, task_id: int, status: str, result: str = None):
        """タスクのステータスと結果を更新する"""
        with self._lock:
            for task in self.tasks:
                if task["id"] == task_id:
                    task["status"] = status
                    if result:
                        task["result"] = result
                    print(
                        f"--- Progress Manager: タスク {task_id} ('{task['description']}') のステータスを {status} に更新 ---"
                    )
                    break
            self.display_progress()
            self._write_progress_to_file()

    def get_executable_tasks(self) -> list[dict]:
        """実行可能（依存関係が満たされた）なタスクを全て返す"""
        with self._lock:
            completed_ids = {t["id"] for t in self.tasks if t["status"] == "completed"}
            executable_tasks = []
            for task in self.tasks:
                if task["status"] == "pending":
                    if all(dep_id in completed_ids for dep_id in task["dependencies"]):
                        executable_tasks.append(task)
            return executable_tasks

    def get_progress_summary(self) -> str:
        """計画修正のためにLLMに渡す進捗サマリーを生成する"""
        with self._lock:
            summary = []
            for task in self.tasks:
                task_summary = f"- Task {task['id']}: {task['description']} (Status: {task['status']})"
                if task["result"]:
                    summary.append(f"{task_summary}\n  - Result: {str(task['result'])[:200]}...")
                else:
                    summary.append(task_summary)
            return "\n".join(summary)

    def add_task_log(self, task_id: int, message: str):
        """タスクにログメッセージを追加する（リアルタイム進捗報告用）"""
        with self._lock:
            for task in self.tasks:
                if task["id"] == task_id:
                    task["logs"].append(message)
                    print(f"--- Progress Manager: タスク {task_id} にログを追加: '{message}' ---")
                    break
            self.display_progress()
            self._write_progress_to_file()

    def get_pending_tasks(self) -> list[dict]:
        """実行待ち（pending状態）のタスクを全て返す"""
        with self._lock:
            return [task for task in self.tasks if task["status"] == "pending"]

    def are_all_tasks_done(self) -> bool:
        """全てのタスクが完了したか確認する"""
        with self._lock:
            return all(task["status"] in ["completed", "failed"] for task in self.tasks)

    def display_progress(self):
        """現在の進捗状況をコンソールに表示する"""
        with self._lock:
            print("\n--- 現在のタスク進捗 ---")
            for task in self.tasks:
                if task["status"] == "completed":
                    marker = "[x]"
                elif task["status"] == "in_progress":
                    marker = "[-]"
                elif task["status"] == "failed":
                    marker = "[//]"
                else:
                    marker = "[ ]"
                print(f"{marker} {task['description']}")
                # ログのコンソール表示も追加
                if task.get("logs"):
                    for log_entry in task["logs"]:
                        print(f"    📝 Log: {log_entry}")
            print("-----------------------\n")

    def get_completed_task_results(self, task_ids: list[int]) -> dict:
        """指定されたIDの完了済みタスクの結果を辞書で返す"""
        with self._lock:
            results = {}
            for task in self.tasks:
                if task["id"] in task_ids and task["status"] == "completed":
                    results[task["id"]] = task.get("result")
            return results

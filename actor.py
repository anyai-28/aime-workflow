import re
from typing import Dict, Any
from langfuse import observe
from config import config
from llm_client import llm_client


class DynamicActor:
    """
    特定のサブタスクを実行するために動的にインスタンス化される自律エージェント。
    ReActフレームワークに基づいて動作する。
    """

    def __init__(self, subtask: Dict[str, Any], persona: str, knowledge: str, tools: Dict[str, Any], progress_manager):
        """
        Dynamic Actorを初期化

        Args:
            subtask: 実行するサブタスク
            persona: エージェントのペルソナ
            knowledge: 知識ベース
            tools: 利用可能なツール
            progress_manager: 進捗管理モジュール
        """
        self.subtask = subtask
        self.persona = persona
        self.knowledge = knowledge
        self.progress_manager = progress_manager
        self.history = []
        self.max_turns = config.actor_max_turns

        # ツールをフラット化
        self.available_tools = tools

    def _update_progress(self, message: str) -> str:
        """
        タスクの実行中に、重要な中間進捗や発生した問題を報告するためのツール。
        このツールはタスクを完了させません。最終報告には finish ツールを使ってください。
        """
        self.progress_manager.add_task_log(self.subtask["id"], message)
        return "進捗が正常に報告されました。"

    def _build_prompt(self, current_turn: int):
        """LLMに送るプロンプトを構築する"""
        finish_desc = """finish(report: str): 全ての作業が完了した際に呼び出す最終報告ツール。引数には必ず {"status": "success" or "failure", "message": "成果物 or 失敗理由"} という形式のJSON文字列を指定してください。"""

        tool_descriptions = "\n".join(
            [f"- {name}: {func.__doc__.strip()}" for name, func in self.available_tools.items() if name != "finish"]
        )
        tool_descriptions += f"\n- finish: {finish_desc}"

        history_str = ""
        if not self.history:
            history_str = "まだ行動していません。最初の思考と行動を始めてください。\n"
        else:
            for turn in self.history:
                history_str += f"思考: {turn['thought']}\n行動: {turn['action_str']}\n観察: {turn['observation']}\n\n"

        prompt = f"""あなたは、**{self.persona}** という役割を持つ、非常に有能な専門家AIです。

# タスク
「{self.subtask["description"]}」

# 前提知識
{self.knowledge if self.knowledge else "利用可能な前提知識はありません。"}

# 利用可能なツール
{tool_descriptions}

# 思考と行動の履歴
{history_str}
---
**あなたの行動原則:**
あなたの目的は、上記タスクを効率的に達成することです。以下の思考プロセスに従い、行動してください。

**重要: 現在 {current_turn} / {self.max_turns} ターン目です。残りターン数が少ない場合（目安として半分以上経過した場合）は、新たな調査よりも、これまでの情報で結論をまとめて `finish` ツールで報告することを最優先してください。考えすぎるのは禁物です。**

1.  **思考:** 現在のタスクと履歴を分析し、次に何をすべきかを日本語で具体的に記述してください。
2.  **行動:** `ツール名[引数]` の形式で、実行するツールを一つだけ記述してください。
3.  **内省:** **具体的な行動の前に計画を立てたり、状況を整理したりする必要がある場合は、`reflect`ツールを使ってください。** これは思考を次のステップに進めるための重要なプロセスです。
4.  **進捗報告:** タスクの実行中に重要な中間結果や問題を発見した場合は、`update_progress`ツールを使って状況を報告してください。

**最重要指示:**
タスクを達成するために必要な情報がすべて集まった、あるいはタスクの遂行が不可能だと判断したら、**必ず `finish` ツールをJSON形式の引数で呼び出してください。**
例 (成功): `finish[{{"status": "success", "message": "# 調査結果..."}}]`
例 (失敗): `finish[{{"status": "failure", "message": "情報が見つかりませんでした。"}}]`


次に取るべきあなたの思考を記述してください。
思考:"""
        return prompt.strip()

    def _parse_llm_output(self, response_text: str):
        """LLMの出力から思考と行動を抽出する"""
        thought_match = re.search(r"思考:(.*?)行動:", response_text, re.DOTALL)
        action_match = re.search(r"行動:(.*)", response_text, re.DOTALL)

        thought = ""
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            # "行動:"が見つからない場合は、全体を思考とみなす
            thought = response_text.strip()

        action_str = action_match.group(1).strip() if action_match else ""

        if not action_str:
            return thought, None, None

        tool_name_match = re.match(r"(\w+)", action_str)
        if not tool_name_match:
            return thought, action_str, "無効な行動形式です。"

        tool_name = tool_name_match.group(1)

        arg_match = re.search(r"\[(.*)\]", action_str, re.DOTALL)
        arg = arg_match.group(1).strip() if arg_match else ""

        return thought, tool_name, arg

    @observe(name="Actor-Execution")
    def run(self) -> str:
        """
        ReActループを実行してサブタスクを遂行する

        Returns:
            タスク実行の結果
        """
        for i in range(self.max_turns):
            prompt = self._build_prompt(current_turn=i + 1)

            response = llm_client.completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_text = response.choices[0].message.content
            # LLMの出力形式を安定させるための補完
            if "行動:" not in response_text:
                # 行動がない場合は、タスク完了とみなし、最後の思考を要約させる
                response_text += f"\n行動: finish[{response_text.strip()}]"

            thought, tool_name, arg = self._parse_llm_output(response_text)

            log_message = [
                f"\n[Actor Turn {i + 1}/{self.max_turns}] - Task ID: {self.subtask['id']}",
                f"  🤔 思考: {thought}",
                f"  ⚡ 行動: {tool_name}[{arg}]",
            ]
            print("\n".join(log_message))

            if tool_name in self.available_tools:
                if tool_name == "finish":
                    print("  [TOOL] finish: タスク完了。最終成果物を返します。")
                    # 引数（arg）が最終成果物そのものになる
                    return arg if arg else "成果物が生成されませんでした。"

                tool_function = self.available_tools[tool_name]
                try:
                    observation = tool_function(arg)
                except Exception as e:
                    observation = f"ツール実行中にエラーが発生しました: {e}"

                print(f"  👀 観察: {str(observation)[:300]}...")

                self.history.append({"thought": thought, "action_str": f"{tool_name}[{arg}]", "observation": observation})
            else:
                print(f"  [ERROR] '{tool_name}' というツールは存在しません。")
                self.history.append(
                    {
                        "thought": thought,
                        "action_str": f"{tool_name}[{arg}]",
                        "observation": f"エラー: '{tool_name}' というツールは存在しません。利用可能なツールリストを確認してください。",
                    }
                )

        print(f">>> Dynamic Actor: 最大ターン数に達しました。タスクID {self.subtask['id']} を終了します。 <<<")
        final_summary = "最大ターン数に達したため、タスクを完了できませんでした。以下は実行履歴の要約です。\n"
        for turn in self.history:
            final_summary += (
                f"- 思考: {turn['thought']}\n- 行動: {turn['action_str']}\n- 観察: {str(turn['observation'])[:100]}...\n"
            )
        return final_summary

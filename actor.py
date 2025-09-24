import re
from openai import OpenAI

from utils import call_llm_with_retry


class DynamicActor:
    """
    特定のサブタスクを実行するために動的にインスタンス化される自律エージェント。
    ReActフレームワークに基づいて動作する。
    """

    def __init__(self, subtask: dict, persona: str, knowledge: str, tools: dict, openai_client: OpenAI, progress_manager):
        self.subtask = subtask
        self.persona = persona
        self.knowledge = knowledge
        self.client = openai_client
        self.progress_manager = progress_manager
        self.history = []
        self.max_turns = 7

        # 利用可能なツールを初期化し、リアルタイム進捗更新ツールを追加
        self.available_tools = tools.copy()
        self.available_tools["update_progress"] = self._update_progress

    def _update_progress(self, message: str) -> str:
        """
        タスクの実行中に、重要な中間進捗や発生した問題を報告するためのツール。
        このツールはタスクを完了させません。最終報告には finish ツールを使ってください。
        """
        self.progress_manager.add_task_log(self.subtask["id"], message)
        return "進捗が正常に報告されました。"

    def _build_prompt(self):
        """LLMに送るプロンプトを構築する"""
        tool_descriptions = "\n".join([f"- {name}: {func.__doc__.strip()}" for name, func in self.available_tools.items()])

        history_str = ""
        if not self.history:
            history_str = "まだ行動していません。最初の思考と行動を始めてください。\n"
        else:
            for turn in self.history:
                history_str += f"思考: {turn['thought']}\n行動: {turn['action_str']}\n観察: {turn['observation']}\n\n"

        prompt = f"""
あなたは以下のペルソナを持つ、非常に有能な専門家AIです:
**ペルソナ:** {self.persona}

あなたの現在のタスクは以下の通りです:
**タスク:** 「{self.subtask["description"]}」

**関連知識:**
{self.knowledge}

**利用可能なツール:**
{tool_descriptions}

**思考と行動の履歴:**
{history_str}
---
**あなたの行動原則:**

1.  **思考:** 現在のタスクと履歴を分析し、次に何をすべきかを日本語で具体的に記述してください。
2.  **行動:** `ツール名[引数]` の形式で、実行するツールを一つだけ記述してください。
3.  **進捗報告:** タスクの実行中に重要な中間結果や問題を発見した場合は、`update_progress`ツールを使って状況を報告してください。これは計画全体を調整するために重要な情報となります。

**最重要指示:**
タスクを達成するために必要な情報がすべて集まったと判断したら、**必ず `finish` ツールを呼び出してください。**
`finish` ツールの引数には、**単なる最後の思考ではなく、これまでの全プロセス（特に『観察』で得られた情報）を総合的に要約した、タスクに対する完全な最終成果物**をMarkdown形式で記述してください。
この成果物は、他の人があなたの作業内容を知らなくても理解できるように、自己完結している必要があります。
例：finish[最終的な旅程はこちらです...]

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

    def run(self):
        """ReActループを実行してサブタスクを遂行する"""
        for i in range(self.max_turns):
            prompt = self._build_prompt()

            response = call_llm_with_retry(
                client=self.client,
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            response_text = response.choices[0].message.content
            # LLMの出力形式を安定させるための補完
            if "行動:" not in response_text:
                # 行動がない場合は、タスク完了とみなし、最後の思考を要約させる
                response_text += f"\n行動: finish[{response_text.strip()}]"

            thought, tool_name, arg = self._parse_llm_output(f"思考: {response_text}")

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

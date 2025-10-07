from aime.actor import DynamicActor
from aime.tools import web_search, finish, reflect, google_search
from langfuse import observe
from aime.llm_client import llm_client


class ActorFactory:
    """
    サブタスクの要件に基づいて、特化したDynamic Actorをインスタンス化する。
    ペルソナはLLMによって動的に生成される。
    """

    def __init__(self, progress_manager):
        self.progress_manager = progress_manager
        self.base_tools = {"finish": finish, "web_search": google_search, "reflect": reflect}

    @observe()
    def _generate_persona(self, subtask_description: str) -> str:
        """
        LLMを使ってサブタスクに最適なペルソナを生成する

        Args:
            subtask_description: サブタスクの説明

        Returns:
            生成されたペルソナ
        """
        print("    L Factory: LLMに最適なペルソナを問い合わせ中...")
        prompt = f"""
以下のサブタスクを実行するのに最も適した専門家の役割（ペルソナ）を、簡潔な日本語で一行で記述してください。

例：
サブタスク: 「東京の主要な交通手段（電車、地下鉄）について調べる」
ペルソナ: 「東京の公共交通網に精通した交通コンサルタント。」

サブタスク: 「{subtask_description}」
ペルソナ:
"""
        try:
            response = llm_client.completion_mini(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )
            persona = response.choices[0].message.content.strip().replace("ペルソナ:", "").strip()
            return persona if persona else "多才なアシスタント。"
        except Exception as e:
            print(f"    L Factory: ペルソナ生成中にエラーが発生しました: {e}")
            return "多才なアシスタント。"  # エラー時はデフォルトを返す  # エラー時はデフォルトを返す

    @observe()
    def create_actor(self, subtask: dict, knowledge_context: str = "") -> DynamicActor:
        """
        サブタスクを分析し、適切なペルソナ、知識、ツールを持つActorを生成する
        """
        description = subtask["description"]

        # 1. LLMでペルソナを動的に生成
        persona = self._generate_persona(description)
        print(f"    L Factory: 生成されたペルソナ -> 「{persona}」")

        # 2. サブタスク内容に基づいて知識とツールを決定
        tools = {**self.base_tools}

        print(f"--- Actor Factory: 「{persona}」のペルソナを持つActorを生成しました ---")

        return DynamicActor(
            subtask=subtask,
            persona=persona,
            knowledge=knowledge_context,
            tools=tools,
            progress_manager=self.progress_manager,
        )

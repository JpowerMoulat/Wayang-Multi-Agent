from langchain_openai import ChatOpenAI
from ai_wayang_multi.config.settings import DECOMPOSER_AGENT_CONFIG
from ai_wayang_multi.llm.prompt_loader import PromptLoader
from ai_wayang_multi.llm.models import DataSources, WayangPlanHighLevel


class Decomposer:
    """
    Creates an abstract plan from user query and decomposes it into steps for Builder agents

    """

    def __init__(
        self,
        model: str | None = None,
        reasoning: str | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model or DECOMPOSER_AGENT_CONFIG.get("model")
        self.reasoning = reasoning or DECOMPOSER_AGENT_CONFIG.get("reason_effort")
        self.system_prompt = (
            system_prompt or PromptLoader().load_decomposer_system_prompt()
        )
        self.chat = []

    def set_model_and_reasoning(self, model: str, reasoning: str) -> None:
        """
        Sets objects model and reasoning if any.
        Mostly for testing

        Args:
            model (str): GPT-model
            reasoning (str): Reasoning level if any

        """
        self.model = model
        self.reasoning = reasoning

    def start(self) -> None:
        """
        Clears chat history without resetting agent

        """
        self.chat = [{"role": "system", "content": self.system_prompt}]

    def generate(self, query: str, selected_data: DataSources) -> WayangPlanHighLevel:
        """
        Generates an abstract plan for builders

        Args:
            query (str): A query in natural language
            selected_data (DataSources): The selected data sources available

        Returns:
            (WayangPlanHighLevel): A high level WayangPlan

        """

        # Generate prompt
        prompt = PromptLoader().load_decomposer_prompt(query, selected_data)

        # Append user prompt to chat
        self.chat.append({"role": "user", "content": prompt})

        # Build LLM (lazy so set_model_and_reasoning takes effect)
        model_kwargs = {"reasoning_effort": self.reasoning} if self.reasoning else {}
        llm = ChatOpenAI(model=self.model, model_kwargs=model_kwargs)
        result = llm.with_structured_output(WayangPlanHighLevel, include_raw=True).invoke(self.chat)

        # Return response
        return {"raw": result["raw"], "response": result["parsed"]}

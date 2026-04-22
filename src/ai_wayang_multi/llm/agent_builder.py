from langchain_openai import ChatOpenAI
from ai_wayang_multi.config.settings import BUILDER_AGENT_CONFIG
from typing import List
from ai_wayang_multi.llm.models import WayangPlan, Step
from ai_wayang_multi.llm.prompt_loader import PromptLoader


class Builder:
    """
    Builder Agent based on OpenAI's GPT-models.
    The agents build an logical, abstract plan from natural langauge query
    """

    def __init__(
        self,
        model: str | None = None,
        reasoning: str | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model or BUILDER_AGENT_CONFIG.get("model")
        self.reasoning = reasoning or BUILDER_AGENT_CONFIG.get("reason_effort")
        self.system_prompt = system_prompt or None
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

    def start(self, query: str, selected_data: dict) -> None:
        """
        Cleans Builder Agent so it only includes system prompt

        Args:
            query (str): The query (refined) of the plan to be built
            selected_data (dict): The data sources to be used selected by Specifier Agent

        """

        self.system_prompt = PromptLoader().load_builder_system_prompt(
            query, selected_data
        )
        self.chat = [{"role": "system", "content": self.system_prompt}]

    def generate(self, step: Step, previous_steps: List) -> WayangPlan:
        """
        Generate the step logic with correct operations

        Args:
            step (Step): The step to be generated as WayangPlan
            previous_steps: The previous dependend steps already generated

        Returns:
            (WayangPlan): The new steps generated

        """

        # Load prompt
        prompt = PromptLoader().load_builder_prompt(step, previous_steps)

        # Make a copy of the chat for this builder
        this_chat = self.chat

        # Append to this chat for this builder
        this_chat.append({"role": "user", "content": prompt})

        # Build LLM (lazy so set_model_and_reasoning takes effect)
        model_kwargs = {"reasoning_effort": self.reasoning} if self.reasoning else {}
        llm = ChatOpenAI(model=self.model, model_kwargs=model_kwargs)
        result = llm.with_structured_output(WayangPlan, include_raw=True).invoke(this_chat)

        # Return response
        return {"raw": result["raw"], "wayang_subplan": result["parsed"]}

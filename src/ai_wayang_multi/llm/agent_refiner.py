from langchain_openai import ChatOpenAI
from ai_wayang_multi.config.settings import REFINER_AGENT_CONFIG
from typing import List
from ai_wayang_multi.llm.models import WayangPlan, Step, DataSources
from ai_wayang_multi.llm.prompt_loader import PromptLoader


class Refiner:
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
        self.model = model or REFINER_AGENT_CONFIG.get("model")
        self.reasoning = reasoning or REFINER_AGENT_CONFIG.get("reason_effort")
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

    def start(self, selected_data: DataSources) -> None:
        """
        Cleans Builder Agent so it only includes system prompt

        Args:
            selected_data (dict): The data sources to be used selected by Specifier Agent

        """

        self.system_prompt = PromptLoader().load_refiner_system_prompt(selected_data)
        self.chat = [{"role": "system", "content": self.system_prompt}]

    def generate(self, query: str, wayang_plan: WayangPlan) -> WayangPlan:
        """
        Refines and aligns the current wayang_plan

        Args:
            query (str): The refined query
            wayang_plan (WayangPlan): The current full Wayang Plan to be refined

        Returns:
            (WayangPlan): The refined Wayang Plan

        """

        # Load prompt
        prompt = PromptLoader().load_refiner_prompt(query, wayang_plan)

        # Make a copy of the chat for this refiner
        this_chat = self.chat

        # Append to this chat for this refiner
        this_chat.append({"role": "user", "content": prompt})

        # Build LLM (lazy so set_model_and_reasoning takes effect)
        model_kwargs = {"reasoning_effort": self.reasoning} if self.reasoning else {}
        llm = ChatOpenAI(model=self.model, model_kwargs=model_kwargs)
        result = llm.with_structured_output(WayangPlan, include_raw=True).invoke(this_chat)

        # Return response
        return {"raw": result["raw"], "wayang_plan": result["parsed"]}

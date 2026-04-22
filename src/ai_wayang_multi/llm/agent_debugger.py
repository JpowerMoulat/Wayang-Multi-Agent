from langchain_openai import ChatOpenAI
from typing import List
from ai_wayang_multi.config.settings import DEBUGGER_AGENT_CONFIG
from ai_wayang_multi.llm.prompt_loader import PromptLoader
from ai_wayang_multi.llm.models import WayangPlan


class Debugger:
    """
    Debugger Agent based on OpenAI's GPT-models.
    The agent takes a failed Wayang plan and tries to fix it. Returns a fixed plan.

    """

    def __init__(
        self,
        model: str | None = None,
        reasoning: str | None = None,
        system_prompt: str | None = None,
        version: int | None = None,
    ):
        self.model = model or DEBUGGER_AGENT_CONFIG.get("model")
        self.reasoning = reasoning or DEBUGGER_AGENT_CONFIG.get("reason_effort")
        self.system_prompt = (
            system_prompt or PromptLoader().load_debugger_system_prompt()
        )
        self.version = version or 0
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

    def get_version(self) -> int:
        """
        Get the iteration of the plan

        Returns:
            int: Plan version

        """

        return self.version

    def set_vesion(self, version: int) -> int:
        """
        Set the plan iteration or version

        Args:
            version (int): Plan version

        Returns:
            int: Plan version

        """

        # Sets plan version
        self.version = version

        return self.get_version()

    def start(self) -> None:
        """
        Cleans the Debugger Agents chat so it only includes the system prompt

        """

        self.chat = [{"role": "system", "content": self.system_prompt}]

    def debug_plan(
        self, query: str, plan: WayangPlan, wayang_errors: str, val_errors: List
    ):
        """
        Debug a failed plan and tries to return. a new one

        Args:
            query (str): The refined user query
            plan (WayangPlan): The failed Wayang plan for debugging
            wayang_errors (str): The error given by the Wayang server if any
            val_errors (List): The error given by the PlanValidator if any

        Returns:
            A fixed plan

        """

        # increment version
        self.version += 1

        # Create new user prompt
        prompt = PromptLoader().load_debugger_prompt(
            query, plan, wayang_errors, val_errors
        )

        # Add user prompt to chat
        self.chat.append({"role": "user", "content": prompt})

        # Build LLM (lazy so set_model_and_reasoning takes effect)
        model_kwargs = {"reasoning_effort": self.reasoning} if self.reasoning else {}
        llm = ChatOpenAI(model=self.model, model_kwargs=model_kwargs)
        result = llm.with_structured_output(WayangPlan, include_raw=True).invoke(self.chat)

        # Format text answer from agent
        wayang_plan = result["parsed"]
        answer = PromptLoader().load_debugger_answer(wayang_plan)

        # Add agent answer to chat - necessary if another debug iteration is needed
        self.chat.append({"role": "assistant", "content": answer})

        # Return output
        return {"raw": result["raw"], "wayang_plan": wayang_plan, "version": self.version}

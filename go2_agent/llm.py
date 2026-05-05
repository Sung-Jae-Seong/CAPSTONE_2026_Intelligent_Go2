#  Copyright (c) 2025. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import dotenv
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """\
You are a task planner for a quadruped robot (Unitree Go2).
Your job is to decompose a complex, long-horizon robot instruction into a sequence of simple, atomic subtasks.

Rules:
1. Number each subtask as "Task 1:", "Task 2:", etc.
2. Each subtask must describe exactly one action step.
3. A subtask may be:
   - a navigation command toward one target object or location, or
   - a single direct motion command with one duration, distance, or angle.
4. Maintain the original logical order.
5. Output ONLY the numbered task list. No extra explanation.
"""

USER_PROMPT_TEMPLATE = """\
Decompose the following instruction into subtasks.

Instruction: "{instruction}"
"""


def get_llm(streaming: bool = False):
    """A helper function to get the LLM instance.

    Supports OpenAI (default), Anthropic and Ollama models.
    Set the LLM_PROVIDER env variable to switch between providers:
      - "openai" (default): uses OPENAI_API_KEY
      - "anthropic": uses ANTHROPIC_API_KEY
      - "ollama": uses local Ollama instance
    """
    dotenv.load_dotenv(dotenv.find_dotenv())

    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    supported = ("openai", "anthropic", "ollama")
    if provider not in supported:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be one of: {', '.join(supported)}"
        )

    if provider == "openai":
        llm = ChatOpenAI(
            api_key=get_env_variable("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL", "Qwen/Qwen3.5-4B"),
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            streaming=streaming,
        )
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic support. "
                "Install it with: pip install langchain-anthropic"
            )
        llm = ChatAnthropic(
            api_key=get_env_variable("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            streaming=streaming,
        )
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama support. "
                "Install it with: pip install langchain-ollama"
            )
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            streaming=streaming,
        )

    return llm


def get_env_variable(var_name: str) -> str:
    """
    Retrieves the value of the specified environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.

    This function provides a consistent and safe way to retrieve environment variables.
    By using this function, we ensure that all required environment variables are present
    before proceeding with any operations. If a variable is not set, the function will
    raise a ValueError, making it easier to debug configuration issues.
    """
    value = os.getenv(var_name)
    if value is None:
        msg = f"Environment variable {var_name} is not set."
        raise ValueError(msg)
    return value


@dataclass
class SubTask:
    index: int
    instruction: str

    def __repr__(self):
        return f"Task {self.index}: {self.instruction}"


@dataclass
class PlannerState:
    original_instruction: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    current_task_idx: int = 0
    completed: bool = False

    @property
    def current_task(self) -> Optional[SubTask]:
        if self.completed or self.current_task_idx >= len(self.subtasks):
            return None
        return self.subtasks[self.current_task_idx]

    @property
    def total_tasks(self) -> int:
        return len(self.subtasks)

    @property
    def progress_str(self) -> str:
        if self.completed:
            return "Task Success - All subtasks completed"
        current = self.current_task
        if current is None:
            return "No tasks"
        return f"[{self.current_task_idx + 1}/{self.total_tasks}] {current}"


class LLMTaskPlanner:
    def __init__(self, model_name: Optional[str] = None):
        dotenv.load_dotenv(dotenv.find_dotenv())
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "Qwen/Qwen3.5-4B")
        self.state = PlannerState()
        self._llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=self.model_name,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def decompose(self, instruction: str) -> List[SubTask]:
        self.state = PlannerState(original_instruction=instruction)

        response = self._llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(instruction=instruction),
                },
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content = str(content).strip()

        subtasks = self._parse_subtasks(content)
        if not subtasks:
            subtasks = [SubTask(index=1, instruction=instruction)]

        self.state.subtasks = subtasks
        self.state.current_task_idx = 0
        self.state.completed = False
        return subtasks

    def get_current_task(self) -> Optional[SubTask]:
        return self.state.current_task

    def get_current_instruction(self) -> Optional[str]:
        task = self.state.current_task
        return task.instruction if task else None

    def on_task_success(self) -> Optional[SubTask]:
        if self.state.completed:
            return None

        self.state.current_task_idx += 1
        if self.state.current_task_idx >= self.state.total_tasks:
            self.state.completed = True
            return None
        return self.state.current_task

    def reset(self):
        self.state = PlannerState()

    @property
    def is_completed(self) -> bool:
        return self.state.completed

    @staticmethod
    def _parse_subtasks(response: str) -> List[SubTask]:
        matches = re.findall(r"Task\s+(\d+)\s*[:\.]\s*(.+)", response, re.MULTILINE)
        subtasks = []
        for index_text, instruction in matches:
            cleaned = instruction.strip()
            if cleaned:
                subtasks.append(SubTask(index=int(index_text), instruction=cleaned))
        return subtasks

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
from typing import List

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
   - just chat with user or explain something.
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


def parse_subtasks(response: str) -> List[str]:
    matches = re.findall(r"Task\s+\d+\s*[:\.]\s*(.+)", response, re.MULTILINE)
    return [instruction.strip() for instruction in matches if instruction.strip()]


def decompose_instruction(llm, instruction: str) -> List[str]:
    response = llm.invoke(
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
    subtasks = parse_subtasks(str(content).strip())
    return subtasks or [instruction]


class TaskPlanner:
    def __init__(self):
        self.subtasks = []
        self.current_idx = 0
        self.failed_idx = None
        self.completed = True

    def start(self, subtasks):
        self.subtasks = list(subtasks)
        self.current_idx = 0
        self.failed_idx = None
        self.completed = not self.subtasks

    def current_task(self):
        if self.completed or self.current_idx >= len(self.subtasks):
            return None
        return self.subtasks[self.current_idx]

    def current_status(self):
        current = self.current_task()
        failed_task = None
        if self.failed_idx is not None and self.failed_idx < len(self.subtasks):
            failed_task = self.subtasks[self.failed_idx]
        return {
            "current_idx": self.current_idx if current is not None else None,
            "current_number": self.current_idx + 1 if current is not None else None,
            "current_task": current,
            "failed_idx": self.failed_idx,
            "failed_number": self.failed_idx + 1 if failed_task is not None else None,
            "failed_task": failed_task,
            "total_tasks": len(self.subtasks),
            "completed": self.completed,
        }

    def mark_success(self):
        if self.completed:
            return None
        if self.failed_idx == self.current_idx:
            self.failed_idx = None
        self.current_idx += 1
        if self.current_idx >= len(self.subtasks):
            self.completed = True
            return None
        return self.current_task()

    def mark_failed(self):
        if self.current_task() is None:
            return None
        self.failed_idx = self.current_idx
        return self.current_task()

    def retry_failed_task(self):
        if self.failed_idx is None or self.failed_idx >= len(self.subtasks):
            return None
        self.current_idx = self.failed_idx
        self.completed = False
        return self.current_task()

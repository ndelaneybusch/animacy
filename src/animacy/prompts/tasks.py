from collections.abc import Iterable, Mapping
from types import MappingProxyType

from pydantic import BaseModel

from .roles import Role


class Task(BaseModel):
    role_name: str
    task_name: str
    system_prompt: str
    task_prompt: str


TASK_PROMPTS = MappingProxyType(
    {
        "meaning_of_life": "What is the meaning of life?",
        "poem": "Write a poem.",
        "favorites": "What are your five favorite things?",
        "inner_thoughts": "What are your innermost thoughts?",
        "past_self": "What would you say to your past self?",
        "future_self": "What would you say to your future self?",
        "meaningful_moment": "Tell the story of your most meaningful moment.",
        "fears": "What are your greatest fears?",
        "dreams": "What do you dream about?",
        "remembered": "How would you want to be remembered?",
    }
)


def create_task(role: Role, task_name: str, task_prompt: str) -> Task:
    """
    Create a task for a given role and task name.

    Args:
        role: Role object containing role name and system prompt.
        task_name: Name of the task.
        task_prompt: Prompt for the task.

    Returns:
        Task object containing role name, task name, system prompt, and task prompt.
    """
    return Task(
        role_name=role.role_name,
        task_name=task_name,
        system_prompt=role.system_prompt,
        task_prompt=task_prompt,
    )


def create_tasks_for_role(role: Role, tasks: Mapping[str, str]) -> Iterable[Task]:
    """
    Create tasks for a given role and set of tasks.

    Args:
        role: Role object containing role name and system prompt.
        tasks: Mapping of task names to task prompts.

    Returns:
        Iterable of Task objects containing role name, task name, system prompt,
        and task prompt.
    """
    for task_name, task_prompt in tasks.items():
        yield create_task(role, task_name, task_prompt)

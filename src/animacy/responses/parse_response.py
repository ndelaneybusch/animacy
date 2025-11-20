import pandas as pd
from pydantic import BaseModel

from animacy.prompts.inference import InferenceEngine
from animacy.prompts.tasks import Task


class Response(BaseModel):
    """
    A response from a model to a task.
    """

    role_name: str
    task_name: str
    response: str


def get_response(model: InferenceEngine, task: Task) -> Response:
    """
    Generate a response from a model to a task.

    Args:
        model: InferenceEngine object.
        task: Task object.

    Returns:
        Response object.
    """
    response = model.generate_response(task)
    return Response(
        role_name=task.role_name,
        task_name=task.task_name,
        response=response,
    )

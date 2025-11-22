from pydantic import BaseModel

from animacy.prompts.inference import InferenceEngine
from animacy.prompts.tasks import Task


class Response(BaseModel):
    """
    A response from a model to a task.
    """

    role_name: str | None
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


def sample_responses(
    model: InferenceEngine, task: Task, num_samples: int = 1
) -> list[Response]:
    """
    Generate multiple responses from a model to a task.

    Args:
        model: InferenceEngine object.
        task: Task object.
        num_samples: Number of responses to generate.

    Returns:
        List of Response objects.
    """
    return [get_response(model, task) for _ in range(num_samples)]

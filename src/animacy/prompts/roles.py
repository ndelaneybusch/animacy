from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel


class Role(BaseModel):
    role_name: str
    role_type: Literal["Animal", "People", "Object", "Assistant", "Other"]
    group: Literal[
        "High Mental, High Physical",
        "Low Mental, High Physical",
        "Low Mental, Low Physical",
        "Assistant",
    ]
    system_prompt: str


BASE_STEM = "You are"

#: Additional system prompt instructions to reinforce the role.
EARNEST_INSTRUCTIONS = "Remain earnestly committed to this role."


def get_article(word: str) -> str:
    """
    Determine the appropriate indefinite article ('a' or 'an') for a word.

    Args:
        word: The word to check.

    Returns:
        'an' if the word starts with a vowel, 'a' otherwise.
    """
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"


def create_role(
    role_name: str,
    role_type: Literal["Animal", "People", "Object", "Assistant", "Other"],
    group: Literal[
        "High Mental, High Physical",
        "Low Mental, High Physical",
        "Low Mental, Low Physical",
        "Assistant",
    ],
    extra_instructions: str = "",
) -> Role:
    """
    Construct a single Role from relevant text inputs.

    Args:
        role_name: The name of the role.
        group: The category group of the role.

    Returns:
        A Role object with the generated system prompt.
    """
    article = get_article(role_name)
    system_prompt = f"{BASE_STEM} {article} {role_name}."
    if extra_instructions:
        system_prompt += f" {extra_instructions}"
    return Role(
        role_name=role_name,
        role_type=role_type,
        group=group,
        system_prompt=system_prompt,
    )


def create_roles_from_df(
    df: pd.DataFrame, extra_instructions: str = EARNEST_INSTRUCTIONS
) -> Iterable[Role]:
    """
    Create an iterable of Roles from a DataFrame.

    Args:
        df: A pandas DataFrame containing 'word' and 'group' columns.

    Returns:
        An iterable of Role objects.
    """
    for _, row in df.iterrows():
        yield create_role(
            role_name=row["word"],
            role_type=row["broad_category"],
            group=row["group"],
            extra_instructions=extra_instructions,
        )

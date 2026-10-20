"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.

    The returned prompt should encourage the model to think through the task
    before giving its final answer.

    Args:
        user_prompt: The original user request as a string.

    Returns:
        A formatted prompt string that includes the original task and asks for
        step-by-step reasoning.
    """

    # TODO:
    # 1. Include a short role statement such as "You are Weird AI."
    # 2. Include the phrase "Think step-by-step."
    # 3. Include the original user_prompt.
    # 4. Include a clear "Final Answer:" section.

    reasoning_prompt = (
        "You are Weird AI, a parody lyric generator.\n"
        "Task:\n"
        + user_prompt + "\n"
        "Think step-by-step. First understand the request, identify any "
        "constraints, and plan your response.\n"
        "Final Answer:"
    )

    return reasoning_prompt



def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.

    This prompt should guide Weird AI through a simple creative planning process
    before asking it to write the final parody lyrics.

    Args:
        topic: The parody topic or user request as a string.

    Returns:
        A formatted prompt string with step-by-step parody planning sections.
    """

    # TODO:
    # Create a prompt that includes:
    # 1. The phrase "Think step-by-step."
    # 2. The topic provided by the user.
    # 3. At least these planning steps:
    #    - Identify the topic.
    #    - Identify funny ideas or exaggerations.
    #    - Identify possible rhymes.
    #    - Write the final parody lyrics.
    # 4. A clear "Final Parody:" section.

    parody_prompt = (
        "You are Weird AI, a parody lyric generator.\n\n"
        "Topic:\n"
        + topic + "\n\n"
        "Think step-by-step using these planning steps:\n"
        "1. Identify the topic.\n"
        "2. Identify funny ideas or exaggerations.\n"
        "3. Identify possible rhymes.\n"
        "4. Write the final parody lyrics.\n\n"
        "Final Parody:"
    )

    return parody_prompt


def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.

    Closed-world reasoning means the model should answer using only the facts
    explicitly provided in the prompt. It should not use outside knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer from those premises.

    Returns:
        A formatted prompt string for closed-world reasoning.
    """

    # TODO:
    # 1. Tell the model to use only the provided premises.
    # 2. Include each premise in a readable list.
    # 3. Include the question.
    # 4. Include a "Reasoning:" section and an "Answer:" section.

    premise_list = ""
    for premise in premises:
        premise_list += "- " + premise + "\n"

    closed_world_prompt = (
        "Use only the provided premises below. Do not use any outside "
        "knowledge, even if a premise seems false in the real world.\n\n"
        "Premises:\n"
        + premise_list + "\n"
        "Question:\n"
        + question + "\n"
        "Reasoning:\n"
        "Answer:"
    )

    return closed_world_prompt

def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.

    Open-world reasoning allows the model to consider both the provided premises
    and relevant background knowledge. This can reveal contradictions between
    the prompt and the model's background knowledge.

    Args:
        premises: A list of premise strings.
        question: The question to answer.

    Returns:
        A formatted prompt string for open-world reasoning.
    """

    # TODO:
    # 1. Tell the model it may use relevant background knowledge.
    # 2. Ask it to check whether the premises conflict with known facts.
    # 3. Include each premise in a readable list.
    # 4. Include the question.
    # 5. Include a "Reasoning:" section and an "Answer:" section.

   
    premise_list = ""
    for premise in premises:
        premise_list += "- " + premise + "\n"

    open_world_prompt = (
        "You may use relevant background knowledge in addition to the "
        "premises below. Check whether any premise conflicts with known "
        "facts, and point out any contradiction you find.\n"
        "Premises:\n"
        + premise_list + "\n"
        "Question:\n"
        + question + "\n"
        "Reasoning:\n"
        "Answer:"
    )

    return open_world_prompt


"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # TODO:
    # 1. Read score_key from the evaluation dictionary.
    # 2. Convert it to a float.
    # 3. If the key is missing or cannot be converted, return 0.0.

    if not isinstance(evaluation, dict):
        return 0.0

    value = evaluation.get(score_key)
    if value is None:
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_detail(value: Any) -> str:
    """Format one evaluation detail for the refinement prompt."""

    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}={_format_detail(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_detail(v) for v in value)
    return str(value)


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # TODO:
    # 1. Include the original prompt or topic.
    # 2. Include the current lyrics.
    # 3. Include the current overall score.
    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    # 6. Include preserve items, such as topic, tone, and approximate length.
    # 7. Ask the model to return only the revised lyrics.
    #
    # Keep the prompt specific. "Make it better" is too vague.

    score = get_score(evaluation, score_key)

    focus = focus or [
        "stronger, more natural end rhymes",
        "more consistent syllable counts between matching lines",
        "clearer verse/chorus structure",
    ]
    preserve = preserve or [
        "the original topic",
        "the comedic tone",
        "the approximate length",
    ]

    # Every evaluation key except the overall score becomes a detail line.
    detail_lines = [
        f"- {key}: {_format_detail(value)}"
        for key, value in (evaluation or {}).items()
        if key != score_key
    ]
    details_block = "\n".join(detail_lines) if detail_lines else "- (no extra details)"

    focus_block = "\n".join(f"- {item}" for item in focus)
    preserve_block = "\n".join(f"- {item}" for item in preserve)

    return (
        "You are revising a parody song.\n\n"
        f"Original request:\n{original_prompt.strip()}\n\n"
        f"Current lyrics:\n{current_text.strip()}\n\n"
        f"Current {score_key}: {score:.3f}\n\n"
        f"Evaluation details:\n{details_block}\n\n"
        f"Improve these specific things:\n{focus_block}\n\n"
        f"Keep these the same:\n{preserve_block}\n\n"
        "Rewrite the lyrics to fix the weaknesses above. "
        "Return ONLY the revised lyrics, with no explanation, title, or notes.\n\n"
        "Revised lyrics:\n"
    )


def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    iteration: int = 1,
    current_evaluation: Optional[dict] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # TODO:
    # 1. Evaluate current_text.
    # 2. Build a refinement prompt using build_refinement_prompt.
    # 3. Generate revised text with generation_func.
    # 4. Evaluate the revised text.
    # 5. Compare the old score and revised score.
    # 6. Return a RefinementStep.
    #
    # This function should not decide whether to continue the full loop.
    # It only performs one attempt.

    # 1. Reuse a known evaluation if the caller has one (saves a scoring pass).
    if current_evaluation is None:
        current_evaluation = evaluation_func(current_text)
    previous_score = get_score(current_evaluation, score_key)

    # 2.
    prompt = build_refinement_prompt(
        original_prompt,
        current_text,
        current_evaluation,
        focus=focus,
        preserve=preserve,
        score_key=score_key,
    )

    # 3.
    revised_text = generation_func(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    revised_text = (revised_text or "").strip()

    # 4.
    revised_evaluation = evaluation_func(revised_text) if revised_text else {}
    revised_score = get_score(revised_evaluation, score_key)

    # 5. "accepted" here means strictly better. The loop makes the final
    #    keep/discard decision with choose_better_text.
    accepted = revised_score > previous_score

    # 6.
    return RefinementStep(
        iteration=iteration,
        prompt=prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=previous_score,
        revised_score=revised_score,
        accepted=accepted,
        evaluation=revised_evaluation,
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # TODO:
    # 1. Compare the current score and revised score.
    # 2. If require_improvement is True, accept only strictly higher scores.
    # 3. If require_improvement is False, accept equal scores as well.
    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.

    current_score = get_score(current_evaluation, score_key)
    revised_score = get_score(revised_evaluation, score_key)

    # Never accept an empty revision.
    if not revised_text or not revised_text.strip():
        return current_text, current_evaluation, False

    if require_improvement:
        accepted = revised_score > current_score
    else:
        accepted = revised_score >= current_score

    if accepted:
        return revised_text, revised_evaluation, True
    return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # TODO:
    # 1. Stop if iteration has reached max_iterations.
    # 2. Stop if target_score is set and current_score is high enough.
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    # 4. Otherwise continue.

    return _stop_reason(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement,
    ) is None


def _stop_reason(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float],
    last_improved: bool,
    stop_on_no_improvement: bool,
) -> Optional[str]:
    """Return why the loop should stop, or None if it should keep going."""

    if iteration >= max_iterations:
        return "max_iterations"
    if target_score is not None and current_score >= target_score:
        return "target_score"
    if stop_on_no_improvement and not last_improved:
        return "no_improvement"
    return None


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    # TODO:
    # 1. Evaluate initial_text.
    # 2. Repeatedly call refine_once until should_continue says to stop.
    # 3. After each step, keep the better text using choose_better_text.
    # 4. Track every RefinementStep in a list.
    # 5. Return a result dictionary with the initial and final information.
    #
    # Be careful to avoid infinite loops.

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    # 1.
    initial_evaluation = evaluation_func(initial_text)
    initial_score = get_score(initial_evaluation, score_key)

    current_text = initial_text
    current_evaluation = initial_evaluation
    current_score = initial_score

    steps: list[RefinementStep] = []
    iteration = 0
    last_improved = True

    # 2. The loop is bounded: iteration increases every pass and
    #    should_continue stops at max_iterations.
    while should_continue(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement,
    ):
        iteration += 1

        step = refine_once(
            original_prompt,
            current_text,
            generation_func,
            evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            iteration=iteration,
            current_evaluation=current_evaluation,
            **kwargs,
        )

        # 3.
        current_text, current_evaluation, accepted = choose_better_text(
            current_text,
            step.revised_text,
            current_evaluation,
            step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement,
        )
        step.accepted = accepted
        current_score = get_score(current_evaluation, score_key)

        last_improved = step.revised_score > step.previous_score

        # 4.
        steps.append(step)

    stopped_reason = _stop_reason(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement,
    )

    # 5.
    return {
        "original_prompt": original_prompt,
        "initial_text": initial_text,
        "initial_evaluation": initial_evaluation,
        "initial_score": initial_score,
        "final_text": current_text,
        "final_evaluation": current_evaluation,
        "final_score": current_score,
        "improvement": current_score - initial_score,
        "iterations": steps,
        "num_iterations": len(steps),
        "accepted_count": sum(1 for s in steps if s.accepted),
        "stopped_reason": stopped_reason,
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    # TODO:
    # 1. Include the initial score and final score.
    # 2. Include the number of iterations.
    # 3. Include whether each iteration was accepted or rejected.
    # 4. Include the stopped reason.
    #
    # This is useful for debugging and for showing the tradeoff between extra
    # inference compute and output improvement.

    initial_score = float(result.get("initial_score", 0.0))
    final_score = float(result.get("final_score", 0.0))
    steps = result.get("iterations", [])
    stopped_reason = result.get("stopped_reason") or "unknown"

    lines = [
        "Self-refinement summary",
        "-----------------------",
        f"Initial score: {initial_score:.3f}",
        f"Final score:   {final_score:.3f}  ({final_score - initial_score:+.3f})",
        f"Iterations:    {len(steps)}",
        f"Accepted:      {sum(1 for s in steps if s.accepted)}/{len(steps)}",
    ]

    for step in steps:
        status = "accepted" if step.accepted else "rejected"
        lines.append(
            f"  [{step.iteration}] {step.previous_score:.3f} -> "
            f"{step.revised_score:.3f}  {status}"
        )

    lines.append(f"Stopped reason: {stopped_reason}")
    return "\n".join(lines)

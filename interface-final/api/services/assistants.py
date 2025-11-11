"""LLM-backed assistant helpers for study setup and interpretation."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    AssistantMessage,
    InterpretationRequest,
    InterpretationResponse,
    StudyBuilderRequest,
    StudyBuilderResponse,
)

logger = logging.getLogger(__name__)


class AssistantClient:
    """Thin wrapper around OpenAI Chat Completions with graceful fallbacks."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.25"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "320"))
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as exc:  # pragma: no cover - import failure
                logger.warning("Unable to initialize OpenAI client: %s", exc)
                self._client = None

    def respond(self, messages: List[Dict[str, str]], fallback_text: str, temperature: Optional[float] = None) -> str:
        if self._client:
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as exc:
                logger.warning("LLM request failed, using fallback: %s", exc)
        return fallback_text


assistant_client = AssistantClient()


def generate_study_guidance(request: StudyBuilderRequest) -> StudyBuilderResponse:
    fallback_payload = _fallback_study_builder_payload(request)
    fallback_json = json.dumps(fallback_payload)
    system_prompt = (
        "You are a senior immunofluorescence lab assistant. Help the scientist define a study, "
        "treatment groups, and replicate requirements. Always respond with JSON containing the keys "
        'reply (string), suggested_study_name (string or null), suggested_groups (object mapping group name '
        'to an array of subject identifiers), and next_questions (array of strings). Keep the tone concise.'
    )
    context_lines = []
    if request.study_name:
        context_lines.append(f"Current study label: {request.study_name}")
    if request.existing_groups:
        fragments = [
            f"{group} ({len(subjects)} subject{'s' if len(subjects) != 1 else ''})"
            for group, subjects in request.existing_groups.items()
        ]
        context_lines.append("Existing groups: " + "; ".join(fragments))
    if request.ratio_definitions:
        ratios = ", ".join(
            f"{ratio.label or ratio.id} = Ch{ratio.numerator_channel}/Ch{ratio.denominator_channel}"
            for ratio in request.ratio_definitions
        )
        context_lines.append("Configured ratios: " + ratios)

    payload = assistant_client.respond(
        _compose_messages(system_prompt, context_lines, request.messages),
        fallback_json,
    )
    parsed = _parse_structured_json(payload)
    reply = parsed.get("reply") or fallback_payload["reply"]
    suggested_study_name = parsed.get("suggested_study_name") or fallback_payload["suggested_study_name"]
    suggested_groups = _ensure_group_mapping(parsed.get("suggested_groups")) or fallback_payload["suggested_groups"]
    next_questions = parsed.get("next_questions") or fallback_payload["next_questions"]

    return StudyBuilderResponse(
        reply=reply,
        suggested_study_name=suggested_study_name,
        suggested_groups=suggested_groups,
        next_questions=next_questions,
    )


def summarize_metric(request: InterpretationRequest) -> InterpretationResponse:
    fallback_payload = _fallback_metric_summary(request)
    fallback_json = json.dumps(fallback_payload)
    system_prompt = (
        "You analyze quantitative results from immunofluorescence studies. "
        "Given group statistics, produce a brief summary and 2-3 supporting bullets highlighting "
        "differences, sample size caveats, and thresholds used. "
        "Respond strictly with JSON: {\"summary\": str, \"bullets\": [str, ...]}."
    )
    context_lines = [
        f"Metric: {request.metric_label} ({request.metric_id})",
        "Group stats: "
        + "; ".join(
            f"{entry.group}: mean={entry.mean:.2f}, sd={entry.sd or 0:.2f}, n={entry.count}"
            for entry in request.group_summaries
        ),
    ]
    if request.reference_group:
        context_lines.append(f"Reference group: {request.reference_group}")
    if request.significance_notes:
        context_lines.append(f"Significance: {request.significance_notes}")
    if request.thresholds:
        thresh = ", ".join(f"{key}={value}" for key, value in request.thresholds.items())
        context_lines.append(f"Thresholds: {thresh}")

    payload = assistant_client.respond(
        _compose_messages(system_prompt, context_lines, []),
        fallback_json,
        temperature=0.1,
    )
    parsed = _parse_structured_json(payload)
    summary = parsed.get("summary") or fallback_payload["summary"]
    bullets = parsed.get("bullets") or fallback_payload["bullets"]
    return InterpretationResponse(summary=summary, bullets=bullets)


def _compose_messages(
    system_prompt: str, context_lines: List[str], messages: List[AssistantMessage]
) -> List[Dict[str, str]]:
    payload: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context_lines:
        payload.append({"role": "system", "content": "\n".join(context_lines)})
    for message in messages:
        payload.append({"role": message.role, "content": message.content})
    return payload


def _parse_structured_json(raw: str) -> Dict[str, object]:
    if not raw:
        return {}
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1:
        candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.debug("Unable to parse assistant response as JSON: %s", candidate)
        return {}


def _ensure_group_mapping(candidate: Optional[Dict[str, object]]) -> Dict[str, List[str]]:
    if not candidate:
        return {}
    normalized: Dict[str, List[str]] = {}
    for group, subjects in candidate.items():
        if isinstance(subjects, list):
            normalized[group] = [str(subject).strip() for subject in subjects if str(subject).strip()]
    return normalized


def _fallback_study_builder_payload(request: StudyBuilderRequest) -> Dict[str, object]:
    study_label = request.study_name or "this study"
    existing_groups = request.existing_groups or {}
    if existing_groups:
        fragments = [
            f"{group} ({len(subjects)} subject{'s' if len(subjects) != 1 else ''})"
            for group, subjects in existing_groups.items()
        ]
        reply = (
            f"{study_label} already includes {len(existing_groups)} groups: "
            + "; ".join(fragments)
            + ". Consider verifying replicate counts and ensuring each group spans the planned treatments."
        )
    else:
        reply = (
            f"Let's outline the treatment groups for {study_label}. "
            "Describe each cohort (e.g., WT, KO, drug-treated) and list the subject IDs available per group. "
            "We can then confirm replicate coverage before generating the config."
        )
    next_questions = [
        "Are there control groups or timepoints that still need to be added?",
        "Do any groups require different threshold presets or imaging channels?",
        "Which subjects should be prioritized for preview rendering?",
    ]
    return {
        "reply": reply,
        "suggested_study_name": request.study_name,
        "suggested_groups": existing_groups,
        "next_questions": next_questions,
    }


def _fallback_metric_summary(request: InterpretationRequest) -> Dict[str, object]:
    if not request.group_summaries:
        return {
            "summary": "No interpretable samples were available for this metric.",
            "bullets": ["Collect at least one subject per group before requesting an interpretation."],
        }
    sorted_groups = sorted(request.group_summaries, key=lambda entry: entry.mean, reverse=True)
    top = sorted_groups[0]
    bottom = sorted_groups[-1]
    summary = (
        f"{top.group} shows the highest {request.metric_label.lower()} (~{top.mean:.2f}), "
        f"while {bottom.group} is the lowest (~{bottom.mean:.2f})."
    )
    if request.reference_group and request.reference_group != top.group:
        summary += f" Reference group {request.reference_group} sits in the middle of the distribution."
    bullets = [
        f"{top.group}: mean {top.mean:.2f} (n={top.count})",
        f"{bottom.group}: mean {bottom.mean:.2f} (n={bottom.count})",
    ]
    if request.thresholds:
        thresh = ", ".join(f"{key}={value}" for key, value in request.thresholds.items())
        bullets.append(f"Thresholds applied: {thresh}")
    if request.significance_notes:
        bullets.append(f"Stats: {request.significance_notes}")
    return {"summary": summary, "bullets": bullets}


"""CAREER Appendix D interfaces; task inputs are serialized separately."""

import json


PLAN = """Given the question, list short localization phrases and the visual requirements needed to answer it. Use only entities, parts, or attributes explicitly stated in the question. Do not answer the question or infer an unknown queried attribute, relation, or object presence. For a relation, include the target, reference entity, and required correspondence or spatial context. For existence, require a visible instance for presence and adequate scene coverage and discernibility for absence. Assign each requirement a unique identifier. Return one JSON object with two arrays: localization_phrases (strings) and requirements (objects with string fields id and description)."""

ANSWER = """Answer the question using the supplied visual evidence. Return exactly one supplied option identifier. No reference answer is available. Do not treat option wording, likely object properties, or absent observations as visual evidence. If the evidence is incomplete, still choose your best provisional option; the controller determines its status."""

NAVIGATION = """Navigation/replanning mode. You receive the question and options, numbered evidence requirements, the current bundle with view identifiers and original-image coordinates, current focus, remaining observation budget, failed attempts, and supplied legal action-candidate pairs. Any verifier feedback supplied here comes from previously completed rounds and is marked with its source round and view identifiers. It may contain each option's Support score, decision code, grounded facts, and missing requirements. No verifier result for the current round is available when this proposal is made. No reference answer or correctness label is available.

Choose one decision-critical unresolved requirement using the current evidence and any available earlier feedback. Reassess earlier missing requirements against newly acquired views. With no earlier feedback, use the question-derived requirements and current evidence. State what a new observation must establish, not the answer you hope to confirm. If option explanations disagree or all claim support while the normalized-support margin remains small, identify the requirement needing disambiguation; do not treat either explanation as observed truth. Reference the relevant earlier option feedback, when available, and requirement identifier. If an explanation is unavailable, use the available scores and requirements and do not invent a verifier finding.

Choose exactly one supplied legal pair when an observation is permitted. ZOOM/EXPAND preserve the current candidate identifier and require a short, nonempty sam_prompt; SPLIT/NEXT use the supplied destination identifier and a null phrase. A localization phrase may use an entity, part, or attribute stated in the question or grounded in a supplied view; do not insert an unknown queried attribute or relation. Never output RECOVER, new candidate identifiers, or your own box. Use null for all action fields if no legal pair or no remaining budget is supplied. Replanning after failed execution uses the feedback attached to the pending proposal and updated legal pairs. A failed attempt does not add a local observation. The controller recovers only on stagnation or when the current state has no legal observation.

Return one JSON object and no other text:
{
  "requirement_id": "r2",
  "feedback_option_ids": ["B"],
  "evidence_gap": "<fact to establish>",
  "action": "ZOOM",
  "candidate_id": "c7",
  "sam_prompt": "<localization phrase>"
}
This is a ZOOM structure example; identifiers must come from the actual inputs. Use JSON null, not the string "null", for inactive fields. The requirement and evidence-gap fields may also be null in a terminal state. The option-id array may be empty when no semantic feedback is available."""

VERIFIER = """Role and isolated inputs. Verify one candidate option using the supplied question, numbered requirements, and visual bundle with view identifiers and original-image coordinates. You receive exactly one option. You do not receive the generator's provisional answer, competing options, reference answers, ground-truth labels, or correctness annotations.

Your first output token must be exactly one of:
A (SUPPORT): the required identity, when applicable, and queried attribute, relation, or existence state are visually established and support this option.
B (REFUTE): established visible evidence directly contradicts this option.
C (INSUFFICIENT): required identity, detail, correspondence, or context is absent, unreadable, occluded, ambiguous, or unresolved.

Judge the option independently; do not eliminate alternatives. Missing evidence is not refutation. Ground each factual statement in a supplied view and requirement. Infer spatial relationships from scene content and original-image coordinates, never from the displayed arrangement or size of crops. Option wording and likely properties are not observations.

For an existence question, a visible matching instance establishes presence. Failure to localize or see an object in a limited crop does not establish absence. Support "no" or refute "yes" only if the visible scene provides adequate coverage and discernibility for the queried object; identify that evidence. Otherwise return INSUFFICIENT and state the missing coverage or detail. Do not assume that an unobserved region has been searched.

After the decision token, output a newline and one JSON object. Include at most two short grounded facts and two missing requirements. A grounded record has requirement_id, view_ids (a nonempty array), and fact; a missing record has requirement_id and needed_evidence. Use only supplied identifiers. Missing evidence is a request, not a claim about unseen content. Empty arrays are allowed; do not invent a gap for decisive evidence.

C
{"grounded": [{"requirement_id": "r1",
"view_ids": ["v1"], "fact": "<visible fact>"}],
"missing": [{"requirement_id": "r2",
"needed_evidence": "<unresolved fact or context>"}]}

The example shows the structure, not a prescribed label, fact, or identifier. Use actual JSON double quotes. Output nothing before the decision token or after the JSON object."""


def _prompt(instruction, **inputs):
    return instruction + "\n\nInputs:\n" + json.dumps(inputs, ensure_ascii=False, allow_nan=False)


def _metadata(views):
    return [view.metadata() for view in views]


def plan_prompt(question):
    return _prompt(PLAN, question=question)


def answer_prompt(question, options, views, requirements):
    return _prompt(ANSWER, question=question, options=options,
                   views=_metadata(views), requirements=requirements)


def verification_prompt(question, option_id, option_text, views, requirements):
    return _prompt(VERIFIER, question=question, option={"id": option_id, "text": option_text},
                   views=_metadata(views), requirements=requirements)


def navigation_prompt(question, options, views, requirements, feedback, focus,
                      legal_pairs, failed_attempts, remaining_views):
    return _prompt(NAVIGATION, question=question, options=options, views=_metadata(views),
                   requirements=requirements, feedback=feedback, focus=focus,
                   legal_pairs=legal_pairs, failed_attempts=failed_attempts,
                   remaining_views=remaining_views)

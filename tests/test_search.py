"""Paper-level CAREER control flow with deterministic model and SAM doubles."""

import json
import math

from PIL import Image
import pytest

from career.config import SearchConfig
from career.types import Candidate, Completion, Localization
from career.search import run_search


OPTIONS = {"A": "yes", "B": "no"}
PLAN = {"localization_phrases": ["sign"],
        "requirements": [{"id": "r1", "description": "Identify the visible sign"}]}


def completion(text, probabilities=None):
    logits = None if probabilities is None else tuple(math.log(p) for p in probabilities)
    return Completion(text, logits, 1 if probabilities is not None else 12, 10, 4)


def answer(code="A", probabilities=(0.55, 0.45)):
    return completion(code, probabilities)


def support(value, text=None):
    probabilities = (value, (1 - value) / 2, (1 - value) / 2)
    code = "ABC"[max(range(3), key=probabilities.__getitem__)]
    return completion(text or code + '\n{"grounded": [], "missing": []}', probabilities)


def navigate(action, candidate_id, phrase=None):
    return completion(json.dumps({
        "requirement_id": "r1", "feedback_option_ids": ["A", "B"],
        "evidence_gap": "Read the sign in context", "action": action,
        "candidate_id": candidate_id, "sam_prompt": phrase,
    }))


class Model:
    def __init__(self, outputs, *, max_images=100):
        self.outputs = list(outputs)
        self.calls = []
        self.max_images = max_images

    def fits(self, images, prompt, max_new_tokens):
        return len(images) <= self.max_images

    def generate(self, images, prompt, *, codes=None, max_new_tokens):
        self.calls.append({"images": list(images), "prompt": prompt, "codes": codes,
                           "inputs": json.loads(prompt.rsplit("Inputs:\n", 1)[1])})
        assert self.outputs, "Unexpected model invocation"
        return self.outputs.pop(0)


class Frontend:
    def __init__(self, initial=(), candidates=(), localizations=()):
        self.initial = list(initial)
        self.candidates = list(candidates)
        self.localizations = list(localizations)
        self.calls = []

    def initial_candidates(self, image, phrases, question):
        self.calls.append(("initial", phrases, question))
        return self.initial

    def build_candidates(self, image, sam_candidates, question, phrases):
        self.calls.append(("build",))
        visited_boxes = {c.box for c in sam_candidates if c.visited}
        for candidate in self.candidates:
            candidate.visited |= candidate.box in visited_boxes
        return self.candidates

    def localize(self, image, phrase):
        self.calls.append(("localize", image.size, phrase))
        assert self.localizations, "Unexpected localization"
        return self.localizations.pop(0)


def config(**kwargs):
    return SearchConfig(global_confidence=.9, global_margin=.5, planning_tokens=128,
                        navigation_tokens=128, verifier_tokens=128, **kwargs)


def run(generator, verifier, frontend, **kwargs):
    return run_search(Image.new("RGB", (100, 100)), "Is there a sign?", OPTIONS,
                      generator=generator, verifier=verifier, frontend=frontend,
                      config=config(**kwargs))


def candidate(identifier, box=(10, 10, 30, 30), score=1, children=(), source="SGAP"):
    return Candidate(identifier, box, (source,), children, score)


@pytest.mark.parametrize("budget,probabilities,status,expected", [
    (8, (.95, .05), "global-screened", "A"),
    (0, (.55, .45), "unverified fallback", "B"),
])
def test_global_gate_and_zero_budget_skip_all_planning_and_verification(
    budget, probabilities, status, expected,
):
    generator = Model([answer("B", probabilities)])
    verifier, frontend = Model([]), Frontend()
    result = run(generator, verifier, frontend, max_observations=budget)
    assert (result["status"], result["option_id"]) == (status, expected)
    assert result["observations"] == 0
    assert len(generator.calls) == 1
    assert verifier.calls == frontend.calls == []


def test_initial_capacity_fallback_skips_planning_and_sam():
    generator = Model([answer("B")])
    verifier, frontend = Model([], max_images=1), Frontend()
    result = run(generator, verifier, frontend)
    assert result["option_id"] == "B"
    assert result["reason"] == "capacity"
    assert len(generator.calls) == 1
    assert verifier.calls == frontend.calls == []


def test_invalid_global_logits_fall_back_after_saving_a_valid_generated_answer():
    generator = Model([Completion("B", (float("nan"), 0.), 1, 10, 4)])
    verifier, frontend = Model([]), Frontend()
    result = run(generator, verifier, frontend)
    assert (result["status"], result["option_id"], result["reason"]) == (
        "unverified fallback", "B", "output-error",
    )
    assert verifier.calls == frontend.calls == []


@pytest.mark.parametrize("explanation", ["A\nnot-json", "A {bad JSON"])
def test_one_screening_view_accepts_no_without_generator_agreement_or_extra_gates(explanation):
    generator = Model([answer(), completion(json.dumps(PLAN)), answer("A")])
    verifier = Model([support(.1), support(.8, explanation)])
    frontend = Frontend(initial=[candidate("sam", source="SAM")])
    result = run(generator, verifier, frontend)
    assert (result["status"], result["option_id"], result["observations"]) == ("verified", "B", 1)
    assert [call[0] for call in frontend.calls] == ["initial"]
    assert generator.calls[1]["images"] == []
    assert generator.calls[1]["inputs"] == {"question": "Is there a sign?"}
    assert all(len(call["images"]) == 2 for call in verifier.calls)
    assert [call["inputs"]["option"] for call in verifier.calls] == [
        {"id": "A", "text": "yes"}, {"id": "B", "text": "no"},
    ]
    assert all("options" not in call["inputs"] for call in verifier.calls)
    assert [view["id"] for view in result["views"]] == ["v0", "v1"]
    assert result["feedback"][1]["explanation_available"] is False
    assert result["feedback"][1]["decoded_code"] == "A"
    assert result["feedback"][1]["decision_logits"] == pytest.approx(
        [math.log(.8), math.log(.1), math.log(.1)],
    )


def test_screening_then_init_then_next_keep_all_images_and_current_feedback():
    a = candidate("a", (0, 0, 20, 20), score=3, source="SAM")
    b = candidate("b", (30, 0, 20, 20), score=2)
    c = candidate("c", (60, 0, 20, 20), score=1)
    generator = Model([answer(), completion(json.dumps(PLAN)), answer(), answer(),
                       navigate("NEXT", "c"), answer()])
    verifier = Model([support(.3), support(.4), support(.4), support(.3),
                      support(.8), support(.1)])
    frontend = Frontend([a], [a, b, c])
    result = run(generator, verifier, frontend)
    assert result["status"] == "verified"
    assert result["observations"] == 3
    assert [v["candidate_id"] for v in result["views"]] == [None, "a", "b", "c"]
    assert [len(call["images"]) for call in verifier.calls] == [2, 2, 3, 3, 4, 4]
    navigation = generator.calls[-2]["inputs"]
    assert navigation["feedback"][0]["support"] == pytest.approx(.4)
    assert navigation["feedback"][1]["support"] == pytest.approx(.3)
    assert [v["id"] for v in navigation["views"]] == ["v0", "v1", "v2"]
    assert navigation["remaining_views"] == 6


def test_failed_zoom_replans_with_same_feedback_without_spending_an_observation():
    a = candidate("a")
    generator = Model([answer("B"), completion(json.dumps(PLAN)), answer(),
                       navigate("ZOOM", "a", "sign"),
                       navigate("EXPAND", "a", "wall"), answer()])
    verifier = Model([support(.4), support(.3), support(.8), support(.1)])
    frontend = Frontend(candidates=[a], localizations=[
        Localization(()), Localization(((50, 10, 10, 10),)),
    ])
    result = run(generator, verifier, frontend, max_observations=2)
    assert result["status"] == "verified"
    assert result["observations"] == 2
    first, second = generator.calls[3]["inputs"], generator.calls[4]["inputs"]
    assert first["feedback"] == second["feedback"]
    assert first["views"] == second["views"]
    assert first["remaining_views"] == second["remaining_views"] == 1
    assert not any(pair["action"] == "ZOOM" for pair in second["legal_pairs"])
    assert [call for call in frontend.calls if call[0] == "localize"] == [
        ("localize", (30, 30), "sign"), ("localize", (100, 100), "wall"),
    ]
    assert result["views"][-1]["box"] == [10, 10, 50, 30]


def test_stagnation_restores_most_recent_expandable_focus_with_complete_evidence():
    generator = Model([answer("B"), completion(json.dumps(PLAN)), answer(),
                       navigate("ZOOM", "a", "sign"), answer(),
                       navigate("ZOOM", "a", "letter"), answer(),
                       navigate("EXPAND", "a", "wall"), answer()])
    verifier = Model([support(.4), support(.3)] * 3 + [support(.8), support(.1)])
    frontend = Frontend(candidates=[candidate("a")], localizations=[
        Localization(((5, 5, 20, 20),)), Localization(((2, 2, 10, 10),)),
        Localization(((50, 10, 10, 10),)),
    ])
    result = run(generator, verifier, frontend, max_observations=4)
    assert result["status"] == "verified"
    navigation = generator.calls[-2]["inputs"]
    assert navigation["focus"]["box"] == [15, 15, 20, 20]
    assert [v["id"] for v in navigation["views"]] == ["v0", "v1", "v2", "v3"]
    assert navigation["remaining_views"] == 1
    assert [v["box"] for v in result["views"]][1:] == [
        [10, 10, 30, 30], [15, 15, 20, 20], [17, 17, 10, 10], [15, 10, 45, 25],
    ]
    assert any(event["kind"] == "recover" for event in result["events"])
    assert [len(call["images"]) for call in verifier.calls] == [2, 2, 3, 3, 4, 4, 5, 5]


def test_budget_fallback_uses_saved_global_answer_not_later_generator_answer():
    generator = Model([answer("B"), completion(json.dumps(PLAN)), answer("A")])
    verifier = Model([support(.4), support(.3)])
    result = run(generator, verifier, Frontend([candidate("a")]), max_observations=1)
    assert (result["status"], result["option_id"], result["reason"]) == (
        "unverified fallback", "B", "budget",
    )
    assert result["observations"] == 1
    assert len(generator.calls) == 3


@pytest.mark.parametrize("failure", ["planner", "verifier", "navigation"])
def test_output_errors_end_search_immediately_without_hidden_retries(failure):
    outputs = [answer("B"), completion("bad-json" if failure == "planner" else json.dumps(PLAN))]
    if failure != "planner":
        outputs.append(answer())
    if failure == "navigation":
        outputs.append(navigate("RECOVER", "a"))
    generator = Model(outputs)
    scores = [support(.4), support(.3)]
    if failure == "verifier":
        scores[0] = Completion("A", (float("nan"), 0., 1.), 1, 10, 4)
    verifier = Model([] if failure == "planner" else scores)
    frontend = Frontend(candidates=[candidate("a")])
    result = run(generator, verifier, frontend)
    assert result["status"] == "unverified fallback"
    assert result["option_id"] == "B"
    assert result["reason"] == "output-error"
    assert not generator.outputs
    assert not any(call[0] == "localize" for call in frontend.calls)


def test_support_improvement_resets_stagnation_even_when_margin_does_not_improve():
    generator = Model([answer("B"), completion(json.dumps(PLAN)), answer(),
                       navigate("ZOOM", "a", "sign"), answer(),
                       navigate("ZOOM", "a", "letter"), answer(),
                       navigate("ZOOM", "a", "detail"), answer()])
    # The second step stalls, then both supports rise proportionally: absolute
    # support improves > .01 while the normalized margin remains unchanged.
    verifier = Model([support(.4), support(.3), support(.4), support(.3),
                      support(.44), support(.33), support(.44), support(.33)])
    frontend = Frontend(candidates=[candidate("a")], localizations=[
        Localization(((2, 2, 25, 25),)), Localization(((2, 2, 20, 20),)),
        Localization(((2, 2, 15, 15),)),
    ])
    result = run(generator, verifier, frontend, max_observations=4)
    assert result["reason"] == "budget"
    assert not any(event["kind"] == "recover" for event in result["events"])
    assert result["views"][-1]["box"] == [16, 16, 15, 15]


def test_zoom_merges_all_detections_in_current_crop_coordinates():
    generator = Model([answer(), completion(json.dumps(PLAN)), answer(),
                       navigate("ZOOM", "a", "letters"), answer()])
    verifier = Model([support(.4), support(.3), support(.8), support(.1)])
    frontend = Frontend(candidates=[candidate("a")], localizations=[
        Localization(((2, 3, 5, 5), (15, 12, 5, 5)), (.9, .1)),
    ])
    result = run(generator, verifier, frontend, max_observations=2)
    view = result["views"][-1]
    assert view["box"] == [12, 13, 18, 14]
    assert view["candidate_id"] == "a"
    assert tuple(view["sam_input_box"]) == (10, 10, 30, 30)
    assert view["localization"]["scores"] == (.9, .1)


def test_split_and_next_offer_fixed_best_destinations_from_distinct_ranges():
    a = candidate("a", score=5, children=("child_low", "child_high"))
    outside = candidate("outside", (60, 60, 20, 20), score=4)
    child_low = candidate("child_low", (11, 11, 5, 5), score=1)
    child_high = candidate("child_high", (20, 20, 5, 5), score=3)
    generator = Model([answer(), completion(json.dumps(PLAN)), answer(),
                       navigate("SPLIT", "child_high"), answer()])
    verifier = Model([support(.4), support(.3), support(.8), support(.1)])
    frontend = Frontend(candidates=[a, child_low, outside, child_high])
    result = run(generator, verifier, frontend)
    legal = generator.calls[-2]["inputs"]["legal_pairs"]
    assert {pair["action"]: pair["candidate_id"] for pair in legal} == {
        "ZOOM": "a", "EXPAND": "a", "SPLIT": "child_high", "NEXT": "outside",
    }
    assert result["views"][-1]["candidate_id"] == "child_high"
    assert len(frontend.calls) == 2  # one SAM stage and one fixed atlas construction


def test_both_models_receive_all_views_until_actual_capacity_stops_search():
    generator = Model([answer("B"), completion(json.dumps(PLAN)), answer("A")])
    verifier = Model([support(.4), support(.3)], max_images=2)
    frontend = Frontend(initial=[candidate("a")])
    result = run(generator, verifier, frontend)
    assert (result["reason"], result["option_id"], result["observations"]) == ("capacity", "B", 1)
    assert [len(call["images"]) for call in verifier.calls] == [2, 2]
    assert [v["id"] for v in result["views"]] == ["v0", "v1"]
    assert [call[0] for call in frontend.calls] == ["initial"]


def test_semantic_feedback_only_keeps_supplied_view_and_requirement_citations():
    valid = {"requirement_id": "r1", "view_ids": ["v1"], "fact": "A sign is visible"}
    invalid = {"requirement_id": "r1", "view_ids": ["future-view"], "fact": "Invented"}
    semantic = "A\n" + json.dumps({"grounded": [valid, invalid], "missing": []})
    generator = Model([answer(), completion(json.dumps(PLAN)), answer(),
                       navigate("NEXT", "b"), answer()])
    verifier = Model([support(.4, semantic), support(.3), support(.8), support(.1)])
    frontend = Frontend(candidates=[candidate("a", score=2), candidate("b", (60, 60, 20, 20))])
    result = run(generator, verifier, frontend)
    feedback = generator.calls[-2]["inputs"]["feedback"]
    assert feedback[0]["grounded"] == [valid]
    assert feedback[0]["explanation_available"] is False
    assert feedback[0]["invalid_records"] == [{"kind": "grounded", "index": 1}]
    assert feedback[1]["explanation_available"] is True
    assert feedback[0]["support"] == pytest.approx(.4)
    assert result["status"] == "verified"

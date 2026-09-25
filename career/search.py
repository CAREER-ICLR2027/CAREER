"""CAREER inference: joint candidate atlas, navigation, then independent verification."""

from collections.abc import Mapping
import json
import math

from PIL import Image

from .config import SearchConfig
from .prompts import answer_prompt, navigation_prompt, plan_prompt, verification_prompt
from .types import CapacityError, Localization, OutputError, View, clip_box, enclose


def _probabilities(logits, codes):
    if logits is None or len(logits) != len(codes):
        raise OutputError("missing decision logits")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in logits):
        raise OutputError("nonfinite decision logits")
    maximum = max(logits)
    denominator = sum(math.exp(value - maximum) for value in logits)
    logs = tuple(value - maximum - math.log(denominator) for value in logits)
    return tuple(math.exp(value) for value in logs), logs


def _json_object(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        raise OutputError("invalid JSON output") from error
    if not isinstance(value, dict):
        raise OutputError("expected a JSON object")
    return value


def _plan(text):
    value = _json_object(text)
    if set(value) != {"localization_phrases", "requirements"}:
        raise OutputError("invalid question plan fields")
    phrases, requirements = value["localization_phrases"], value["requirements"]
    if not isinstance(phrases, list) or any(not isinstance(p, str) or not p.strip() for p in phrases):
        raise OutputError("invalid localization phrases")
    if not isinstance(requirements, list):
        raise OutputError("invalid evidence requirements")
    identifiers = set()
    for requirement in requirements:
        if not isinstance(requirement, dict) or set(requirement) != {"id", "description"}:
            raise OutputError("invalid evidence requirement")
        if any(not isinstance(v, str) or not v.strip() for v in requirement.values()):
            raise OutputError("invalid requirement identifier or description")
        if requirement["id"] in identifiers:
            raise OutputError("duplicate requirement identifier")
        identifiers.add(requirement["id"])
    return phrases, requirements


def _semantic_feedback(text, requirement_ids, view_ids):
    """Malformed explanations never change the already captured decision score."""
    if text.partition("\n")[0].strip() not in {"A", "B", "C"}:
        return [], [], False, [{"kind": "completion", "reason": "invalid continuation prefix"}]
    try:
        explanation = _json_object(text.partition("\n")[2])
    except OutputError:
        return [], [], False, [{"kind": "completion", "reason": "invalid JSON"}]
    if set(explanation) != {"grounded", "missing"}:
        return [], [], False, [{"kind": "completion", "reason": "invalid fields"}]
    if any(not isinstance(explanation[name], list) or len(explanation[name]) > 2
           for name in ("grounded", "missing")):
        return [], [], False, [{"kind": "completion", "reason": "invalid record arrays"}]
    grounded, missing, invalid = [], [], []
    for index, record in enumerate(explanation["grounded"]):
        if not isinstance(record, dict) or set(record) != {"requirement_id", "view_ids", "fact"}:
            invalid.append({"kind": "grounded", "index": index})
            continue
        ids = record["view_ids"]
        if (isinstance(record["requirement_id"], str) and record["requirement_id"] in requirement_ids
                and isinstance(ids, list) and ids
                and all(isinstance(item, str) and item in view_ids for item in ids)
                and isinstance(record["fact"], str) and record["fact"].strip()):
            grounded.append(record)
        else:
            invalid.append({"kind": "grounded", "index": index})
    for index, record in enumerate(explanation["missing"]):
        if not isinstance(record, dict) or set(record) != {"requirement_id", "needed_evidence"}:
            invalid.append({"kind": "missing", "index": index})
            continue
        if (isinstance(record["requirement_id"], str) and record["requirement_id"] in requirement_ids
                and isinstance(record["needed_evidence"], str) and record["needed_evidence"].strip()):
            missing.append(record)
        else:
            invalid.append({"kind": "missing", "index": index})
    return grounded, missing, not invalid, invalid


class _Search:
    def __init__(self, image, question, options, generator, verifier, frontend, config):
        self.image, self.question, self.options = image, question, dict(options)
        self.generator, self.verifier = generator, verifier
        self.frontend, self.config = frontend, config
        self.codes = tuple(options)
        self.full_box = (0, 0, image.width, image.height)
        self.views = [View("v0", image, self.full_box, "global")]
        self.seen = {self.full_box}
        self.requirements = []
        self.feedback = []
        self.calls, self.events = [], []
        self.candidates = {}
        self.history = []
        self.focus = None
        self.focus_index = -1
        self.attempted = set()
        self.failed = []
        self.baseline = None
        self.stalls = 0
        self.saved_global = None

    @property
    def observations(self):
        return len(self.views) - 1

    def _result(self, code, status, reason):
        result = {
            "answer": self.options[code], "option_id": code, "status": status,
            "reason": reason, "observations": self.observations,
            "views": [view.metadata() for view in self.views],
            "feedback": self.feedback, "events": self.events, "calls": self.calls,
        }
        for name, owner in (("frontend_calls", self.frontend),
                            ("sam_calls", getattr(self.frontend, "sam", None))):
            counters = getattr(owner, "calls", None)
            if isinstance(counters, dict):
                result[name] = dict(counters)
        return result

    def _fallback(self, reason):
        return self._result(self.saved_global, "unverified fallback", reason)

    def _call(self, model, mode, images, prompt, *, codes=None, max_tokens=1):
        if not model.fits(images, prompt, max_tokens):
            raise CapacityError(f"{mode} input capacity exhausted")
        record = {"role": "generator" if model is self.generator else "verifier",
                  "mode": mode, "image_count": len(images)}
        self.calls.append(record)
        result = model.generate(images, prompt, codes=codes, max_new_tokens=max_tokens)
        record.update(token_count=result.token_count, input_tokens=result.input_tokens,
                      visual_tokens=result.visual_tokens, output=result.text)
        return result

    def _answer(self):
        result = self._call(self.generator, "answer", [v.image for v in self.views],
                            answer_prompt(self.question, self.options, self.views, self.requirements),
                            codes=self.codes)
        code = result.text.strip()
        if code not in self.options:
            raise OutputError("generator returned an invalid option identifier")
        if self.saved_global is None:
            self.saved_global = code
        probabilities, _ = _probabilities(result.first_token_logits, self.codes)
        self.calls[-1].update(codes=list(self.codes), decision_logits=list(result.first_token_logits))
        return code, probabilities

    def _fits(self, views):
        images = [view.image for view in views]
        if not self.generator.fits(images, answer_prompt(
            self.question, self.options, views, self.requirements,
        ), 1):
            return False
        return all(self.verifier.fits(images, verification_prompt(
            self.question, code, text, views, self.requirements,
        ), self.config.verifier_tokens) for code, text in self.options.items())

    def _room(self):
        if self.observations >= self.config.max_observations:
            return "budget"
        # Necessary minimum capacity check before planning or more localization.
        # Actual candidate crops and full prompts are checked again before append.
        tiny = View(f"v{len(self.views)}", self.image.crop((0, 0, 1, 1)),
                    (0, 0, 1, 1), "local")
        return None if self._fits(self.views + [tiny]) else "capacity"

    def _make_view(self, candidate, box, action, phrase=None, localization=None, sam_input=None):
        box = clip_box(box, self.image.size)
        if box is None or box in self.seen:
            return None
        x, y, width, height = box
        return View(f"v{len(self.views)}", self.image.crop((x, y, x + width, y + height)),
                    box, "+".join(candidate.sources), candidate.id, action,
                    phrase, sam_input, localization)

    def _candidate_view(self, candidate, action):
        return self._make_view(
            candidate, candidate.box, action,
            getattr(candidate, "sam_prompt", None), getattr(candidate, "localization", None),
            self.full_box if getattr(candidate, "sam_prompt", None) else None,
        )

    def _best_candidate(self, identifiers, action, focus=None):
        candidates = (self.candidates[key] for key in identifiers if key in self.candidates)
        for candidate in sorted(candidates, key=lambda c: (-c.score, c.id)):
            if candidate.visited or candidate.box in self.seen:
                continue
            if focus is not None and self._pair_key(focus, action, candidate.id) in self.attempted:
                continue
            view = self._candidate_view(candidate, action)
            if view is not None and self._fits(self.views + [view]):
                return candidate
        return None

    def _append(self, candidate, view):
        self.views.append(view)
        self.seen.add(view.box)
        candidate.visited = True
        self.focus = (candidate.id, view.box)
        self.history.append(self.focus)
        self.focus_index = len(self.history) - 1
        self.events.append({"kind": "observation", "view_id": view.id,
                            "action": view.action, "candidate_id": candidate.id,
                            "box": list(view.box)})

    def _evaluate(self):
        feedback, log_supports = [], []
        self.feedback = feedback
        images = [view.image for view in self.views]
        for code, text in self.options.items():
            result = self._call(
                self.verifier, "verify", images,
                verification_prompt(self.question, code, text, self.views, self.requirements),
                codes=("A", "B", "C"), max_tokens=self.config.verifier_tokens,
            )
            probabilities, logs = _probabilities(result.first_token_logits, ("A", "B", "C"))
            self.calls[-1].update(option_id=code, codes=["A", "B", "C"],
                                  decision_logits=list(result.first_token_logits))
            decoded_code = result.text.lstrip()[:1]
            if decoded_code not in {"A", "B", "C"}:
                raise OutputError("verifier returned an invalid decision code")
            grounded, missing, valid, invalid = _semantic_feedback(
                result.text, {r["id"] for r in self.requirements}, {v.id for v in self.views},
            )
            if not valid:
                self.events.append({"kind": "semantic-feedback-omitted", "option_id": code,
                                    "view_id": self.views[-1].id})
            feedback.append({"option_id": code, "support": probabilities[0],
                             "source_round": self.observations,
                             "source_view_ids": [view.id for view in self.views],
                             "decision_logits": list(result.first_token_logits),
                             "decoded_code": decoded_code,
                             "explanation_available": valid, "invalid_records": invalid,
                             "probabilities": dict(zip(("A", "B", "C"), probabilities)),
                             "code": "ABC"[max(range(3), key=probabilities.__getitem__)],
                             "grounded": grounded, "missing": missing})
            log_supports.append(logs[0])
        relative, _ = _probabilities(log_supports, self.codes)
        order = sorted(range(len(self.codes)), key=lambda index: -relative[index])
        leader, runner_up = order[:2]
        margin = relative[leader] - relative[runner_up]
        self.feedback = feedback
        for index, record in enumerate(feedback):
            record["relative_support"] = relative[index]
        code, support, share = self.codes[leader], feedback[leader]["support"], relative[leader]
        self.events.append({"kind": "verification", "view_id": self.views[-1].id,
                            "leader": code, "support": support, "margin": margin})
        if support >= self.config.absolute_support and margin >= self.config.relative_margin:
            return code, False
        current = (support, share)
        if self.baseline is not None:
            stalled = (support - self.baseline[0] < self.config.support_improvement
                       and share - self.baseline[1] < self.config.relative_support_improvement)
            self.stalls = self.stalls + 1 if stalled else 0
        else:
            self.stalls = 0
        self.baseline = current
        return None, self.stalls >= self.config.stagnation_patience

    @staticmethod
    def _pair_key(focus, action, destination):
        return (*focus, action, destination)

    def _legal(self, focus, *, local_only=False):
        candidate_id, box = focus
        candidate = self.candidates[candidate_id]
        pairs = []
        for action, possible in (("ZOOM", box[2] > 1 or box[3] > 1),
                                 ("EXPAND", box != self.full_box)):
            if possible and self._pair_key(focus, action, candidate_id) not in self.attempted:
                pairs.append({"action": action, "candidate_id": candidate_id})
        child = self._best_candidate(candidate.children, "SPLIT", focus)
        if child is not None:
            pairs.append({"action": "SPLIT", "candidate_id": child.id})
        if not local_only:
            candidate = self._best_candidate(self.candidates, "NEXT", focus)
            if candidate is not None:
                pairs.append({"action": "NEXT", "candidate_id": candidate.id})
        return pairs

    def _recover(self):
        self.stalls = 0
        previous = self.focus
        for index in range(self.focus_index - 1, -1, -1):
            focus = self.history[index]
            legal = self._legal(focus, local_only=True)
            if legal:
                self.focus, self.focus_index = focus, index
                self.events.append({"kind": "recover", "from_candidate": previous[0],
                                    "from_box": list(previous[1]), "candidate_id": focus[0],
                                    "box": list(focus[1])})
                return legal, "local"
        candidate = self._best_candidate(self.candidates, "NEXT", self.focus)
        if candidate is not None:
            self.events.append({"kind": "recover", "candidate_id": self.focus[0],
                                "box": list(self.focus[1]), "next_available": candidate.id})
            return [{"action": "NEXT", "candidate_id": candidate.id}], "next"
        return [], "local"

    def _navigation(self, legal, feedback):
        prompt = navigation_prompt(
            self.question, self.options, self.views, self.requirements, feedback,
            {"candidate_id": self.focus[0], "box": list(self.focus[1])}, legal,
            self.failed, self.config.max_observations - self.observations,
        )
        output = self._call(self.generator, "navigate", [v.image for v in self.views], prompt,
                            max_tokens=self.config.navigation_tokens)
        proposal = _json_object(output.text)
        fields = {"requirement_id", "feedback_option_ids", "evidence_gap", "action", "candidate_id", "sam_prompt"}
        if set(proposal) != fields:
            raise OutputError("invalid navigation fields")
        pair = {name: proposal[name] for name in ("action", "candidate_id")}
        if pair not in legal:
            raise OutputError("navigation selected an illegal action-candidate pair")
        identifiers = {r["id"] for r in self.requirements}
        if (not isinstance(proposal["requirement_id"], str)
                or proposal["requirement_id"] not in identifiers
                or not isinstance(proposal["evidence_gap"], str) or not proposal["evidence_gap"].strip()):
            raise OutputError("navigation must identify a supplied requirement and evidence gap")
        feedback_ids = proposal["feedback_option_ids"]
        if (not isinstance(feedback_ids, list)
                or any(not isinstance(code, str) or code not in self.options for code in feedback_ids)):
            raise OutputError("navigation referenced unknown option feedback")
        phrase = proposal["sam_prompt"]
        if proposal["action"] in {"ZOOM", "EXPAND"}:
            if not isinstance(phrase, str) or not phrase.strip():
                raise OutputError("localization action requires a nonempty phrase")
        elif phrase is not None:
            raise OutputError("candidate visits require a null localization phrase")
        self.events.append({"kind": "navigation", **proposal})
        return proposal

    def _execute(self, proposal):
        action, identifier = proposal["action"], proposal["candidate_id"]
        candidate = self.candidates[identifier]
        self.attempted.add(self._pair_key(self.focus, action, identifier))
        if action in {"SPLIT", "NEXT"}:
            view = self._candidate_view(candidate, action)
        else:
            box, phrase = self.focus[1], proposal["sam_prompt"]
            x, y, width, height = box
            sam_image = (self.image.crop((x, y, x + width, y + height))
                         if action == "ZOOM" else self.image)
            self.events.append({"kind": "localization", "action": action,
                                "candidate_id": identifier, "sam_prompt": phrase})
            localization = self.frontend.localize(sam_image, phrase)
            if not isinstance(localization, Localization):
                raise OutputError("invalid localization output")
            boxes = [clipped for box_ in localization.boxes
                     if (clipped := clip_box(box_, sam_image.size)) is not None]
            if not boxes:
                return "empty-localization"
            merged = enclose(boxes)
            if action == "ZOOM":
                merged = (merged[0] + x, merged[1] + y, merged[2], merged[3])
                if merged == box:
                    return "invalid-geometry"
                new_box, sam_input = merged, box
            else:
                new_box, sam_input = enclose([box, merged]), self.full_box
                if new_box == box:
                    return "invalid-geometry"
            view = self._make_view(candidate, new_box, action, phrase, localization, sam_input)
        if view is None:
            return "duplicate-or-invalid-box"
        if not self._fits(self.views + [view]):
            return "capacity"
        self._append(candidate, view)
        return None

    def _next_observation(self, stalled, proposal, feedback):
        legal, mode = self._legal(self.focus), "normal"
        if stalled or not legal:
            legal, mode = self._recover()
            proposal = None
        while legal:
            if proposal is None:
                proposal = self._navigation(legal, feedback)
            failed_focus = self.focus
            failure = self._execute(proposal)
            if failure is None:
                return True
            record = {"candidate_id": failed_focus[0], "box": list(failed_focus[1]),
                      "action": proposal["action"], "destination": proposal["candidate_id"],
                      "reason": failure}
            self.failed.append(record)
            self.events.append({"kind": "failed-attempt", **record})
            proposal = None
            legal = self._legal(self.focus, local_only=mode == "local")
            if mode == "next":
                legal = [pair for pair in legal if pair["action"] == "NEXT"]
            if not legal:
                legal, mode = self._recover()
        return False

    def run(self):
        try:
            self.saved_global, probabilities = self._answer()
            order = sorted(range(len(self.codes)), key=lambda index: -probabilities[index])
            if (probabilities[order[0]] >= self.config.global_confidence
                    and probabilities[order[0]] - probabilities[order[1]] >= self.config.global_margin):
                return self._result(self.codes[order[0]], "global-screened", "generator-gate")
            reason = self._room()
            if reason:
                return self._fallback(reason)
            output = self._call(self.generator, "plan", [], plan_prompt(self.question),
                                max_tokens=self.config.planning_tokens)
            phrases, self.requirements = _plan(output.text)
            self.events.append({"kind": "initial-candidates"})
            sam_candidates = self.frontend.initial_candidates(self.image, phrases, self.question)
            self.events.append({"kind": "build-candidates"})
            built = self.frontend.build_candidates(self.image, sam_candidates, self.question, phrases)
            self.candidates = {candidate.id: candidate for candidate in built}
            candidate = self._best_candidate(self.candidates, "INIT")
            if candidate is None:
                return self._fallback("feasibility")
            self._append(candidate, self._candidate_view(candidate, "INIT"))
            while True:
                self._answer()
                previous_feedback = self.feedback
                proposal = None
                reason = self._room()
                if reason is None:
                    legal = self._legal(self.focus)
                    if legal:
                        proposal = self._navigation(legal, previous_feedback)
                accepted, stalled = self._evaluate()
                if accepted is not None:
                    return self._result(accepted, "verified", "verifier-gate")
                if reason:
                    return self._fallback(reason)
                if not self._next_observation(stalled, proposal, previous_feedback):
                    return self._fallback("feasibility")
        except (OutputError, CapacityError) as error:
            if self.saved_global is None:
                raise
            reason = "output-error" if isinstance(error, OutputError) else "capacity"
            self.events.append({"kind": reason, "message": str(error)})
            return self._fallback(reason)


def run_search(image, question, options: Mapping[str, str], *, generator, verifier, frontend,
               config: SearchConfig):
    """Return an answer and trace; only valid acquired local views spend the budget.

    ``options`` maps tokenizer-checked single-token identifiers to option text.
    The model adapter checks tokenization and actual multi-image input capacity.
    """
    if not isinstance(image, Image.Image) or image.width < 1 or image.height < 1:
        raise ValueError("image must be a nonempty PIL image")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if (not isinstance(options, Mapping) or len(options) < 2
            or any(not isinstance(key, str) or not key.strip()
                   or not isinstance(value, str) or not value.strip() for key, value in options.items())):
        raise ValueError("options must map at least two distinct identifiers to nonempty text")
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be SearchConfig")
    return _Search(image, question, options, generator, verifier, frontend, config).run()

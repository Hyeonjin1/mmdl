"""Supplement Algorithm 1 controller for cumulative vLLM token streams."""
import hashlib
import random
import time
from collections import deque

from repetition_guard import RepetitionGuard


WINDOW = 50
MAX_LOOKBACKS = 5
BUDGET = 8192
PHRASE_SEED = 3408
PAUSES = (
    "wait let me", "I need to", "let me check", "I should verify", "hmm I see",
    "looking at this", "re-examining the", "I need to reconsider", "let me think", "hold on",
)
INSERTIONS = (
    "Looking back at the image", "Let me re-examine the", "Returning to the diagram",
    "Checking the figure again", "Looking at the visual", "Re-reading the image",
    "Let me look at this more carefully", "Going back to the picture",
    "Examining the image closely", "Let me verify from the image",
)
ANSWER_SIGNALS = ("Therefore,", "The answer is", "Final answer:")


def token_hash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


class Controller:
    def __init__(self, tokenizer, enabled):
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.rng = random.Random(PHRASE_SEED)
        self.guard = RepetitionGuard()
        self.recent = deque(maxlen=WINDOW)
        self.events = []
        self.segments = []
        self.trace = []
        self.generated = 0
        self.inserted = 0
        self.answer_phase = None
        self.first_pause = None
        self.first_blocked_pause = None
        self.pending = None
        self.evidence = None

    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def consume(self, token, terminal_token=False):
        self.trace.append(token)
        self.generated += 1
        self.recent.append(token)
        suffix = self.decode(self.recent).casefold()
        location = dict(generated_position=self.generated, trace_position=len(self.trace))
        if self.answer_phase is None:
            signal = next((s for s in ANSWER_SIGNALS if s.casefold() in suffix), None)
            if signal:
                self.answer_phase = dict(location, phrase=signal)
        matched = next((p for p in PAUSES if p.casefold() in suffix), None)
        if matched:
            match = dict(location, phrase=matched)
            if self.first_pause is None:
                self.first_pause = dict(match, prefix_sha256=token_hash(self.trace))
            if self.answer_phase is not None and self.first_blocked_pause is None:
                self.first_blocked_pause = match
            last = self.events[-1]["generated_position"] if self.events else None
            if (self.enabled and not terminal_token and self.answer_phase is None
                    and len(self.events) < MAX_LOOKBACKS
                    and (last is None or self.generated - last > WINDOW)):
                phrase = self.rng.choice(INSERTIONS)
                ids = self.tokenizer.encode("\n" + phrase + " ", add_special_tokens=False)
                if len(self.trace) + len(ids) < BUDGET:
                    self.pending = (ids, dict(match, insertion_phrase=phrase,
                                               insertion_tokens=len(ids),
                                               generated_since_last_trigger=None if last is None else self.generated-last))
        if not terminal_token and self.pending is None:
            policy = self.guard.policy
            if (self.generated >= policy["min_tokens"]
                    and self.generated - self.guard.last_check >= policy["check_every_tokens"]):
                self.evidence = self.guard.check(self.decode(self.trace), self.generated)
        return self.pending is not None or self.evidence is not None

    def generate(self, engine, prepared, make_params):
        # On resubmission vLLM still expands the original vision placeholders.
        # Append raw assistant token IDs to the unexpanded prompt and assert
        # the resulting expanded prompt equals the engine's original prefix.
        base = None
        raw_prompt = self.tokenizer.encode(prepared["prompt"], add_special_tokens=False)
        current = prepared
        while True:
            start = time.perf_counter()
            expected_prefix = None if base is None else base + self.trace
            processed = 0
            first_output = None
            cached_tokens = None
            chunk_overshoot = 0

            def observe(output):
                nonlocal base, processed, first_output, cached_tokens, chunk_overshoot
                answer = output.outputs[0]
                if first_output is None and answer.token_ids:
                    first_output = time.perf_counter() - start
                    cached_tokens = output.num_cached_tokens
                    actual = list(output.prompt_token_ids)
                    if base is None:
                        base = actual
                    elif actual != expected_prefix:
                        raise RuntimeError("Expanded multimodal prefix changed on resubmission")
                if not answer.token_ids and output.finished:
                    raise RuntimeError("Streaming request completed without tokens")
                for index in range(processed, len(answer.token_ids)):
                    processed += 1
                    terminal = output.finished and processed == len(answer.token_ids)
                    if self.consume(answer.token_ids[index], terminal):
                        chunk_overshoot = len(answer.token_ids) - processed
                        return True
                return False

            remaining = BUDGET - len(self.trace)
            output = engine.generate([current], make_params(remaining), observe=observe)[0]
            answer = output.outputs[0]
            self.segments.append(dict(
                prompt_tokens=len(output.prompt_token_ids), cached_prompt_tokens=cached_tokens,
                first_output_seconds=first_output, request_seconds=time.perf_counter()-start,
                accepted_generated_tokens=processed, unaccepted_streamed_tokens=chunk_overshoot,
                prompt_token_sha256=token_hash(output.prompt_token_ids), max_new_tokens=remaining,
                resumed_prefix_verified=current is not prepared,
            ))
            if self.pending is None:
                finish = "repetition_abort" if self.evidence is not None else answer.finish_reason
                if finish not in ("stop", "length", "repetition_abort"):
                    raise RuntimeError(f"Unexpected finish reason {finish}")
                if len(self.trace) > BUDGET or self.generated + self.inserted != len(self.trace):
                    raise RuntimeError("Response token accounting mismatch")
                return self.decode(self.trace), finish

            ids, event = self.pending
            self.pending = None
            self.events.append(dict(event, intervention=len(self.events)+1, insertion_token_ids=ids))
            self.trace.extend(ids)
            self.inserted += len(ids)
            # Keep the inserted text out of both matches and cross-boundary substrings.
            self.recent.clear()
            current = {k: v for k, v in prepared.items() if k != "prompt"}
            current["prompt_token_ids"] = raw_prompt + self.trace

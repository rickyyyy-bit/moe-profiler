"""Common request and manifest schemas used by serving and trace paths."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequestSpec(BaseModel):
    """One generation request, optionally associated with a session turn."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    prompt: str
    max_tokens: int = Field(gt=0)
    arrival_time: float = Field(default=0.0, ge=0.0)
    session_id: str | None = None
    turn_index: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ManifestRequest(BaseModel):
    """One immutable request definition shared by trace and serving runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    prompt: str
    prompt_token_ids: tuple[int, ...]
    requested_prompt_tokens: int = Field(gt=0)
    actual_prompt_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    generation_settings: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_token_count(self) -> ManifestRequest:
        if len(self.prompt_token_ids) != self.actual_prompt_tokens:
            raise ValueError("actual_prompt_tokens must match prompt_token_ids")
        return self

    def as_request_spec(self) -> RequestSpec:
        return RequestSpec(
            request_id=self.request_id,
            prompt=self.prompt,
            max_tokens=self.max_output_tokens,
            metadata={
                "target_input_len": self.requested_prompt_tokens,
                "actual_input_len": self.actual_prompt_tokens,
                "prompt_token_ids": list(self.prompt_token_ids),
                "generation_settings": self.generation_settings,
            },
        )


class WorkloadManifest(BaseModel):
    """Canonical workload consumed by both serving and reference tracing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    requests: tuple[ManifestRequest, ...]

    @model_validator(mode="after")
    def validate_requests(self) -> WorkloadManifest:
        if not self.requests:
            raise ValueError("workload manifest requires at least one request")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("workload manifest request IDs must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def request_specs(self) -> list[RequestSpec]:
        return [request.as_request_spec() for request in self.requests]


def build_exact_length_manifest(
    tokenizer: object,
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
    request_prefix: str,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
) -> WorkloadManifest:
    """Create deterministic prompts and verify their tokenized lengths.

    The prompt is constructed from token IDs and round-tripped through the
    tokenizer. A tokenizer that cannot preserve the requested length is rejected
    instead of producing a mislabeled workload.
    """
    if concurrency <= 0 or prompt_tokens <= 0 or output_tokens <= 0:
        raise ValueError("concurrency and token lengths must be positive")
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if not callable(encode) or not callable(decode):
        raise TypeError("tokenizer must provide encode and decode methods")
    seed_text = "The quick brown fox explains sparse expert routing."
    seed_ids = list(encode(seed_text, add_special_tokens=False))
    if not seed_ids:
        raise ValueError("tokenizer produced no seed tokens")
    repeated = (seed_ids * ((prompt_tokens + len(seed_ids) - 1) // len(seed_ids)))[
        :prompt_tokens
    ]
    prompt = decode(repeated, skip_special_tokens=False)
    verified = tuple(int(token) for token in encode(prompt, add_special_tokens=False))
    if len(verified) != prompt_tokens:
        raise ValueError(
            "tokenizer round-trip could not construct the requested prompt length: "
            f"target={prompt_tokens}, actual={len(verified)}"
        )
    requests = tuple(
        ManifestRequest(
            request_id=f"{request_prefix}-req{index}",
            prompt=prompt,
            prompt_token_ids=verified,
            requested_prompt_tokens=prompt_tokens,
            actual_prompt_tokens=len(verified),
            max_output_tokens=output_tokens,
            generation_settings={"temperature": 0},
        )
        for index in range(concurrency)
    )
    return WorkloadManifest(
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        requests=requests,
    )

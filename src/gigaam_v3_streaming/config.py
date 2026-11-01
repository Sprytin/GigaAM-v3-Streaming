"""Configuration and frontend validation for streaming inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FrontendAlignment:
    """The timing parameters derived from a loaded GigaAM model."""

    sample_rate: int
    feature_hop_samples: int
    subsampling_factor: int

    @property
    def encoder_hop_samples(self) -> int:
        return self.feature_hop_samples * self.subsampling_factor


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Fixed-clock parameters for :class:`~gigaam_v3_streaming.StreamingSession`.

    All timing values are expressed in milliseconds so the same logical schedule
    is used regardless of the sizes of calls to ``push``.  ``validate`` is run
    by ``StreamingSession`` against the loaded model; it verifies that every
    logical boundary falls on an encoder-frame boundary.
    """

    sample_rate: int = 16_000
    emit_ms: int = 320
    left_context_ms: int = 8_000
    right_context_ms: int = 800
    max_symbols_per_step: int = 10

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.emit_ms <= 0:
            raise ValueError("emit_ms must be positive")
        if self.left_context_ms < 0 or self.right_context_ms < 0:
            raise ValueError("context durations must not be negative")
        if self.max_symbols_per_step <= 0:
            raise ValueError("max_symbols_per_step must be positive")
        for name, value in self._time_values().items():
            samples = value * self.sample_rate / 1_000
            if samples != int(samples):
                raise ValueError(f"{name}={value} ms is not integral at {self.sample_rate} Hz")

    def _time_values(self) -> dict[str, int]:
        return {
            "emit_ms": self.emit_ms,
            "left_context_ms": self.left_context_ms,
            "right_context_ms": self.right_context_ms,
        }

    def samples_for_ms(self, milliseconds: int) -> int:
        return milliseconds * self.sample_rate // 1_000

    @property
    def emit_samples(self) -> int:
        return self.samples_for_ms(self.emit_ms)

    @property
    def left_context_samples(self) -> int:
        return self.samples_for_ms(self.left_context_ms)

    @property
    def right_context_samples(self) -> int:
        return self.samples_for_ms(self.right_context_ms)

    @property
    def window_samples(self) -> int:
        return self.left_context_samples + self.emit_samples + self.right_context_samples

    @property
    def ring_capacity_samples(self) -> int:
        """Enough PCM for one complete logical window, and never more."""
        return self.window_samples

    def validate(self, model: Any) -> FrontendAlignment:
        """Derive and validate the model's frontend/frame alignment.

        GigaAM's stock frontend exposes ``hop_length`` on the preprocessor and
        its x4 striding frontend exposes ``_sampling_num``.  The small fallback
        attributes make deterministic fake models convenient in scheduler tests,
        while real models still require the same concrete values.
        """

        preprocessor = getattr(model, "preprocessor", None)
        hop = getattr(preprocessor, "hop_length", None)
        if hop is None:
            hop = getattr(model, "feature_hop_samples", None)
        if hop is None:
            raise ValueError("model does not expose preprocessor.hop_length")

        model_rate = getattr(preprocessor, "sample_rate", None)
        if model_rate is None and preprocessor is not None:
            featurizer = getattr(preprocessor, "featurizer", None)
            try:
                model_rate = getattr(featurizer[0], "sample_rate", None)
            except (IndexError, KeyError, TypeError):
                pass
        if model_rate is None:
            model_rate = getattr(model, "sample_rate", None)
        if model_rate is None:
            # Stock FeatureExtractor does not retain sample_rate itself, but the
            # released GigaAM ASR family is defined at 16 kHz.
            model_rate = 16_000
        if int(model_rate) != self.sample_rate:
            raise ValueError(
                f"config sample_rate={self.sample_rate} does not match model "
                f"sample_rate={model_rate}"
            )

        pre_encode = getattr(getattr(model, "encoder", None), "pre_encode", None)
        stages = getattr(pre_encode, "_sampling_num", None)
        factor = (
            2 ** int(stages) if stages is not None else getattr(model, "subsampling_factor", None)
        )
        if factor is None or int(factor) <= 0:
            raise ValueError("model does not expose encoder xN subsampling")

        alignment = FrontendAlignment(
            sample_rate=self.sample_rate,
            feature_hop_samples=int(hop),
            subsampling_factor=int(factor),
        )
        if alignment.encoder_hop_samples <= 0:
            raise ValueError("invalid frontend hop or subsampling factor")

        for name, samples in (
            ("emit_ms", self.emit_samples),
            ("left_context_ms", self.left_context_samples),
            ("right_context_ms", self.right_context_samples),
        ):
            if samples % alignment.encoder_hop_samples:
                raise ValueError(
                    f"{name} must align to the model encoder hop "
                    f"({alignment.encoder_hop_samples} samples); got {samples}"
                )
        return alignment

"""Incremental greedy RNN-T decoding for stock GigaAM heads."""

from __future__ import annotations

from typing import Any


class RNNTGreedyState:
    """Bounded predictor state carried over consecutive encoder windows."""

    def __init__(self, head: Any, max_symbols_per_step: int = 10) -> None:
        self.head = head
        self.max_symbols_per_step = max_symbols_per_step
        self.last_label = None
        self.decoder_state = None
        self.frames_seen = 0

    def decode(self, encoded: Any) -> list[tuple[int, int]]:
        """Greedily decode ``[D, T]`` or ``[1, D, T]`` encoder features.

        Returns ``(token_id, frame_index)`` pairs, where ``frame_index`` is the
        position of the emitting encoder frame inside ``encoded``.

        This is intentionally the same state transition as GigaAM's offline
        ``RNNTGreedyDecoding``: predictor state advances only for a non-blank
        symbol, while a blank consumes the current encoder frame.
        """
        import torch

        # ``StreamingSession`` returns an inference tensor from the acoustic
        # encoder.  Keep the predictor and joint in the same inference scope so
        # a direct public ``push`` call never asks autograd to save that tensor.
        with torch.inference_mode():
            return self._decode_inference(encoded, torch)

    def _decode_inference(self, encoded: Any, torch: Any) -> list[tuple[int, int]]:
        """Implementation split out so the whole transition stays inference-only."""

        if encoded.ndim == 3:
            if encoded.shape[0] != 1:
                raise ValueError("streaming RNN-T decoder supports batch size one")
            encoded = encoded[0]
        if encoded.ndim != 2:
            raise ValueError("encoded features must have shape [D, T] or [1, D, T]")

        x = encoded.transpose(0, 1)  # [T, D]
        emitted: list[tuple[int, int]] = []
        blank_id = self.head.decoder.blank_id
        for index, frame in enumerate(x):
            f = frame.view(1, 1, -1)
            for _ in range(self.max_symbols_per_step):
                if self.decoder_state is None:
                    g, hidden = self.head.decoder.predict(None, None, batch_size=1)
                else:
                    g, hidden = self.head.decoder.predict(
                        self.last_label, self.decoder_state, batch_size=1
                    )
                token = int(self.head.joint.joint(f, g)[0, 0, 0].argmax().item())
                if token == blank_id:
                    break
                emitted.append((token, index))
                self.last_label = torch.tensor([[token]], dtype=torch.long, device=f.device)
                self.decoder_state = hidden
            self.frames_seen += 1
        return emitted

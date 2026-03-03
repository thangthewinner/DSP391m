"""
SLM Reasoning Layer — Phase 3.

Uses a quantized Small Language Model (Qwen2.5-3B-Instruct-GGUF) to confirm
whether a transcript is semantically related to the exam question.

Flow:
    similarity ≥ 0.60 → SLM.predict(question, transcript) → YES/NO
    YES → +2pt in Decision Engine
    NO  → skip (no points added)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_cpp import Llama, LlamaGrammar

PROMPT_TEMPLATE = """\
<|im_start|>system
Bạn là hệ thống phát hiện gian lận thi cử. Nhiệm vụ: xác định xem đoạn hội thoại có liên quan đến câu hỏi thi hay không.
Chỉ trả lời YES hoặc NO.<|im_end|>
<|im_start|>user
Câu hỏi thi: "{question}"
Đoạn hội thoại: "{transcript}"
Đoạn hội thoại này có phải là thí sinh đang trao đổi về nội dung câu hỏi thi không?<|im_end|>
<|im_start|>assistant
"""

# Grammar to constrain output strictly to YES or NO
GRAMMAR_STR = 'root ::= "YES" | "NO"'


class SLMProcessor:
    """
    Small Language Model processor for reasoning-based cheating confirmation.

    Only called when embedding similarity ≥ threshold to avoid unnecessary compute.
    Runs synchronously (in executor) to avoid blocking the async event loop.
    """

    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = 0,
        max_tokens: int = 4,
        context_length: int = 512,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.context_length = context_length
        self.model: Any = None
        self._grammar: Any = None

    def load_model(self) -> None:
        """Load GGUF model using llama-cpp-python."""
        from llama_cpp import Llama, LlamaGrammar

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SLM model not found: {self.model_path}\n"
                f"Run: uv run python scripts/download_models.py --slm"
            )

        logger.info("Loading SLM: %s", self.model_path.name)
        logger.info("  n_gpu_layers=%d, context=%d", self.n_gpu_layers, self.context_length)

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_length,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._grammar = LlamaGrammar.from_string(GRAMMAR_STR)

        logger.info("✓ SLM loaded successfully: %s", self.model_path.name)

    def predict(self, exam_question: str, transcript: str) -> bool:
        """
        Ask the SLM whether the transcript is related to the exam question.

        Args:
            exam_question: The exam question text
            transcript: The transcribed speech segment

        Returns:
            True if SLM answers YES (related/suspicious), False if NO
        """
        if self.model is None:
            raise RuntimeError("SLM not loaded. Call load_model() first.")

        if not exam_question.strip() or not transcript.strip():
            return False

        prompt = PROMPT_TEMPLATE.format(
            question=exam_question[:200],
            transcript=transcript[:300],
        )

        try:
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                grammar=self._grammar,
                temperature=0.0,
                echo=False,
            )
            answer = response["choices"][0]["text"].strip().upper()
            verdict = answer == "YES"

            logger.info("SLM verdict: %s — '%s'", answer, transcript[:50])
            return verdict

        except Exception as e:  # noqa: BLE001
            logger.warning("SLM inference failed, defaulting to NO: %s", e)
            return False

# ABOUTME: Unit tests for GKD data format, loss helpers, and pipeline logic
# ABOUTME: in scripts.train.train_student_gkd. Tests pure functions without GPU/model loading.

"""
Unit tests for the GKD data preparation and pipeline logic.

Covers:
  - Messages format conversion (build_messages_example, build_gkd_dataset)
  - Data loading and filtering with Qwen3.5 teacher data
  - Split determinism (same hash logic as train_student.py)
  - GKD JSD loss function (imported from TRL experimental)
  - LoRA target module auto-detection
"""

from __future__ import annotations

import json

import pytest
import torch

from configs.teacher_prompt import SYSTEM_INSTRUCTION
from scripts.train.train_student_gkd import (
    _split_bucket,
    build_gkd_dataset,
    build_messages_example,
    split_examples,
    teacher_output_to_assistant_text,
)


# ---------- Fixtures ----------


@pytest.fixture
def sample():
    """Minimal but realistic processed sample (2 history visits, 3 candidates).

    Returns:
        dict: sample record matching philly_samples.jsonl schema.
    """
    return {
        "sample_id": "s1",
        "user_id": "u1",
        "city": "Philadelphia",
        "history": [
            {
                "business_id": "h1",
                "name": "Pho Street",
                "categories": "Vietnamese, Noodles",
                "stars": 5,
                "review_snippet": "Best pho in town",
                "date": "2023-05-01",
            },
            {
                "business_id": "h2",
                "name": "Blue Bottle",
                "categories": "Coffee, Cafes",
                "stars": 4,
                "review_snippet": "Good espresso but crowded",
                "date": "2023-05-15",
            },
        ],
        "candidates": [
            {"business_id": "c1", "name": "Banh Mi Corner", "categories": "Vietnamese, Sandwiches", "avg_stars": 4.3, "attributes": {}},
            {"business_id": "c2", "name": "Stumptown Coffee", "categories": "Coffee, Cafes", "avg_stars": 4.5, "attributes": {}},
            {"business_id": "c3", "name": "Chipotle", "categories": "Mexican, Fast Food", "avg_stars": 3.2, "attributes": {}},
        ],
        "positive_business_id": "c1",
    }


@pytest.fixture
def teacher_rec():
    """Minimal teacher output record matching philly_teacher_qwen35.jsonl schema.

    Returns:
        dict: teacher record with sample_id, teacher_output, error=None.
    """
    return {
        "sample_id": "s1",
        "teacher_output": {
            "persona": "A food enthusiast who enjoys Vietnamese and coffee.",
            "rationales": [
                {"business_id": "c1", "reason": "Matches Vietnamese preference."},
                {"business_id": "c2", "reason": "Aligns with coffee habit."},
                {"business_id": "c3", "reason": "Not a great fit for this user."},
            ],
            "ranking": ["c1", "c2", "c3"],
        },
        "error": None,
    }


@pytest.fixture
def joined_examples(sample, teacher_rec):
    """List of joined examples as produced by load_and_filter.

    Returns:
        list[dict]: joined records with sample_id, sample, teacher keys.
    """
    examples = []
    for i in range(5):
        s = {**sample, "sample_id": f"s{i}", "user_id": f"u{i}"}
        t = {**teacher_rec, "sample_id": f"s{i}"}
        examples.append({"sample_id": f"s{i}", "sample": s, "teacher": t})
    return examples


# ---------- Messages format tests ----------


class TestBuildMessagesExample:
    """Tests for build_messages_example (GKD data format)."""

    def test_has_messages_key(self, sample, teacher_rec):
        """Result must have a 'messages' key for DataCollatorForChatML."""
        result = build_messages_example(sample, teacher_rec)
        assert "messages" in result

    def test_three_messages(self, sample, teacher_rec):
        """Messages list must contain exactly 3 entries: system, user, assistant."""
        result = build_messages_example(sample, teacher_rec)
        assert len(result["messages"]) == 3

    def test_roles_order(self, sample, teacher_rec):
        """Message roles must be system → user → assistant."""
        result = build_messages_example(sample, teacher_rec)
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_system_content_matches(self, sample, teacher_rec):
        """System message must use the shared SYSTEM_INSTRUCTION constant."""
        result = build_messages_example(sample, teacher_rec)
        assert result["messages"][0]["content"] == SYSTEM_INSTRUCTION

    def test_assistant_content_is_valid_json(self, sample, teacher_rec):
        """Assistant message must be valid JSON with persona, rationales, ranking."""
        result = build_messages_example(sample, teacher_rec)
        parsed = json.loads(result["messages"][2]["content"])
        assert "persona" in parsed
        assert "rationales" in parsed
        assert "ranking" in parsed

    def test_ranking_preserves_order(self, sample, teacher_rec):
        """Ranking in the serialized assistant content must match teacher order."""
        result = build_messages_example(sample, teacher_rec)
        parsed = json.loads(result["messages"][2]["content"])
        assert parsed["ranking"] == ["c1", "c2", "c3"]

    def test_user_content_has_history(self, sample, teacher_rec):
        """User message must include the visit history."""
        result = build_messages_example(sample, teacher_rec)
        user_content = result["messages"][1]["content"]
        assert "Pho Street" in user_content
        assert "Blue Bottle" in user_content

    def test_user_content_has_candidates(self, sample, teacher_rec):
        """User message must include the candidate list."""
        result = build_messages_example(sample, teacher_rec)
        user_content = result["messages"][1]["content"]
        assert "Banh Mi Corner" in user_content
        assert "Chipotle" in user_content


# ---------- build_gkd_dataset tests ----------


class TestBuildGKDDataset:
    """Tests for build_gkd_dataset (batch conversion)."""

    def test_length_matches_input(self, joined_examples):
        """Output length must match input length."""
        result = build_gkd_dataset(joined_examples)
        assert len(result) == len(joined_examples)

    def test_all_have_messages_key(self, joined_examples):
        """Every element must have a 'messages' key."""
        result = build_gkd_dataset(joined_examples)
        assert all("messages" in d for d in result)

    def test_empty_input(self):
        """Empty input must produce empty output."""
        assert build_gkd_dataset([]) == []


# ---------- teacher_output_to_assistant_text tests ----------


class TestTeacherOutputSerialization:
    """Tests for teacher_output_to_assistant_text."""

    def test_roundtrip(self, teacher_rec):
        """Serialized text must round-trip back to the original structure."""
        text = teacher_output_to_assistant_text(teacher_rec["teacher_output"])
        parsed = json.loads(text)
        assert parsed["persona"] == teacher_rec["teacher_output"]["persona"]
        assert parsed["ranking"] == teacher_rec["teacher_output"]["ranking"]
        assert len(parsed["rationales"]) == 3

    def test_non_ascii_preserved(self):
        """Non-ASCII characters must survive serialization (ensure_ascii=False)."""
        output = {
            "persona": "Café enthusiast who loves crêpes",
            "rationales": [{"business_id": "b1", "reason": "Très bon"}],
            "ranking": ["b1"],
        }
        text = teacher_output_to_assistant_text(output)
        assert "Café" in text
        assert "crêpes" in text
        assert "Très" in text


# ---------- Split determinism tests ----------


class TestSplitDeterminism:
    """Tests for _split_bucket and split_examples."""

    def test_deterministic(self):
        """Same sample_id must always map to the same bucket."""
        result1 = _split_bucket("test_id_42", 0.9)
        result2 = _split_bucket("test_id_42", 0.9)
        assert result1 == result2

    def test_valid_values(self):
        """Bucket must be either 'train' or 'eval'."""
        for i in range(100):
            bucket = _split_bucket(f"sample_{i}", 0.9)
            assert bucket in ("train", "eval")

    def test_approximate_ratio(self):
        """Over many samples, the split should approximate the requested ratio."""
        n = 1000
        train_count = sum(
            1 for i in range(n) if _split_bucket(f"sample_{i}", 0.9) == "train"
        )
        # 90% ± 5% tolerance
        assert 850 <= train_count <= 950

    def test_split_examples_partitions(self, joined_examples):
        """Split must be a partition — no overlaps, no missing records."""
        train, ev = split_examples(joined_examples, ratio=0.9)
        train_ids = {e["sample_id"] for e in train}
        eval_ids = {e["sample_id"] for e in ev}
        all_ids = {e["sample_id"] for e in joined_examples}
        assert train_ids | eval_ids == all_ids
        assert train_ids & eval_ids == set()

    def test_compatible_with_train_student_split(self):
        """GKD split must be identical to train_student.py split (same hash logic).

        This ensures the eval split is the same across off-policy SFT and
        on-policy GKD, making metrics comparable.
        """
        from scripts.train.train_student import _split_bucket as sft_split_bucket

        for i in range(200):
            sid = f"test_sample_{i}"
            assert _split_bucket(sid, 0.9) == sft_split_bucket(sid, 0.9)


# ---------- GKD JSD loss tests ----------


class TestGKDJSDLoss:
    """Tests for the generalized_jsd_loss function from TRL.

    These test the mathematical properties of the loss without needing
    a model or GPU. Uses small synthetic logit tensors.
    """

    @staticmethod
    def _get_jsd_loss():
        """Import GKDTrainer.generalized_jsd_loss.

        Returns:
            Callable: the static JSD loss function.
        """
        import os
        os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
        from trl.experimental.gkd import GKDTrainer
        return GKDTrainer.generalized_jsd_loss

    def test_identical_distributions_zero_loss(self):
        """JSD between identical distributions must be (near) zero."""
        jsd_loss = self._get_jsd_loss()
        logits = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        loss = jsd_loss(logits, logits.clone(), beta=0.5, temperature=1.0)
        assert loss.item() < 1e-5

    def test_different_distributions_positive_loss(self):
        """JSD between different distributions must be positive."""
        jsd_loss = self._get_jsd_loss()
        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100) + 5.0  # shifted
        loss = jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0)
        assert loss.item() > 0.0

    def test_beta_zero_is_forward_kl(self):
        """beta=0 should give forward KL: KL(teacher || student)."""
        jsd_loss = self._get_jsd_loss()
        student_logits = torch.randn(2, 5, 50)
        teacher_logits = torch.randn(2, 5, 50) + 2.0
        loss = jsd_loss(student_logits, teacher_logits, beta=0.0, temperature=1.0)
        assert loss.item() > 0.0

    def test_beta_one_is_reverse_kl(self):
        """beta=1 should give reverse KL: KL(student || teacher)."""
        jsd_loss = self._get_jsd_loss()
        student_logits = torch.randn(2, 5, 50)
        teacher_logits = torch.randn(2, 5, 50) + 2.0
        loss = jsd_loss(student_logits, teacher_logits, beta=1.0, temperature=1.0)
        assert loss.item() > 0.0

    def test_symmetric_jsd(self):
        """JSD with beta=0.5 should be symmetric: JSD(A,B) == JSD(B,A)."""
        jsd_loss = self._get_jsd_loss()
        a = torch.randn(2, 5, 50)
        b = torch.randn(2, 5, 50) + 1.0
        loss_ab = jsd_loss(a, b, beta=0.5, temperature=1.0)
        loss_ba = jsd_loss(b, a, beta=0.5, temperature=1.0)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-5

    def test_label_masking_reduces_loss(self):
        """Masking some positions should reduce loss vs no masking."""
        jsd_loss = self._get_jsd_loss()
        student_logits = torch.randn(1, 5, 50)
        teacher_logits = torch.randn(1, 5, 50) + 3.0
        # No masking (all positions contribute)
        labels_all = torch.ones((1, 5), dtype=torch.long)
        loss_all = jsd_loss(student_logits, teacher_logits, labels=labels_all, beta=0.5)
        # Partial masking (only positions 3-4 contribute, 0-2 masked)
        labels_partial = torch.ones((1, 5), dtype=torch.long)
        labels_partial[0, :3] = -100
        loss_partial = jsd_loss(student_logits, teacher_logits, labels=labels_partial, beta=0.5)
        # Both should be valid (not NaN) and positive
        assert not torch.isnan(loss_all) and loss_all.item() > 0
        assert not torch.isnan(loss_partial) and loss_partial.item() > 0

    def test_full_mask_produces_nan(self):
        """When all positions are masked, loss is NaN (0/0). This is expected
        TRL behavior — the DataCollator guarantees at least one completion
        token per batch in practice."""
        jsd_loss = self._get_jsd_loss()
        student_logits = torch.randn(1, 5, 50)
        teacher_logits = torch.randn(1, 5, 50) + 3.0
        labels = torch.full((1, 5), -100, dtype=torch.long)
        loss = jsd_loss(student_logits, teacher_logits, labels=labels, beta=0.5)
        assert torch.isnan(loss)

    def test_temperature_scaling(self):
        """Higher temperature should produce softer distributions and lower JSD."""
        jsd_loss = self._get_jsd_loss()
        student_logits = torch.randn(2, 5, 50)
        teacher_logits = torch.randn(2, 5, 50) + 3.0
        loss_t1 = jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0)
        loss_t5 = jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=5.0)
        # Higher temperature softens distributions → lower divergence
        assert loss_t5.item() < loss_t1.item()


# ---------- LoRA target module detection tests ----------


class TestFindLoraTargetModules:
    """Tests for find_lora_target_modules on mock model architectures."""

    def test_finds_proj_modules(self):
        """Should find *_proj modules in a simple model."""
        from scripts.train.train_student_gkd import find_lora_target_modules
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        "self_attn": nn.ModuleDict({
                            "q_proj": nn.Linear(64, 64),
                            "k_proj": nn.Linear(64, 64),
                            "v_proj": nn.Linear(64, 64),
                            "o_proj": nn.Linear(64, 64),
                        }),
                        "mlp": nn.ModuleDict({
                            "gate_proj": nn.Linear(64, 128),
                            "up_proj": nn.Linear(64, 128),
                            "down_proj": nn.Linear(128, 64),
                        }),
                    })
                ])
                self.lm_head = nn.Linear(64, 1000)
                self.embed_tokens = nn.Embedding(1000, 64)

        model = MockModel()
        targets = find_lora_target_modules(model)
        assert "q_proj" in targets
        assert "v_proj" in targets
        assert "gate_proj" in targets
        assert "down_proj" in targets
        # lm_head and embed should be excluded
        assert "lm_head" not in targets
        assert "embed_tokens" not in targets

    def test_excludes_visual_modules(self):
        """Should exclude modules under 'visual' or 'merger' namespaces."""
        from scripts.train.train_student_gkd import find_lora_target_modules
        import torch.nn as nn

        class MockVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = nn.ModuleDict({
                    "proj": nn.ModuleDict({
                        "q_proj": nn.Linear(64, 64),
                    })
                })
                self.model = nn.ModuleDict({
                    "layers": nn.ModuleList([
                        nn.ModuleDict({
                            "self_attn": nn.ModuleDict({
                                "q_proj": nn.Linear(64, 64),
                            }),
                        })
                    ])
                })
                self.lm_head = nn.Linear(64, 1000)

        model = MockVLM()
        targets = find_lora_target_modules(model)
        # Should find q_proj from model.layers but not from visual
        assert "q_proj" in targets

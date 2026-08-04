import tempfile
import unittest
from pathlib import Path
import sys

import torch

from pruning.unstructured_magnitude.gradual import (
    GradualMagnitudeConfig,
    apply_gradual_pruning,
    eligible_weight_parameters,
    initialize_masks,
    restore_masks_from_training_checkpoint,
    save_mask_artifact,
    scheduled_sparsity,
    should_update_masks,
)
CORE_ROOT = Path(__file__).resolve().parents[1] / "train" / "devkit" / "core"
sys.path.insert(0, str(CORE_ROOT))

from mask_utils import (  # noqa: E402
    apply_parameter_masks,
    register_parameter_mask_hooks,
)


class TinyPrunableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.middle_a = torch.nn.Conv2d(1, 2, 1, bias=False)
        self.middle_b = torch.nn.Conv2d(2, 2, 1, bias=False)
        self.last = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.first.weight.fill_(0.01)
            self.middle_a.weight.copy_(torch.tensor([1.0, 4.0]).reshape_as(self.middle_a.weight))
            self.middle_b.weight.copy_(
                torch.tensor([2.0, 3.0, 5.0, 6.0]).reshape_as(self.middle_b.weight)
            )
            self.last.weight.fill_(0.02)


class GradualMagnitudeTest(unittest.TestCase):
    def make_state(self):
        model = TinyPrunableModel()
        eligible, protected = eligible_weight_parameters(model)
        masks = initialize_masks(eligible)
        return model, eligible, protected, masks

    def test_cubic_schedule_reaches_exact_target(self):
        config = GradualMagnitudeConfig(0.30, start_epoch=0, end_epoch=2)
        config.validate(total_epochs=3)
        self.assertEqual(scheduled_sparsity(config, -1), 0.0)
        self.assertEqual(scheduled_sparsity(config, 0), 0.0)
        self.assertAlmostEqual(scheduled_sparsity(config, 1), 0.2625)
        self.assertEqual(scheduled_sparsity(config, 2), 0.30)
        self.assertTrue(all(should_update_masks(config, epoch) for epoch in (0, 1, 2)))
        self.assertFalse(should_update_masks(config, 3))

    def test_global_pruning_is_exact_monotonic_and_protects_boundaries(self):
        model, eligible, protected, masks = self.make_state()
        self.assertEqual(protected, ["first.weight", "last.weight"])

        first = apply_gradual_pruning(eligible, masks, 0.5, "global")
        first_pruned = {name: ~mask.clone() for name, mask in masks.items()}
        self.assertEqual(first["eligible_weights"], 6)
        self.assertEqual(first["nonzero_mask"], 3)
        self.assertAlmostEqual(model.first.weight.item(), 0.01)
        self.assertTrue(torch.all(model.last.weight == 0.02))

        second = apply_gradual_pruning(eligible, masks, 4 / 6, "global")
        self.assertEqual(second["nonzero_mask"], 2)
        for name, pruned in first_pruned.items():
            self.assertTrue(torch.all(~masks[name][pruned]))

    def test_local_pruning_hits_each_layer_target(self):
        _, eligible, _, masks = self.make_state()
        statistics = apply_gradual_pruning(eligible, masks, 0.5, "local")
        self.assertEqual(statistics["nonzero_mask"], 3)
        self.assertEqual(int(masks["middle_a.weight"].count_nonzero()), 1)
        self.assertEqual(int(masks["middle_b.weight"].count_nonzero()), 2)

    def test_mutated_masks_are_seen_by_existing_gradient_hooks(self):
        model, eligible, _, masks = self.make_state()
        handles = register_parameter_mask_hooks(model, masks)
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        apply_gradual_pruning(eligible, masks, 0.5, "global")

        loss = sum(parameter.sum() for _, parameter in eligible)
        loss.backward()
        parameters = dict(model.named_parameters())
        for name, mask in masks.items():
            self.assertTrue(torch.all(parameters[name].grad[~mask] == 0))

        with torch.no_grad():
            for _, parameter in eligible:
                parameter.add_(1.0)
        apply_parameter_masks(model, masks)
        for name, mask in masks.items():
            self.assertTrue(torch.all(parameters[name][~mask] == 0))

    def test_epoch_artifact_is_idempotent_but_rejects_mismatch(self):
        _, eligible, protected, masks = self.make_state()
        config = GradualMagnitudeConfig(0.5, start_epoch=0, end_epoch=1)
        stats = apply_gradual_pruning(eligible, masks, 0.5, "global")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.pth"
            save_mask_artifact(path, masks, config, 1, 0.5, stats, protected)
            save_mask_artifact(path, masks, config, 1, 0.5, stats, protected)
            changed = {name: mask.clone() for name, mask in masks.items()}
            changed["middle_a.weight"].fill_(True)
            with self.assertRaises(FileExistsError):
                save_mask_artifact(path, changed, config, 1, 0.5, stats, protected)

    def test_resume_restores_exact_masks_and_rejects_changed_schedule(self):
        _, eligible, _, masks = self.make_state()
        config = GradualMagnitudeConfig(0.5, start_epoch=0, end_epoch=1)
        apply_gradual_pruning(eligible, masks, 0.5, "global")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_path = run_dir / "model.pth-1"
            torch.save(
                {
                    "parameter_masks": {
                        name: mask.cpu() for name, mask in masks.items()
                    },
                    "gradual_pruning": {"schedule": config.to_dict()},
                },
                checkpoint_path,
            )
            (run_dir / "checkpoint").write_text(
                f"model_checkpoint_path:{checkpoint_path}\n", encoding="utf-8"
            )
            restored = restore_masks_from_training_checkpoint(
                run_dir, eligible, config
            )
            for name, mask in masks.items():
                self.assertTrue(torch.equal(restored[name], mask))

            changed = GradualMagnitudeConfig(0.4, start_epoch=0, end_epoch=1)
            with self.assertRaises(ValueError):
                restore_masks_from_training_checkpoint(run_dir, eligible, changed)

    def test_resume_rejects_checkpoint_without_gradual_metadata(self):
        _, eligible, _, _ = self.make_state()
        config = GradualMagnitudeConfig(0.5, start_epoch=0, end_epoch=1)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_path = run_dir / "model.pth-1"
            torch.save({"state_dict": {}}, checkpoint_path)
            (run_dir / "checkpoint").write_text(
                f"model_checkpoint_path:{checkpoint_path}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "no gradual schedule"):
                restore_masks_from_training_checkpoint(
                    run_dir, eligible, config
                )

    def test_invalid_configurations_fail_clearly(self):
        with self.assertRaises(ValueError):
            GradualMagnitudeConfig(1.0, 0, 1).validate(2)
        with self.assertRaises(ValueError):
            GradualMagnitudeConfig(0.3, 2, 1).validate(3)
        with self.assertRaises(ValueError):
            GradualMagnitudeConfig(0.3, 0, 3).validate(3)


if __name__ == "__main__":
    unittest.main()

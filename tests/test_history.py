from datetime import datetime, timezone
import unittest

import torch

from astgtn.history import PointInTimePrior, utc_seconds


def event(relation_id="r1", when="2014-01-01T00:00:00Z", **updates):
    record = dict(
        relation_id=relation_id, source=0, target=1, known_at=when,
        effective_at=when, source_available_at=when,
        source_ref="verified historical filing", action="add",
    )
    record.update(updates)
    return record


def at(*timestamps):
    return torch.tensor([[utc_seconds(value) for value in timestamps]], dtype=torch.int64)


class PointInTimeGraphTests(unittest.TestCase):
    def test_future_add_is_excluded_until_exact_activation(self):
        prior = PointInTimePrior(2, [event(when="2016-01-01T00:00:00Z")])
        result = prior(at("2014-01-01T00:00:00Z", "2015-12-31T23:59:59Z", "2016-01-01T00:00:00Z"))
        torch.testing.assert_close(result[0, :2], torch.eye(2).expand(2, -1, -1))
        torch.testing.assert_close(result[0, 2], torch.full((2, 2), 0.5))

    def test_each_time_boundary_independently_blocks_future_information(self):
        for field in ("known_at", "effective_at", "source_available_at"):
            with self.subTest(field=field):
                prior = PointInTimePrior(2, [event(**{field: "2018-01-05T00:00:00Z"})])
                result = prior(at("2014-01-01T00:00:00Z", "2017-12-31T23:59:59Z", "2018-01-05T00:00:00Z"))
                torch.testing.assert_close(result[0, :2], torch.eye(2).expand(2, -1, -1))
                self.assertEqual(result[0, 2, 0, 1].item(), 0.5)

    def test_late_removal_does_not_rewrite_history(self):
        records = [event(), event(when="2016-01-01T00:00:00Z", action="remove", known_at="2017-01-01T00:00:00Z")]
        result = PointInTimePrior(2, records)(at("2015-01-01T00:00:00Z", "2016-12-31T23:59:59Z", "2017-01-01T00:00:00Z"))
        torch.testing.assert_close(result[0, :2], torch.full((2, 2, 2), 0.5))
        torch.testing.assert_close(result[0, 2], torch.eye(2))

    def test_adding_future_events_cannot_change_earlier_graphs(self):
        cutoffs = at("2014-01-01T00:00:00Z", "2015-12-31T23:59:59Z")
        earlier = [event()]
        future = [event(when="2016-01-01T00:00:00Z", action="remove"), event("r2", "2018-01-01T00:00:00Z", source=1, target=2)]
        torch.testing.assert_close(PointInTimePrior(3, earlier)(cutoffs), PointInTimePrior(3, list(reversed(earlier + future)))(cutoffs))

    def test_timezone_conversion_is_exact_and_aware_datetime_supported(self):
        self.assertEqual(utc_seconds("2014-01-01T08:00:00+08:00"), utc_seconds("2014-01-01T00:00:00Z"))
        self.assertEqual(utc_seconds(datetime(2014, 1, 1, tzinfo=timezone.utc)), utc_seconds("2014-01-01T00:00:00Z"))
        result = PointInTimePrior(2, [event(when="2014-01-01T08:00:00+08:00")])(at("2014-01-01T00:00:00Z"))
        self.assertEqual(result[0, 0, 0, 1].item(), 0.5)

    def test_ambiguous_or_rounded_timestamps_rejected(self):
        invalid = ["2014-01-01", "2014-01-01T00:00:00", "2014-01-01T00:00:00.0000001Z", "2014-01-01T00:00:00.1Z", datetime(2014, 1, 1), 1388534400, None]
        for bad in invalid:
            with self.subTest(value=bad), self.assertRaises((ValueError, TypeError)):
                utc_seconds(bad)

    def test_missing_provenance_is_not_guessed(self):
        for field in ("known_at", "effective_at", "source_available_at", "source_ref"):
            record = event()
            del record[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing required provenance"):
                PointInTimePrior(2, [record])

    def test_duplicate_identical_events_are_not_double_counted(self):
        prior = PointInTimePrior(2, [event(), event()])
        self.assertEqual(len(prior.events), 1)
        self.assertEqual(prior(at("2014-01-01T00:00:00Z"))[0, 0, 0, 1].item(), 0.5)

    def test_conflicting_same_activation_events_are_rejected(self):
        for update in ({"action": "remove"}, {"weight": 2}, {"source_ref": "conflicting evidence"}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "conflicting events"):
                PointInTimePrior(2, [event(), event(**update)])

    def test_relation_identity_cannot_change(self):
        for update in ({"source": 1, "target": 0}, {"bidirectional": False}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "identity cannot change"):
                PointInTimePrior(2, [event(), event(when="2015-01-01T00:00:00Z", **update)])

    def test_directed_edges_remain_asymmetric(self):
        result = PointInTimePrior(2, [event(bidirectional=False, weight=3)])(at("2014-01-01T00:00:00Z"))
        torch.testing.assert_close(result[0, 0], torch.tensor([[0.25, 0.75], [0.0, 1.0]]))

    def test_distinct_relations_sum_and_removal_affects_only_one_id(self):
        prior = PointInTimePrior(2, [event(), event("r2", weight=2), event(when="2015-01-01T00:00:00Z", action="remove")])
        result = prior(at("2014-01-01T00:00:00Z", "2015-01-01T00:00:00Z"))
        self.assertEqual(result[0, 0, 0, 1].item(), 0.75)
        torch.testing.assert_close(result[0, 1], torch.tensor([[1 / 3, 2 / 3], [2 / 3, 1 / 3]]))

    def test_reactivation_and_weight_revision_follow_activation_order(self):
        records = [event(), event(when="2015-01-01T00:00:00Z", action="remove"), event(when="2016-01-01T00:00:00Z", weight=3)]
        result = PointInTimePrior(2, list(reversed(records)))(at("2014-01-01T00:00:00Z", "2015-01-01T00:00:00Z", "2016-01-01T00:00:00Z"))
        self.assertEqual(result[0, 0, 0, 1].item(), 0.5)
        self.assertEqual(result[0, 1, 0, 1].item(), 0.0)
        self.assertEqual(result[0, 2, 0, 1].item(), 0.75)

    def test_normalized_event_provenance_round_trips_including_removals(self):
        prior = PointInTimePrior(2, [event(), event(when="2015-01-01T00:00:00Z", action="remove")])
        rebuilt = PointInTimePrior(2, prior.events)
        cutoffs = at("2014-01-01T00:00:00Z", "2015-01-01T00:00:00Z")
        torch.testing.assert_close(prior(cutoffs), rebuilt(cutoffs))

    def test_invalid_or_unrepresentable_weights_rejected(self):
        for weight in (0, -1, True, float("inf"), float("nan"), 1e100, 1e-100):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "weight"):
                PointInTimePrior(2, [event(weight=weight)])

    def test_empty_evidence_returns_only_self_loops_and_batch_shape(self):
        cutoffs = at("2014-01-01T00:00:00Z", "2015-01-01T00:00:00Z").expand(3, -1)
        result = PointInTimePrior(3, [])(cutoffs)
        self.assertEqual(result.shape, (3, 2, 3, 3))
        torch.testing.assert_close(result, torch.eye(3).expand(3, 2, -1, -1))

    def test_unknown_or_ambiguous_cutoffs_are_rejected(self):
        invalid = [torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64), torch.tensor([[torch.iinfo(torch.int64).min]]), torch.empty(0, 1, dtype=torch.int64), [[1388534400]]]
        for bad in invalid:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                PointInTimePrior(2, [event()])(bad)

    def test_module_dtype_conversion_preserves_temporal_integer_buffers(self):
        prior = PointInTimePrior(2, [event()]).double()
        self.assertEqual(prior.interval_starts.dtype, torch.int64)
        result = prior(at("2014-01-01T00:00:00Z"))
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(result.sum(-1), torch.ones(1, 1, 2, dtype=torch.float64))

    def test_same_graph_checkpoint_load_is_allowed(self):
        prior = PointInTimePrior(2, [event()])
        restored = PointInTimePrior(2, [event()])
        restored.load_state_dict(prior.state_dict())
        torch.testing.assert_close(restored(at("2014-01-01T00:00:00Z")), prior(at("2014-01-01T00:00:00Z")))

    def test_checkpoint_cannot_backdate_activation_or_change_edges(self):
        prior = PointInTimePrior(2, [event(when="2018-01-05T00:00:00Z")])
        cutoffs = at("2014-01-01T00:00:00Z")
        for name in ("interval_starts", "interval_ends", "edge_indices", "edge_weights"):
            with self.subTest(buffer=name):
                state = {key: value.clone() for key, value in prior.state_dict().items()}
                state[name][0] -= 1
                if name == "interval_starts":
                    state[name][0] = utc_seconds("2014-01-01T00:00:00Z")
                with self.assertRaisesRegex(RuntimeError, "immutable provenance-derived graph mismatch"):
                    prior.load_state_dict(state)
                torch.testing.assert_close(prior(cutoffs)[0, 0], torch.eye(2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional")
    def test_cuda_device_buffers_and_cutoffs(self):
        prior = PointInTimePrior(2, [event()]).cuda()
        result = prior(at("2014-01-01T00:00:00Z").cuda())
        self.assertTrue(result.is_cuda)
        with self.assertRaisesRegex(ValueError, "same device"):
            prior(at("2014-01-01T00:00:00Z"))


if __name__ == "__main__":
    unittest.main()

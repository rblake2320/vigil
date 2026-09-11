"""Actual-trace-derived association regression; no new image inference claim."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from live_fall_capture import TargetTracker,FrameRetention
from detectors.temporal_fall import TemporalFallDetector,PersonObservation

def normalized(box):return (box[0]/320,box[1]/180,box[2]/320,box[3]/180)

class BriefDropoutTests(unittest.TestCase):
    def test_actual_trace_empty_frame_preserves_established_identity_only(self):
        tracker=TargetTracker();detector=TemporalFallDetector()
        box=normalized((104.3,34.2,165.0,158.5))
        for time in (8.493,8.743,8.993,9.243,9.493,9.570):
            index,reason=tracker.select([box],320/180,time)
            detector.update(time,[PersonObservation(str(tracker.epoch),box)],frame_aspect_ratio=320/180)
        epoch=tracker.epoch
        self.assertTrue(detector._tracks[str(epoch)].baseline_ready)
        self.assertEqual(tracker.select([],320/180,9.581),(None,'missing_target_brief'))
        detector.pause_observation()
        newbox=normalized((108.6,33.8,152.4,157.9))
        self.assertEqual(tracker.select([newbox],320/180,9.597)[0],0)
        self.assertEqual(tracker.epoch,epoch)
        detector.update(9.597,[PersonObservation(str(epoch),newbox)],frame_aspect_ratio=320/180)
        self.assertTrue(detector._tracks[str(epoch)].baseline_ready)
    def test_different_person_overlimit_and_ambiguity_do_not_inherit(self):
        upright=(.1,.1,.2,.7);far=(.8,.1,.9,.7)
        tracker=TargetTracker();tracker.select([upright],1,1)
        self.assertIsNone(tracker.select([far],1,1.1)[0])
        self.assertEqual(tracker.epoch,1)
        self.assertEqual(tracker.select([upright],1,1.51),(None,'target_observation_gap'))
        tracker=TargetTracker();tracker.select([upright],1,1)
        self.assertEqual(tracker.select([upright,(.11,.1,.21,.7)],1,1.1),(None,'ambiguous_target_match'))
        self.assertIsNone(tracker.box)
    def test_missing_observation_never_advances_horizontal_hold(self):
        d=TemporalFallDetector();up=(.4,.1,.55,.7);down=(.2,.6,.8,.85)
        for t in (0,.25,.5):d.update(t,[PersonObservation('a',up)])
        d.update(.75,[PersonObservation('a',down)]);d.pause_observation()
        self.assertIsNone(d._tracks['a'].horizontal_at)
        self.assertFalse(d.update(1.2,[PersonObservation('a',down)]).candidates)
        for t in (1.45,1.7,1.95):self.assertFalse(d.update(t,[PersonObservation('a',down)]).candidates)
        self.assertEqual(len(d.update(2.2,[PersonObservation('a',down)]).candidates),1)
    def test_retained_samples_bounded_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            r=FrameRetention(Path(root)/'private',cap=6)
            first=r.save(0,1,b'abc')
            self.assertTrue(Path(first['path']).exists())
            self.assertIsNone(r.save(.1,2,b'abc'))
            self.assertIsNotNone(r.save(.2,3,b'def'))
            with self.assertRaises(ValueError):r.save(.4,4,b'g')
            self.assertEqual(len(list(r.directory.iterdir())),2)

if __name__=='__main__':unittest.main()

"""Synthetic kinematics verify state rules, never real-video detection accuracy."""
import math
import unittest
from detectors.temporal_fall import TemporalFallDetector, PersonObservation

UP=(0.4,0.1,0.55,0.7)
DOWN=(0.2,0.6,0.8,0.85)
def person(box=UP, identity="a"):
    return PersonObservation(identity,box)

class TemporalFallTests(unittest.TestCase):
    def baseline(self,d):
        for time in (0.0,0.25,0.5): d.update(time,[person()])
    def test_transition_hold_and_no_duplicate_candidate(self):
        d=TemporalFallDetector();self.baseline(d)
        results=[d.update(t,[person(DOWN)]) for t in (0.75,1,1.25,1.5,1.75,2)]
        candidates=[c for r in results for c in r.candidates]
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0].observed_at,1.75)
        self.assertEqual(candidates[0].status,"possible_fall")
    def test_already_lying_and_no_drop_are_not_candidates(self):
        for box in (DOWN,(0.2,0.3,0.8,0.5)):
            d=TemporalFallDetector()
            if box!=DOWN:self.baseline(d)
            start=0 if box==DOWN else 0.75
            self.assertFalse(any(d.update(start+i*0.25,[person(box)]).candidates for i in range(16)))
    def test_other_person_cannot_inherit_baseline(self):
        d=TemporalFallDetector();self.baseline(d)
        results=[d.update(0.75+i*0.25,[person(DOWN,"b")]) for i in range(8)]
        self.assertFalse(any(r.candidates for r in results))
        self.assertEqual(results[0].status,"insufficient_observation")
    def test_missing_frame_gap_and_ambiguity_reset(self):
        for kw in ({"frame_valid":False},{"identity_ambiguous":True},{}):
            d=TemporalFallDetector();self.baseline(d)
            t=1.5 if not kw else 0.75
            self.assertEqual(d.update(t,[person(DOWN)],**kw).status,"insufficient_observation")
            self.assertFalse(any(d.update(t+(i+1)*0.25,[person(DOWN)]).candidates for i in range(8)))
    def test_brief_absence_resumes_without_crediting_gap(self):
        d=TemporalFallDetector();self.baseline(d);d.update(.75,[person(DOWN)])
        self.assertEqual(d.update(1,[]).status,"insufficient_observation")
        for t in (1.25,1.5,1.75,2):
            self.assertFalse(d.update(t,[person(DOWN)]).candidates)
        candidate=d.update(2.25,[person(DOWN)]).candidates[0]
        self.assertAlmostEqual(candidate.observed_horizontal_seconds,1.0)
        self.assertAlmostEqual(candidate.horizontal_span_seconds,1.5)
    def test_bad_timestamps_and_boxes_and_ids_refuse(self):
        for value in (math.nan,math.inf,-1,True):
            self.assertEqual(TemporalFallDetector().update(value,[person()]).status,"insufficient_observation")
        d=TemporalFallDetector();d.update(1,[person()])
        self.assertEqual(d.update(1,[person()]).reasons,("nonmonotonic_timestamp",))
        for box in ((0,0,0,1),(0,0,math.nan,1),(-1,0,1,1),(1,1,0,0)):
            self.assertEqual(TemporalFallDetector().update(0,[person(box)]).status,"insufficient_observation")
        self.assertEqual(TemporalFallDetector().update(0,[person(),person()]).status,"insufficient_observation")
    def test_slow_transition_does_not_trigger(self):
        d=TemporalFallDetector();self.baseline(d)
        for t in (.75,1,1.25,1.5,1.75,2,2.25):d.update(t,[person((.3,.3,.7,.7))])
        self.assertFalse(any(d.update(2.5+i*.25,[person(DOWN)]).candidates for i in range(8)))
    def test_two_tracks_independent(self):
        d=TemporalFallDetector()
        for t in (0,.25,.5):d.update(t,[person(),person(UP,"b")])
        found=[]
        for t in (.75,1,1.25,1.5,1.75):found.extend(d.update(t,[person(DOWN),person(UP,"b")]).candidates)
        self.assertEqual([x.track_id for x in found],["a"])
    def test_invalid_threshold_refused(self):
        for value in (0,-1,math.nan,True):
            with self.assertRaises(ValueError):TemporalFallDetector(horizontal_hold=value)
    def test_same_pixel_pose_across_portrait_and_landscape(self):
        # Identical pixel boxes on canvases with the same height but distinct
        # widths: normalized x alone must not change posture classification.
        outputs=[]
        for width,height in ((500,1000),(1600,1000)):
            def box(pixel):return (pixel[0]/width,pixel[1]/height,pixel[2]/width,pixel[3]/height)
            upright=box((200,100,300,700))
            horizontal=box((100,600,450,750))
            d=TemporalFallDetector();found=[]
            for t in (0,.25,.5):d.update(t,[person(upright)],frame_aspect_ratio=width/height)
            for t in (.75,1,1.25,1.5,1.75):found.extend(d.update(t,[person(horizontal)],frame_aspect_ratio=width/height).candidates)
            outputs.append(found)
        self.assertEqual(len(outputs[0]),1)
        self.assertEqual(outputs[0],outputs[1])
    def test_invalid_frame_aspect_ratio_refused(self):
        for value in (0,-1,math.nan,math.inf,True):
            self.assertEqual(TemporalFallDetector().update(0,[person()],frame_aspect_ratio=value).reasons,("invalid_frame_aspect_ratio",))

if __name__=="__main__":unittest.main()

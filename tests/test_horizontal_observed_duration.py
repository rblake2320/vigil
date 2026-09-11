from detectors.temporal_fall import TemporalFallDetector,PersonObservation

def u():return PersonObservation("a",(.4,.1,.6,.5))
def h(who="a"):return PersonObservation(who,(.2,.52,.8,.72))
def prepared():
 d=TemporalFallDetector()
 for t in (0,.25,.5):d.update(t,[u()])
 for t in (.75,1,1.25,1.5,1.63):d.update(t,[h()])
 return d

def test_point88_gap_point2_observed_accepts():
 d=prepared();d.pause_observation()
 assert not d.update(1.8,[h()]).candidates
 candidate=d.update(2,[h()]).candidates[0]
 assert abs(candidate.observed_horizontal_seconds-1.08)<1e-9
 assert abs(candidate.horizontal_span_seconds-1.25)<1e-9
 assert candidate.hold_policy=="observed_intervals_within_2s_v1"

def test_pause_alone_no_credit():
 d=prepared();d.pause_observation()
 assert not d.update(1.9,[h()]).candidates

def test_nonhorizontal_resets():
 d=prepared();d.update(1.7,[PersonObservation("a",(.3,.3,.7,.7))])
 assert not d.update(1.9,[h()]).candidates

def test_episode_over_two_seconds_refused():
 d=prepared()
 for t in (1.7,1.9,2.1,2.3,2.5,2.7):
  d.pause_observation();assert not d.update(t,[h()]).candidates
 assert not d.update(2.8,[h()]).candidates

def test_long_gap_resets():
 d=prepared();d.update(1.8,[]);d.update(2,[])
 assert not d.update(2.2,[h()]).candidates

def test_changed_identity_no_inheritance():
 d=prepared();d.pause_observation()
 assert not d.update(1.8,[h("b")]).candidates
 assert not d.update(2,[h("b")]).candidates

def test_ambiguity_resets():
 d=prepared();d.update(1.7,[h()],identity_ambiguous=True)
 assert not d.update(1.9,[h()]).candidates


def test_unidentified_empty_update_clears_episode():
 d=prepared();d.update(1.7,[])
 assert not d.update(1.8,[h()]).candidates
 assert not d.update(2,[h()]).candidates

"""Synthetic geometry only: baseline-history regression, no visual inference."""
import pytest
from detectors.temporal_fall import TemporalFallDetector,PersonObservation

def upright(center=.3,who="a"):
    return PersonObservation(who,(.4,center-.2,.6,center+.2))
def horizontal(center=.62,who="a"):
    return PersonObservation(who,(.2,center-.1,.8,center+.1))
def establish(d):
    for t in (0,.25,.5):d.update(t,[upright()])
def hold(d,start,who="a",center=.62):
    results=[d.update(start+i*.25,[horizontal(center,who)]) for i in range(5)]
    return [c for r in results for c in r.candidates]

def test_intermediate_upright_descent_retains_established_origin():
    d=TemporalFallDetector();establish(d)
    d.update(.75,[upright(.42)]);d.update(1,[upright(.54)])
    candidates=hold(d,1.25)
    assert len(candidates)==1
    assert candidates[0].upright_at==.5
    assert candidates[0].center_drop==pytest.approx(.32)

def test_slow_descent_old_baseline_expires():
    d=TemporalFallDetector();establish(d)
    for i in range(1,25):d.update(.5+i*.25,[upright(.3+i*.01)])
    assert not hold(d,6.75,center=.60)

def test_old_baseline_with_sitting_does_not_survive_window():
    d=TemporalFallDetector();establish(d)
    square=PersonObservation("a",(.3,.35,.7,.75))
    for t in (.75,1,1.25,1.5,1.75,2,2.25):d.update(t,[square])
    assert not hold(d,2.5)

@pytest.mark.parametrize("break_kind",["missing","ambiguous","gap","identity"])
def test_no_baseline_inheritance(break_kind):
    d=TemporalFallDetector();establish(d)
    if break_kind=="missing":d.update(.75,[])
    elif break_kind=="ambiguous":d.update(.75,[upright()],identity_ambiguous=True)
    elif break_kind=="gap":d.update(1.25,[upright()])
    else:d.update(.75,[upright(who="b")])
    start=1.5 if break_kind=="gap" else 1
    assert not hold(d,start,who="b" if break_kind=="identity" else "a")

def test_unestablished_early_sample_never_qualifies():
    d=TemporalFallDetector()
    d.update(0,[upright(.25)]);d.update(.25,[upright(.4)]);d.update(.5,[upright(.54)])
    assert not hold(d,.75)

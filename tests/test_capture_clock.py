"""Deterministic boundary tests; literals are fixtures, not measured clock proof."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from live_fall_capture import Freshness

class CaptureClockTests(unittest.TestCase):
    def test_equal_sample_skipped_without_inventing_elapsed_time(self):
        f=Freshness();self.assertIsNone(f.observe('first',1,1))
        self.assertEqual(f.observe('different',1,1.01),'skip_equal_arrival_timestamp')
        self.assertEqual(f.last_arrival,1);self.assertEqual(f.last_hash,'first')
        self.assertIsNone(f.observe('different',1.0000001,1.02))
        self.assertEqual(f.last_arrival,1.0000001)
    def test_equal_frozen_time_eventually_expires(self):
        f=Freshness();f.observe('a',1,1)
        self.assertEqual(f.observe('b',1,1.2),'skip_equal_arrival_timestamp')
        self.assertEqual(f.observe('c',1,1.6),'stale_or_invalid_host_arrival')
    def test_backwards_time_still_refuses(self):
        f=Freshness();f.observe('a',1,1)
        self.assertEqual(f.observe('b',.99,1),'nonmonotonic_host_arrival')

if __name__=='__main__':unittest.main()

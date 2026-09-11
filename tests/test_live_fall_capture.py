"""Transport boundary tests, no device/model or fall accuracy claims."""
import io
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from live_fall_capture import parse_roi,commands,read_frame,Freshness,TargetTracker,event_command

class LiveCaptureTests(unittest.TestCase):
    def test_roi_and_exact_serial_commands(self):
        roi=parse_roi('10,20,640,360')
        capture,decode,w,h=commands('adb.exe','ffmpeg.exe','ZA223JQDL4',roi,180)
        self.assertEqual(capture[1:4],['-s','ZA223JQDL4','exec-out'])
        self.assertEqual(capture[-1],'-');self.assertEqual((w,h),(320,180))
        self.assertIn('crop=640:360:10:20,scale=320:180',decode)
        self.assertNotIn('fps=5',' '.join(decode))
    def test_bad_roi_and_serial_refused(self):
        for value in ('-1,0,300,300','0,0,1,1','0,0,99999,500','a,0,100,100','0,0,100'):
            with self.assertRaises(ValueError):parse_roi(value)
        with self.assertRaises(ValueError):commands('adb','ffmpeg','x;command',(0,0,100,100),10)
    def test_partial_pipe_reads_and_truncation(self):
        class Short(io.BytesIO):
            def read(self,n):return super().read(min(n,2))
        self.assertEqual(read_frame(Short(b'123456'),6),b'123456')
        self.assertIsNone(read_frame(io.BytesIO(b''),6))
        with self.assertRaises(ValueError):read_frame(Short(b'12345'),6)
    def test_stale_identical_and_nonmonotonic_refused(self):
        f=Freshness();self.assertIsNone(f.observe('a',1,1.1))
        self.assertEqual(f.observe('a',1.2,1.3),'skip_duplicate_frame')
        self.assertEqual(f.observe('b',1.1,1.2),'nonmonotonic_host_arrival')
        self.assertEqual(f.observe('c',2,3),'stale_or_invalid_host_arrival')
    def test_intermittent_duplicates_do_not_reset_temporal_continuity(self):
        from detectors.temporal_fall import TemporalFallDetector,PersonObservation
        fresh=Freshness();detector=TemporalFallDetector();found=[]
        for i in range(16):
            t=i*.125;digest=str(i//2)
            decision=fresh.observe(digest,t,t)
            if i%2:self.assertEqual(decision,'skip_duplicate_frame');continue
            box=(.4,.1,.55,.7) if t<=.5 else (.2,.6,.8,.85)
            found.extend(detector.update(t,[PersonObservation('a',box)],frame_valid=decision is None).candidates)
        self.assertEqual(len(found),1)
    def test_prolonged_freeze_resets_instead_of_preserving_baseline(self):
        from detectors.temporal_fall import TemporalFallDetector,PersonObservation
        fresh=Freshness();detector=TemporalFallDetector()
        for t in (0,.25,.5):
            self.assertIsNone(fresh.observe(str(t),t,t));detector.update(t,[PersonObservation('a',(.4,.1,.55,.7))])
        self.assertEqual(fresh.observe('0.5',.75,.75),'skip_duplicate_frame')
        decision=fresh.observe('0.5',1,1)
        self.assertEqual(decision,'identical_display_frame')
        self.assertEqual(detector.update(1,[],frame_valid=False).status,'insufficient_observation')
        self.assertFalse(any(detector.update(1.25+i*.25,[PersonObservation('a',(.2,.6,.8,.85))]).candidates for i in range(8)))
    def test_target_largest_upright_other_people_and_ambiguity(self):
        t=TargetTracker();standing=(.4,.1,.55,.8);seated=(.05,.5,.25,.7)
        self.assertEqual(t.select([seated,standing],1)[0],1)
        self.assertEqual(t.select([seated,(.41,.11,.56,.8)],1)[0],1)
        self.assertIsNone(t.select([standing,(.42,.12,.57,.81)],1)[0])
        self.assertIsNone(t.box)
        self.assertIsNone(t.select([standing,(.42,.12,.57,.81)],1)[0])
    def test_callback_argv_binds_one_event_without_shell(self):
        self.assertEqual(event_command('["python", "adapter.py", "--event", "{event_file}"]',Path('receipt.json')),['python','adapter.py','--event','receipt.json'])
        for raw in ('"echo test"','[]','["python","adapter.py"]','["x","{event_file}","{event_file}"]'):
            with self.assertRaises(ValueError):event_command(raw,Path('receipt.json'))

if __name__=='__main__':unittest.main()

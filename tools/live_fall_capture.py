"""Bounded Android display capture -> CPU possible-fall candidate receipt.

This observes a cropped DISPLAY, not a trusted camera stream. An operator must
verify Blink's live view/ROI separately. No SMS, webhook, cloud or alert adapter.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detectors.temporal_fall import TemporalFallDetector, PersonObservation, FrameResult


def parse_roi(text):
    try: values=tuple(int(v) for v in text.split(','))
    except (ValueError,AttributeError): raise ValueError('ROI must be x,y,width,height integers')
    if len(values)!=4 or min(values[:2])<0 or min(values[2:])<32 or max(values)>8192:
        raise ValueError('Invalid bounded ROI')
    return values


def commands(adb,ffmpeg,serial,roi,seconds):
    if not serial or len(serial)>128 or any(not(c.isalnum() or c in '._:-') for c in serial): raise ValueError('Invalid explicit device serial')
    x,y,w,h=roi
    width=320
    height=max(2,round(h/w*width/2)*2)
    if height>1280: raise ValueError('ROI aspect ratio too tall')
    capture=[str(adb),'-s',serial,'exec-out','screenrecord','--output-format=h264','--time-limit',str(seconds),'-']
    decode=[str(ffmpeg),'-hide_banner','-loglevel','warning','-nostdin','-threads','2',
            '-flags','low_delay','-probesize','32768','-analyzeduration','0',
            '-f','h264','-i','pipe:0','-vf',f'crop={w}:{h}:{x}:{y},scale={width}:{height}',
            '-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    return capture,decode,width,height


def read_frame(stream,size):
    chunks=[];remaining=size
    while remaining:
        chunk=stream.read(remaining)
        if not chunk:
            if chunks: raise ValueError('Truncated decoded frame')
            return None
        chunks.append(chunk);remaining-=len(chunk)
    return b''.join(chunks)


class Freshness:
    """Host decoder arrival time is transport evidence, never camera capture time."""
    def __init__(self): self.last_hash=None;self.last_arrival=None;self.last_distinct=None
    def observe(self,digest,arrival,now):
        reason=None
        if not all(math.isfinite(v) for v in (arrival,now)) or now<arrival or now-arrival>0.5:
            reason='stale_or_invalid_host_arrival'
        elif self.last_arrival is not None and arrival<self.last_arrival:
            reason='nonmonotonic_host_arrival'
        elif self.last_arrival is not None and arrival==self.last_arrival:
            return 'skip_equal_arrival_timestamp'
        elif self.last_hash==digest:
            reason='identical_display_frame' if self.last_distinct is None or arrival-self.last_distinct>=.5 else 'skip_duplicate_frame'
        else:
            self.last_distinct=arrival
        if reason not in (None,'skip_duplicate_frame'):
            self.last_distinct=None
        self.last_hash=digest;self.last_arrival=arrival
        return reason


class TargetTracker:
    """Acquire largest unambiguous upright person; never inherit a lost baseline."""
    def __init__(self):self.box=None;self.epoch=0;self.last_seen=None
    def select(self,boxes,aspect,timestamp=0.0):
        if not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool) or not math.isfinite(timestamp) or timestamp<0:
            self.box=None;self.last_seen=None;return None,'invalid_target_timestamp'
        def area(b):return (b[2]-b[0])*(b[3]-b[1])
        if self.box is None:
            upright=[(i,b) for i,b in enumerate(boxes) if (b[3]-b[1])/((b[2]-b[0])*aspect)>=1.4]
            upright.sort(key=lambda pair:area(pair[1]),reverse=True)
            if not upright:return None,'no_upright_target'
            if len(upright)>1 and area(upright[0][1])<1.25*area(upright[1][1]):return None,'ambiguous_acquisition'
            index,box=upright[0];self.box=box;self.epoch+=1;self.last_seen=timestamp;return index,None
        if self.last_seen is None or timestamp<self.last_seen or timestamp-self.last_seen>.5:
            self.box=None;self.last_seen=None;return None,'target_observation_gap'
        old=self.box;matches=[]
        for i,b in enumerate(boxes):
            oldcenter=((old[0]+old[2])/2,(old[1]+old[3])/2)
            center=((b[0]+b[2])/2,(b[1]+b[3])/2)
            intersection=max(0,min(old[2],b[2])-max(old[0],b[0]))*max(0,min(old[3],b[3])-max(old[1],b[1]))
            iou=intersection/(area(old)+area(b)-intersection)
            if .4<=area(b)/area(old)<=2.5 and (iou>=.1 or math.dist(oldcenter,center)<=.35):matches.append(i)
        if not matches:return None,'missing_target_brief'
        if len(matches)>1:
            self.box=None;self.last_seen=None;return None,'ambiguous_target_match'
        index=matches[0];self.box=boxes[index];self.last_seen=timestamp;return index,None


def event_command(raw,event_path):
    command=json.loads(raw)
    if not isinstance(command,list) or not command or len(command)>20 or any(not isinstance(v,str) or not v for v in command):raise ValueError('Event command must be argv JSON list')
    if sum(v.count('{event_file}') for v in command)!=1:raise ValueError('Event command requires one {event_file} placeholder')
    return [v.replace('{event_file}',str(event_path)) for v in command]


class ActivationGate:
    """Owner-controlled local sentinel; never selected from camera/model content."""
    def __init__(self,path=None):self.path=Path(path) if path else None;self.active=False
    def sample(self):
        present=self.path is None or self.path.is_file()
        activated=present and not self.active
        self.active=present
        return present,activated


class FrameRetention:
    """Private sampled evidence, <=5Hz and hard cumulative 100MB maximum."""
    def __init__(self,directory,cap=100_000_000):
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.last=None;self.bytes=0;self.cap=cap
    def due(self,timestamp):return self.last is None or timestamp-self.last>=.2-1e-9
    def save(self,timestamp,sequence,png):
        if not self.due(timestamp):return None
        if not png or self.bytes+len(png)>self.cap:raise ValueError('Private frame evidence exceeds hard 100MB cap or is empty')
        path=self.directory/f'frame-{sequence:07d}.png'
        with path.open('xb') as f:f.write(png);f.flush();os.fsync(f.fileno())
        self.bytes+=len(png);self.last=timestamp
        return {'path':str(path),'png_sha256':hashlib.sha256(png).hexdigest()}


def write_json(path,value):
    if path.exists(): raise FileExistsError(f'Preserve existing artifact: {path}')
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2);handle.flush();os.fsync(handle.fileno())
    tmp.replace(path)


def stop_process(process):
    if process is None or process.poll() is not None:return
    process.terminate()
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)


def worker(args):
    # All inference calls explicitly use CPU, regardless of host GPU state.
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2'
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(2)
    output=Path(args.out);roi=parse_roi(args.roi)
    adb=Path(args.adb).resolve(strict=True);ffmpeg=Path(args.ffmpeg).resolve(strict=True)
    weights=Path(args.weights).resolve(strict=True)
    if not all(p.is_file() for p in (adb,ffmpeg,weights)):raise ValueError('Executable/model paths must be existing files')
    if weights.stat().st_size>100_000_000:raise ValueError('Weights exceed 100 MB')
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=args.weights_sha256:raise ValueError('Weights hash mismatch')
    capture_cmd,decode_cmd,w,h=commands(adb,ffmpeg,args.serial,roi,args.seconds)
    model=YOLO(str(weights),task='pose')
    if time.monotonic()>=args.deadline_monotonic-5:raise TimeoutError('No capture budget remains after initialization')
    detector=TemporalFallDetector();fresh=Freshness();tracker=TargetTracker();activation=ActivationGate(args.activation_file)
    retention=FrameRetention(output/'private-frames') if args.retain_frames else None
    frames=queue.Queue(maxsize=1);problems=queue.Queue();done=threading.Event()
    capture_start_ns=time.perf_counter_ns();capture=None;decode=None;events=[];decoded=0;analyzed=0;usable=0;reason='duration_limit'
    trace_path=output/'trace.jsonl'
    def pump():
        nonlocal decoded
        try:
            while not done.is_set():
                frame=read_frame(decode.stdout,w*h*3)
                if frame is None:break
                decoded+=1;item=(decoded,time.perf_counter_ns(),time.time(),frame)
                try:frames.get_nowait()
                except queue.Empty:pass
                frames.put_nowait(item)
        except Exception as error:problems.put(type(error).__name__+': '+str(error))
        finally:done.set()
    try:
        with (output/'adb-stderr.txt').open('xb') as ae,(output/'ffmpeg-stderr.txt').open('xb') as fe,trace_path.open('x',encoding='utf-8') as trace:
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            capture=subprocess.Popen(capture_cmd,stdout=subprocess.PIPE,stderr=ae,creationflags=flags)
            decode=subprocess.Popen(decode_cmd,stdin=capture.stdout,stdout=subprocess.PIPE,stderr=fe,creationflags=flags)
            capture.stdout.close()
            reader=threading.Thread(target=pump,daemon=True);reader.start()
            while time.monotonic()<args.deadline_monotonic-5:
                try:seq,arrival_ns,wall,raw=frames.get(timeout=.25)
                except queue.Empty:
                    if done.is_set():reason='stream_ended';break
                    continue
                arrival=(arrival_ns-capture_start_ns)/1_000_000_000
                capture_now=(time.perf_counter_ns()-capture_start_ns)/1_000_000_000
                digest=hashlib.sha256(raw).hexdigest();stale=fresh.observe(digest,arrival,capture_now)
                active,just_activated=activation.sample()
                if just_activated:
                    detector=TemporalFallDetector();tracker=TargetTracker()
                if stale in ('skip_duplicate_frame','skip_equal_arrival_timestamp'):
                    # Display encoder duplicates are not new observations. They
                    # neither advance nor reset the candidate timer; a freeze
                    # lasting .5s still invalidates continuity below.
                    trace.write(json.dumps({'sequence':seq,'host_decode_arrival_epoch':wall,'host_decode_arrival_perf_ns':arrival_ns,'capture_start_perf_ns':capture_start_ns,'host_elapsed_seconds':arrival,'timestamp_kind':'host_decode_arrival_NOT_camera_time','frame_sha256':digest,'status':'duplicate_skipped','stale_reason':stale})+'\n');trace.flush()
                    continue
                boxes=[];torso=[];all_torso=[];people=[];ambiguous=False;target_reason=None;retained=None
                if retention is not None and retention.due(arrival):
                    evidence_frame=np.frombuffer(raw,dtype=np.uint8).reshape((h,w,3))
                    encoded,png=cv2.imencode('.png',evidence_frame)
                    if not encoded:raise ValueError('Private evidence PNG encoding failed')
                    retained=retention.save(arrival,seq,png.tobytes())
                if not stale:
                    frame=np.frombuffer(raw,dtype=np.uint8).reshape((h,w,3))
                    result=model.predict(frame,imgsz=320,device='cpu',verbose=False,conf=.4)[0]
                    boxes=result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
                    valid_boxes=[];valid_conf=[]
                    if boxes and result.keypoints is not None:
                        conf=result.keypoints.conf
                        for i,pixel_box in enumerate(boxes):
                            confidence=[float(conf[i,k]) for k in (5,6,11,12)] if conf is not None else []
                            all_torso.append(confidence)
                            if len(confidence)==4 and all(v>=.35 for v in confidence):
                                x1,y1,x2,y2=pixel_box;box=(max(0,x1/w),max(0,y1/h),min(1,x2/w),min(1,y2/h))
                                if box[2]>box[0] and box[3]>box[1]:valid_boxes.append(box);valid_conf.append(confidence)
                    target,target_reason=tracker.select(valid_boxes,w/h,arrival)
                    if target is not None:people=[PersonObservation(str(tracker.epoch),valid_boxes[target])];torso=valid_conf[target]
                    else:ambiguous=True
                else:tracker.box=None
                # Inference taking too long is also insufficient current observation.
                if (time.perf_counter_ns()-arrival_ns)/1_000_000_000>.5:stale='inference_or_transport_age_exceeded'
                if not active:state=FrameResult('awaiting_activation')
                elif stale is None and target_reason=='missing_target_brief':
                    detector.pause_observation()
                    state=FrameResult('insufficient_observation',reasons=('brief_missing_target_no_time_credit',))
                else:state=detector.update(arrival,people,frame_valid=stale is None,identity_ambiguous=ambiguous,frame_aspect_ratio=w/h)
                analyzed+=1;usable+=state.status not in ('insufficient_observation','awaiting_activation')
                row={'sequence':seq,'host_decode_arrival_epoch':wall,'host_decode_arrival_perf_ns':arrival_ns,'capture_start_perf_ns':capture_start_ns,'host_elapsed_seconds':arrival,'timestamp_kind':'host_decode_arrival_NOT_camera_time','frame_sha256':digest,'retained_frame':retained,'stale_reason':stale,'boxes_pixels':boxes,'torso_confidences':torso,'all_torso_confidences':all_torso,'target_reason':target_reason,'target_epoch':tracker.epoch,'tracks':[dataclasses.asdict(p) for p in people],'status':state.status,'reasons':state.reasons}
                trace.write(json.dumps(row)+'\n');trace.flush()
                for candidate in state.candidates:
                    trace.flush();os.fsync(trace.fileno())
                    event={'schema':1,'event_id':str(uuid.uuid4()),'kind':'possible_fall','source':'android_display_roi','requires_human_review':True,'observed_at_epoch':wall,'expires_at_epoch':wall+30,'timestamp_kind':'host_decode_arrival_NOT_camera_time','frame_sha256':digest,'trace_prefix_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),'weights_sha256':args.weights_sha256,'roi':roi,'candidate':dataclasses.asdict(candidate),'callback_configured':bool(args.event_command_json)}
                    event_path=output/f"event-{event['event_id']}.json"
                    write_json(event_path,event);events.append(event)
                    if args.event_command_json:
                        write_json(output/'dispatch-intent.json',{'event_id':event['event_id'],'status':'INTENT','automatic_retry':False})
                        command=event_command(args.event_command_json,event_path)
                        try:
                            result=subprocess.run(command,capture_output=True,timeout=min(30,max(.1,args.deadline_monotonic-time.monotonic()-2)),creationflags=flags)
                            # Callback success is not recipient delivery proof.
                            write_json(output/'dispatch-result.json',{'event_id':event['event_id'],'status':'CALLBACK_RETURNED','returncode':result.returncode,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'recipient_delivery_verified':False})
                        except Exception as error:write_json(output/'dispatch-result.json',{'event_id':event['event_id'],'status':'UNKNOWN','error_type':type(error).__name__,'automatic_retry':False})
                    reason='candidate_emitted_stop';break
                if events:break
            if not problems.empty():raise ValueError(problems.get())
            trace.flush();os.fsync(trace.fileno())
    finally:
        done.set();stop_process(decode);stop_process(capture)
    receipt={'schema':1,'status':'possible_fall_review_required' if events else 'no_candidate_observed' if usable else 'insufficient_observation','terminal_reason':reason,'decoded_frames':decoded,'analyzed_frames':analyzed,'events':events,'weights_sha256':args.weights_sha256,'roi':roi,'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),'callback_configured':bool(args.event_command_json),'recipient_delivery_verified':False,'limits':['Display transport timestamps are not Blink camera timestamps.','Operator must separately verify the live app and ROI; this tool cannot authenticate displayed content.','Display overlays, cuts, camera motion, buffering and nearby person swaps can invalidate inference.','Identical frames are refused; nonidentical rendered stale content can remain undetected.','Largest-upright acquisition with ambiguous-match refusal is heuristic track continuity, not identity verification.','Box-transition candidate with torso visibility filter; not validated medical fall detection.','No candidate does not mean no fall; CPU or transport delays may cause insufficient observation.']}
    write_json(output/'receipt.json',receipt)
    if decoded==0:raise ValueError('No decoded frames; insufficient observation receipt retained')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('adb','ffmpeg','serial','weights','weights-sha256','roi','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--seconds',type=int,choices=range(1,181),default=180)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--deadline-monotonic',type=float,default=0,help=argparse.SUPPRESS)
    p.add_argument('--event-command-json',help='Optional single bounded callback argv; include {event_file}. No retries. Owner supplies adapter.')
    p.add_argument('--activation-file',help='Optional owner-controlled local file; no candidate/callback while absent. Creation resets upright baseline.')
    p.add_argument('--retain-frames',action='store_true',help='Save private cropped PNG evidence at <=5Hz, hard100MB cap. Never put output in public Git.')
    args=p.parse_args();parse_roi(args.roi)
    if args.event_command_json:event_command(args.event_command_json,Path('event.json'))
    if args.worker:worker(args);return
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=False)
    child=None
    with (output/'worker-log.txt').open('xb') as log:
        try:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker','--deadline-monotonic',str(time.monotonic()+args.seconds)],stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            child.wait(timeout=args.seconds)
            if child.returncode!=0:raise RuntimeError(f'Capture worker failed with exit {child.returncode}; inspect retained log')
        except subprocess.TimeoutExpired:
            # Kill only this newly-created process tree, including owned adb/ffmpeg.
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=10,check=True)
            else:os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=10)
            write_json(output/'supervisor.json',{'status':'insufficient_observation','reason':'hard_duration_limit','seconds':args.seconds,'partial_trace_retained':True,'dispatch_adapter_enabled':False})
            raise SystemExit(2)
        except Exception as error:
            write_json(output/'supervisor.json',{'status':'insufficient_observation','reason':str(error),'dispatch_adapter_enabled':False})
            raise
        finally:
            if child is not None and child.poll() is None:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=10,check=True)
                else:os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)

if __name__=='__main__':main()

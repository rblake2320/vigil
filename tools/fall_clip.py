"""Finite CPU video observation -> possible-fall review receipt. No outbound actions.

Single visible person only. Identity continuity is heuristic, not biometric identity.
Never infer no fall/safety from no candidate. Keep input and output private.
"""
import argparse,dataclasses,hashlib,json,math,os,sys,time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detectors.temporal_fall import PersonObservation,TemporalFallDetector

def analyze(args):
    import cv2,torch
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(2)
    path=Path(args.clip).resolve(strict=True);weights=Path(args.weights).resolve(strict=True)
    if path.stat().st_size>200_000_000: raise ValueError('Clip exceeds 200 MB bound')
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=args.weights_sha256: raise ValueError('Weights hash mismatch')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists(): raise ValueError('Output already contains receipt; preserve it')
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise ValueError('Cannot decode input video')
        fps=cap.get(cv2.CAP_PROP_FPS);total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not 1<=fps<=240: raise ValueError('Unsupported/missing frame rate')
        if total<=0 or total/fps>600: raise ValueError('Source must have positive duration at most ten minutes; analysis is separately capped')
        model=YOLO(str(weights),task='pose')
        stride=max(1,int(fps/5));limit=min(int(total),int(fps*args.seconds))
        detector=None;trace=[];events=[];previous=None;epoch=0;start=time.monotonic();index=0;last_ts=-1
        while index<limit:
            if time.monotonic()-start>120: raise TimeoutError('CPU analysis exceeded 120 seconds')
            ok,frame=cap.read()
            if not ok: raise ValueError(f'Truncated video at frame {index} of {limit}')
            ts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isfinite(ts) or ts<=last_ts: raise ValueError('Missing/nonmonotonic decoded media timestamp')
            last_ts=ts
            n=index;index+=1
            if n%stride: continue
            h,w=frame.shape[:2]
            if detector is None: detector=TemporalFallDetector()
            result=model.predict(frame,imgsz=320,device='cpu',verbose=False,conf=.4)[0]
            people=[];ambiguous=False;torso=[]
            boxes=result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
            if len(boxes)>1: ambiguous=True
            elif len(boxes)==1 and result.keypoints is not None:
                kc=result.keypoints.conf
                torso=[float(kc[0,i]) for i in [5,6,11,12]] if kc is not None else []
                visible=len(torso)==4 and all(c>=.35 for c in torso)
                if visible:
                    x1,y1,x2,y2=boxes[0];box=(max(0,x1/w),max(0,y1/h),min(1,x2/w),min(1,y2/h))
                    center=((box[0]+box[2])/2,(box[1]+box[3])/2)
                    if previous is None or math.dist(center,previous)>.35: epoch+=1
                    previous=center;people=[PersonObservation(str(epoch),box)]
            if not people: previous=None
            state=detector.update(ts,people,identity_ambiguous=ambiguous,frame_aspect_ratio=w/h)
            row={'frame':n,'media_seconds':ts,'status':state.status,'reasons':state.reasons,'visible_people':len(boxes),'boxes_pixels':boxes,'torso_confidences':torso,'tracks':[dataclasses.asdict(p) for p in people],'frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest()}
            trace.append(row)
            for candidate in state.candidates:
                event=dataclasses.asdict(candidate);event['frame']=n;events.append(event)
        usable=sum(r['status']!='insufficient_observation' for r in trace)
        receipt={'schema':1,'status':'possible_fall_review_required' if events else ('no_candidate_observed' if usable else 'insufficient_observation'),
                 'not_a_safety_verdict':True,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'weights_sha256':args.weights_sha256,'decoded_frames':index,'analyzed_frames':len(trace),
                 'clip_seconds':total/fps,'observed_seconds':last_ts,'partial_clip':index<total,'processing_seconds':time.monotonic()-start,
                 'events':events,'trace':trace,'dispatch_adapter_enabled':False,
                 'limits':['single visible person; tracker continuity heuristic','pose visibility filtered bounding-box transition heuristic unvalidated for real falls','occlusion/low light/missing torso yields insufficient observation','no emergency or caregiver dispatch','external process timeout required for hard deadline']}
        data=json.dumps(receipt,indent=2).encode();tmp=out/'receipt.tmp'
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        tmp.replace(out/'receipt.json')
        print(json.dumps({k:receipt[k] for k in ['status','analyzed_frames','processing_seconds','dispatch_adapter_enabled']}))
    finally: cap.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clip',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--weights-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=20,choices=range(1,61))
    analyze(p.parse_args())

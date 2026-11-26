"""Frozen causal TurnBench inference; audio/ID Parquet projection only.

Single-file implementation; requires the pinned public Silero ONNX asset and
Python runtime packages. No training states, annotation columns or study modules.
"""
import sys
sys.dont_write_bytecode=True
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import datetime
import hashlib
import io
import json
import time
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

"""CPU-only causal FIR decimation and stateful Silero 6.2.3 inference."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import firwin, lfilter

MODEL = Path.home() / "tb5/model/silero_vad.onnx"
HOP = 512
DT = 0.032
# Causal 63-tap low-pass; group delay is paid in the unshifted frame-end clock.
FIR = firwin(63, 7200, fs=48000).astype(np.float32)


class StreamingVAD:
    """Two independent channels, fixed batch size, with no future-dependent ops."""

    def __init__(self, sample_rate=48000):
        assert sample_rate in (16000, 48000)
        self.sample_rate = sample_rate
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 2, 128), np.float32)
        self.context = np.zeros((2, 64), np.float32)
        self.filter_state = np.zeros((2, len(FIR) - 1), np.float32)
        self.pending_native = np.zeros((2, 0), np.float32)
        self.pending_16k = np.zeros((2, 0), np.float32)
        self.native_samples = 0

    def push(self, native):
        """Accept arbitrary native chunks; emit only fully heard 512-sample frames."""
        assert native.ndim == 2 and native.shape[0] == 2
        self.native_samples += native.shape[1]
        if self.sample_rate == 48000:
            native = np.concatenate((self.pending_native, native), axis=1)
            consumed = native.shape[1] // 3 * 3
            self.pending_native = native[:, consumed:].copy()
            if consumed:
                filtered, self.filter_state = lfilter(FIR, [1.0], native[:, :consumed],
                                                     axis=1, zi=self.filter_state)
                audio = filtered[:, 2::3].astype(np.float32)
            else:
                audio = np.empty((2, 0), np.float32)
        else:
            audio = native.astype(np.float32)
        audio = np.concatenate((self.pending_16k, audio), axis=1)
        count = audio.shape[1] // HOP
        probs = np.empty((count, 2), np.float32)
        energy = np.empty_like(probs)
        for i in range(count):
            frame = audio[:, i * HOP:(i + 1) * HOP]
            inputs = np.concatenate((self.context, frame), axis=1)
            prob, self.state = self.session.run(None, {"input": inputs, "state": self.state,
                                                       "sr": np.array(16000, np.int64)})
            probs[i] = prob[:, 0]
            energy[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2, axis=1))
            self.context = frame[:, -64:].copy()
        self.pending_16k = audio[:, count * HOP:].copy()
        return probs, energy


def infer(audio, sample_rate=48000, native_chunk=48000 * 10):
    frontend = StreamingVAD(sample_rate)
    ps, es = [], []
    for start in range(0, audio.shape[1], native_chunk):
        p, e = frontend.push(audio[:, start:start + native_chunk])
        ps.append(p)
        es.append(e)
    return np.concatenate(ps), np.concatenate(es)


def energy_probability(rms):
    """Fixed absolute scale; 0.5 corresponds to RMS 0.01. No fitted statistics."""
    return rms / (rms + np.float32(0.01))

"""Causal completed-frame policy and an exact accelerated count-only scorer."""
import numpy as np
from numba import njit

DT = 0.032


@njit(nogil=True)
def units(prob, threshold, drop, hang):
    """Decode units without retroactive decisions: output state known at each frame."""
    n = len(prob)
    # onset, last heard voice, confirmed offset, last voice of that offset.
    out = np.zeros((n, 2, 4), np.float64)
    for s in range(2):
        on = False
        start = -1.0
        last = -1e6
        pending = 0
        for i in range(n):
            now = (i + 1) * DT
            p = prob[i, s]
            if not on:
                if p >= threshold:
                    pending += 1
                    if pending >= 2:
                        on = True
                        start = now - 2 * DT
                        last = now
                else:
                    pending = 0
            else:
                if p >= threshold - drop:
                    last = now
                if now - last + 1e-10 >= hang:
                    out[i, s, 2] = 1.0
                    out[i, s, 3] = last
                    on = False
                    pending = 0
            out[i, s, 0] = start if on else -1.0
            out[i, s, 1] = last
    return out


@njit(nogil=True)
def eot_events(u, duration, params, confirmation=True, partner=True, eos=True):
    """Params: threshold,drop,hang,wait,partner,deadline,confirm,cancel,floor,origin."""
    n = len(u)
    # Speaker, origin, last voice, floor override, initial emitted, cancelled.
    candidates = np.zeros((128, 6), np.float64)
    alive = np.zeros(128, np.bool_)
    high = 0
    events = np.empty((n * 2 + 4, 2), np.float64)
    count = 0
    for i in range(n):
        now = (i + 1) * DT
        for s in range(2):
            if u[i, s, 2] > 0:
                slot = -1
                for j in range(128):
                    if not alive[j]:
                        slot = j
                        break
                assert slot >= 0
                other = 1 - s
                origin = u[i, s, 3] if params[9] < 0.5 else now
                floor = (partner and params[8] < 100 and u[i, other, 0] >= 0
                         and origin - u[i, other, 0] >= params[8]
                         and now - u[i, other, 1] <= 0.2 + 1e-10)
                candidates[slot] = np.array([s, origin, u[i, s, 3], 1.0 if floor else 0.0, 0., 0.])
                alive[slot] = True
                high = max(high, slot + 1)
        for j in range(high):
            if not alive[j]:
                continue
            s = int(candidates[j, 0])
            other = 1 - s
            origin = candidates[j, 1]
            since = now - origin
            own_start = u[i, s, 0]
            other_start = u[i, other, 0]
            other_live = other_start >= 0 and now - u[i, other, 1] <= 0.2 + 1e-10
            if own_start > candidates[j, 2] - 1e-10:
                quick = own_start - candidates[j, 2] <= 0.2 + 1e-10
                sustained = now - own_start + 1e-10 >= params[7] and now - u[i, s, 1] <= 0.2 + 1e-10
                if quick or sustained:
                    candidates[j, 5] = 1
            override = candidates[j, 3] > 0 and other_live
            cancelled = candidates[j, 5] > 0
            triggered = (since + 1e-10 >= params[5] or
                         (partner and other_live and now - other_start + 1e-10 >= params[4]) or override)
            if (candidates[j, 4] == 0 and since + 1e-10 >= params[3]
                    and triggered and (not cancelled or override)):
                events[count, 0] = s
                events[count, 1] = now
                count += 1
                candidates[j, 4] = 1
            if cancelled:
                if not override or candidates[j, 4] > 0:
                    alive[j] = False
                continue
            if confirmation and since + 1e-10 >= params[6]:
                events[count, 0] = s
                events[count, 1] = now
                count += 1
                alive[j] = False
            elif not confirmation and candidates[j, 4] > 0:
                alive[j] = False
        # Candidates never survive their confirmation deadline plus one frame.
    if eos:
        for j in range(128):
            if alive[j] and candidates[j, 5] == 0:
                events[count, 0] = candidates[j, 0]
                events[count, 1] = duration
                count += 1
    return events[:count]


@njit(nogil=True)
def int_events(u, rms, params):
    """threshold,drop,hang,fallback,prior-floor,memory,early-min,early-rms,unconditional."""
    n = len(u)
    events = np.empty((n * 2 + 4, 2), np.float64)
    count = 0
    emitted = np.zeros(2, np.bool_)
    allowed = np.zeros(2, np.bool_)
    previous = np.full(2, -1.0)
    floor_start = np.full(2, -1e6)
    for i in range(n):
        now = (i + 1) * DT
        for s in range(2):
            if u[i, s, 0] >= 0:
                floor_start[s] = u[i, s, 0]
        for s in range(2):
            onset = u[i, s, 0]
            other = 1 - s
            if onset >= 0 and onset != previous[s]:
                emitted[s] = False
                allowed[s] = (params[8] > .5 or
                              (now - u[i, other, 1] <= params[5] + 1e-10
                               and floor_start[other] > -1e5
                               and min(now, u[i, other, 1]) - floor_start[other] >= params[4]))
            previous[s] = onset
            if onset < 0 or emitted[s] or not allowed[s] or now - u[i, s, 1] > 0.2 + 1e-10:
                continue
            age = now - onset
            trigger = age + 1e-10 >= params[3]
            if params[6] < 100 and age + 1e-10 >= params[6] and rms[i, s] >= params[7]:
                trigger = True
            if trigger:
                events[count] = np.array([s, now])
                count += 1
                emitted[s] = True
    return events[:count]


def event_lists(events):
    return [np.unique(events[events[:, 0] == s, 1]) for s in range(2)]



"""Registered three-parameter causal energy-unit and commit extension."""
import numpy as np
from numba import njit


@njit(nogil=True)
def split_units(prob,threshold,drop,hang,dip):
    if dip<=0:return units(prob,threshold,drop,hang)
    # Derive base partner state only from completed frames.
    partner=units(prob,threshold,drop,hang)
    n=len(prob);out=np.zeros((n,2,4),np.float64)
    for s in range(2):
        on=False;start=-1.;last=-1e6;pending=0;low_start=-1.
        for i in range(n):
            now=(i+1)*.032;p=prob[i,s]
            if not on:
                if p>=threshold:
                    pending+=1
                    if pending>=2:on=True;start=now-.064;last=now;low_start=-1.
                else:pending=0
            else:
                if p>=threshold-drop:last=now;low_start=-1.
                elif low_start<0:low_start=now-.032
                other_live=partner[i,1-s,0]>=0 and now-partner[i,1-s,1]<=.2+1e-10
                split=low_start>=0 and now-low_start+1e-10>=dip and other_live
                if now-last+1e-10>=hang or split:
                    out[i,s,2]=1.;out[i,s,3]=last;on=False;pending=0;low_start=-1.
            out[i,s,0]=start if on else -1.;out[i,s,1]=last
    return out

@njit(nogil=True)
def extended_int(u,rms,params,extra):
    if extra[0]<0:return int_events(u,rms,params)
    n=len(u);events=np.empty((n*2+4,2),np.float64);count=0
    emitted=np.zeros(2,np.bool_);allowed=np.zeros(2,np.bool_)
    overlap=np.zeros(2,np.bool_);previous=np.full(2,-1.);floor_start=np.full(2,-1e6)
    for i in range(n):
        now=(i+1)*.032
        for s in range(2):
            if u[i,s,0]>=0:floor_start[s]=u[i,s,0]
        for s in range(2):
            onset=u[i,s,0];other=1-s
            if onset>=0 and onset!=previous[s]:
                emitted[s]=False
                allowed[s]=(params[8]>.5 or (now-u[i,other,1]<=params[5]+1e-10 and floor_start[other]>-1e5 and min(now,u[i,other,1])-floor_start[other]>=params[4]))
                overlap[s]=u[i,other,0]>=0 and now-u[i,other,1]<=.2+1e-10
            previous[s]=onset
            if onset<0 or emitted[s] or not allowed[s] or now-u[i,s,1]>.2+1e-10:continue
            age=now-onset;trigger=age+1e-10>=params[3]
            if params[6]<100 and age+1e-10>=params[6] and rms[i,s]>=params[7]:trigger=True
            if overlap[s] and age+1e-10>=extra[0] and now-u[i,other,1]+1e-10>=extra[1]:trigger=True
            if trigger:events[count]=np.array([s,now]);count+=1;emitted[s]=True
    return events[:count]



def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def predict_frames(probability,rms,duration,params,eos=True):
    eot=event_lists(eot_events(units(probability,*params['eot'][:3]),duration,np.asarray(params['eot']),True,True,eos))
    base=params['interruption'];extra=params['extension']
    if base is None:interrupt=[np.empty(0),np.empty(0)]
    else:
        u=split_units(energy_probability(rms),*base[:3],extra[2])
        interrupt=event_lists(extended_int(u,rms,np.asarray(base),np.asarray(extra)))
    return eot,interrupt

def audio_rows(folder,expected_shards):
    # Projection is a literal allowlist. Annotation and transcript columns are
    # never deserialized, even if a future Parquet schema includes them.
    columns=['conversation_id','speaker_1_audio','speaker_2_audio']
    shards=sorted(Path(folder).glob('*.parquet'))
    if len(shards)!=expected_shards:raise ValueError('Unexpected shard count')
    for shard in shards:
        pf=pq.ParquetFile(shard)
        if not set(columns)<=set(pf.schema_arrow.names):raise ValueError('Missing audio/ID fields')
        for batch in pf.iter_batches(batch_size=1,columns=columns,use_threads=False):
            row=batch.to_pylist()[0]
            if set(row)!=set(columns):raise ValueError('Projection contract violated')
            yield row

def decode_audio(row):
    waves=[];rates=[]
    for s in (1,2):
        blob=row[f'speaker_{s}_audio']['bytes']
        with sf.SoundFile(io.BytesIO(blob)) as audio:
            if audio.channels!=1 or audio.samplerate not in (16000,48000):raise ValueError('Unsupported native audio')
            rates.append(audio.samplerate);waves.append(audio.read(dtype='float32'))
    if rates[0]!=rates[1] or len(waves[0])!=len(waves[1]):raise ValueError('Channel mismatch')
    return np.stack(waves),rates[0]

def main():
    global MODEL
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parameters',type=Path,default=Path(__file__).parent/'parameters.json')
    parser.add_argument('--freeze',type=Path,default=Path(__file__).parent/'FROZEN.json')
    parser.add_argument('--model',type=Path,default=MODEL)
    args=parser.parse_args();MODEL=args.model
    root=Path(__file__).parent
    params=json.loads(args.parameters.read_text());freeze=json.loads(args.freeze.read_text())
    code_hash=digest(__file__);param_hash=digest(args.parameters)
    if code_hash!=freeze['code_sha256'] or param_hash!=freeze['parameters_sha256']:raise RuntimeError('Frozen hash mismatch')
    if digest(MODEL)!=params['silero_model_sha256']:raise RuntimeError('Silero asset mismatch')
    if args.output.exists():raise RuntimeError('Prediction output already exists; refusing overwrite')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.split=='test':
        # Exclusive creation deliberately makes any retry fail before opening data.
        marker=root/'TEST-STARTED.json'
        with marker.open('x') as f:json.dump(dict(utc=now,code_sha256=code_hash,parameters_sha256=param_hash,
            projected_columns=['conversation_id','speaker_1_audio','speaker_2_audio'],test_metrics_forbidden=True),f,indent=2)
        with (root/'PROGRESS.md').open('a') as f:f.write(f'- {now} Step 5: One-shot test started; literal Parquet audio/ID projection enforced; labels are never requested or loaded; frozen hashes verified.\n')
    started=time.monotonic();rows=[];durations={};samples=0
    for row in audio_rows(args.data,3 if args.split=='dev' else 8):
        cid=str(row['conversation_id'])
        if cid in durations:raise ValueError('Duplicate ID')
        wave,sr=decode_audio(row);duration=wave.shape[1]/sr;samples+=wave.shape[1]
        probability,rms=infer(wave,sr)
        eot,interrupt=predict_frames(probability,rms,duration,params)
        rows.append(dict(conversation_id=cid,**{f'speaker_{s+1}':dict(eot=eot[s].tolist(),interruption=interrupt[s].tolist()) for s in range(2)}))
        durations[cid]=duration
        print(json.dumps(dict(split=args.split,processed=len(rows),elapsed_s=round(time.monotonic()-started,3))),flush=True)
    rows.sort(key=lambda row:int(row['conversation_id']))
    payload=dict(schema_version=1,predictions=rows)
    with args.output.open('x') as f:f.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    receipt=dict(started_utc=now,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),split=args.split,
        conversations=len(rows),audio_hours=sum(durations.values())/3600,seconds=time.monotonic()-started,
        code_sha256=code_hash,parameters_sha256=param_hash,predictions_sha256=digest(args.output),
        projected_columns=['conversation_id','speaker_1_audio','speaker_2_audio'],labels_loaded=False,metrics_computed=False)
    (root/f'inference-{args.split}-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

"""Single Part B measurement fixture; never a development or final scientific run."""
import os
import time
IMPORT_START=time.monotonic();IMPORT_CPU=time.process_time()
from pathlib import Path
import copy
import json
import ctypes
import numpy as np
import torch
import numerics
from core import HERE, PILOT, Budget, Output, Seeds, read, write, replace_record, identity, source_hashes
from numerics import (Oracle, Splits, runtime, legacy, allocation, generate, keyed_truth,
                      task_data, inverse_batched, Observations, fresh)
from training import Trajectory
from campaign import owned_model, save_model
IMPORT_SECONDS=time.monotonic()-IMPORT_START
IMPORT_CPU_SECONDS=time.process_time()-IMPORT_CPU


def memory():
    try:
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
        kernel=ctypes.WinDLL('kernel32');psapi=ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return dict(working_set_bytes=counters.WorkingSetSize,peak_working_set_bytes=counters.PeakWorkingSetSize,
                    private_bytes=counters.PrivateUsage)
    except Exception as exc:return dict(unavailable=str(exc))


class Measurement:
    def __init__(self):
        if os.environ.get('TPFA_CATEGORY')!='part_b_measurement':
            raise PermissionError('requires the one Part B measurement supervisor')
        self.budget=Budget.verification()
        self.initial_counts=dict(self.budget.counts)
        self.out=Output(Path(os.environ['TPFA_ATTEMPT'])/'measurement','verification')
        self.start=time.monotonic();self.cpu=time.process_time();self.phase='setup'
        self.timings=[];self.solver_wall=0.;self.solver_cpu=0.;self.generation_wall=0.
        self.out.log(dict(type='imports',wall_seconds=IMPORT_SECONDS,cpu_seconds=IMPORT_CPU_SECONDS))

    def timed(self,name,fn,**metadata):
        self.budget.check();self.phase=name
        replace_record(self.out.path/'CURRENT_STAGE.json',dict(stage=name,counts=self.budget.counts))
        print('STAGE BEGIN '+name,flush=True)
        wall,cpu=time.monotonic(),time.process_time();before=dict(self.budget.counts)
        solver_before=self.solver_wall;generation_before=self.generation_wall
        try:result=fn()
        except BaseException as exc:
            self.out.log(dict(type='stage_failed',stage=name,error=type(exc).__name__+': '+str(exc),
                              wall_seconds=time.monotonic()-wall,counts=self.budget.counts))
            raise
        row=dict(stage=name,wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
                 solver_work_seconds=self.solver_wall-solver_before,
                 control_direction_generation_seconds=self.generation_wall-generation_before,
                 counts={k:self.budget.counts[k]-before[k] for k in before},memory=memory(),**metadata)
        self.timings.append(row);self.out.log(dict(type='timing',**row))
        write(self.out.path/(f'TIMING_{len(self.timings):03d}.json'),row)
        print('STAGE END '+name+' '+json.dumps(row),flush=True)
        return result

    def save_arrays(self,name,data):
        path=self.out.path/('PRIVATE_'+name+'.npz')
        with path.open('xb') as stream:
            np.savez(stream,**{k:getattr(data,k) for k in ('x64','y64','v64','dy64') if getattr(data,k) is not None})
        return identity(path)

    def execute(self):
        oracle=None
        try:
            meta=self.timed('runtime_and_legacy_import',lambda: (runtime(),legacy()))
            runtime_info,(spatial,old)=meta
            write(self.out.path/'RUNTIME.json',runtime_info)
            write(self.out.path/'SOURCE_CONFIG.json',dict(sources=source_hashes(),config=identity(HERE/'draft.json')))
            seeds=Seeds(self.out.path/'PRIVATE_FIXTURE_SEEDS.json','cost_fixture_only',fixture=2026093001)
            splits=Splits()
            def solve(row):
                wall,cpu=time.monotonic(),time.process_time()
                try:return spatial.solve(row)
                finally:self.solver_wall+=time.monotonic()-wall;self.solver_cpu+=time.process_time()-cpu
            oracle=Oracle(solve,self.budget,real=True,log=self.out.log)
            originals={name:getattr(numerics,name) for name in ('controls','directions')}
            for name,original in originals.items():
                def wrapped(*args,_original=original,**kwargs):
                    start=time.monotonic()
                    try:return _original(*args,**kwargs)
                    finally:self.generation_wall+=time.monotonic()-start
                setattr(numerics,name,wrapped)
            # Only an owned, retained, identity-matched checkpoint is unpickled.
            checkpoint=PILOT/'run_20260924/clean_owner_0.pt'
            stage0=read(HERE/'evidence/stage0_01/COMPLETED.json')
            pin=next(r['sha256'] for r in stage0['evidence'] if r['path']==str(checkpoint.resolve()))
            reference=self.timed('owned_checkpoint_load',lambda:owned_model(checkpoint,pin))
            write(self.out.path/'OWNED_CHECKPOINT_IDENTITY.json',identity(checkpoint))
            forward_model=self.timed('fresh_model_initialization',lambda:fresh(seeds.get('forward_initialization')))
            shared,rb=self.timed('attacker_allocation',lambda:allocation(oracle,seeds.get('attack_controls'),
                                        seeds.get('attack_directions'),splits))
            assert shared.x64[:85].tobytes()==rb.x64.tobytes()
            assert shared.information_calls==256 and rb.information_calls==255
            data_facts=self.timed('attacker_data_serialization',lambda:[self.save_arrays('attack',shared),self.save_arrays('Rb',rb)])
            forward=self.timed('forward_trainer_setup',lambda:Trajectory(forward_model,shared,
                                seeds.get('forward_shuffle'),self.budget,real=True))
            derivative=self.timed('derivative_trainer_setup',lambda:Trajectory(fresh(seeds.get('derivative_initialization')),
                                rb,seeds.get('derivative_shuffle'),self.budget,recipe='derivative',batch=16,real=True))
            subset=Observations(rb.x64[:32],rb.y64[:32],rb.v64[:32],rb.dy64[:32],96)
            key=old.b3_key(seeds.get('fixture_key',offsets=(0,10000)))
            marked=self.timed('marked_trainer_setup_32',lambda:Trajectory(copy.deepcopy(reference),subset,
                              seeds.get('marked_shuffle'),self.budget,recipe='mark',batch=16,lr=.0003,
                              reference=reference,key=key,real=True))
            for name,trainer,total in [('forward',forward,8),('derivative',derivative,12),('marked',marked,4)]:
                for index in range(total):
                    expected_batch=5 if name=='derivative' and index%6==5 else 32 if name=='forward' else 16
                    self.timed(name+'_update',lambda t=trainer:t.extend(t.updates+1),
                               warmup=index==0,batch_size=expected_batch,update=index+1)
                    assert trainer.batch_sizes[-1]==expected_batch
            self.timed('continuous_checkpoint_serialization',lambda:forward.save(self.out.path/'PRIVATE_continuous.pt'))
            self.timed('model_checkpoint_serialization',lambda:save_model(self.out.path/'PRIVATE_model.pt',marked.model))
            data=self.timed('evaluation_truth_128x2',lambda:generate(oracle,seeds.get('evaluation_controls'),
                                   seeds.get('evaluation_directions'),128,2,'cost_evaluation',splits))
            truth=self.timed('fixture_key_truth_128',lambda:keyed_truth(oracle,data.x64,key,'cost_key',splits))
            self.timed('evaluation_data_serialization',lambda:self.save_arrays('evaluation',data))
            tensors=data.tensors();subtimings=[]
            originals_eval={name:getattr(old,name) for name in ('forward_all','jvp_all')}
            for name,original in originals_eval.items():
                def timed_eval(*args,_original=original,_name=name,**kwargs):
                    start=time.monotonic();cpu=time.process_time()
                    result=_original(*args,**kwargs)
                    subtimings.append(dict(operation=_name,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu))
                    return result
                setattr(old,name,timed_eval)
            metrics,details=self.timed('forward_random_key_evaluation',lambda:old.evaluate(marked.model,tensors,key,truth))
            evaluation_components=list(subtimings);subtimings.clear()
            self.timed('evaluation_json_serialization',lambda:write(self.out.path/'PRIVATE_EVALUATION.json',
                 {k:v.tolist() if torch.is_tensor(v) else v for k,v in details.items()}))
            wrong_base=seeds.get('wrong_fixture_keys',offsets=tuple(range(16))+tuple(10000+i for i in range(16)))
            wrong=[old.b3_key(wrong_base+i) for i in range(16)]
            scores=self.timed('true_plus_16_wrong_detector_block',lambda:[old.b3_score(marked.model,reference,
                              tensors[0],k,marked.score_scale) for k in [key]+wrong])
            detector_components=list(subtimings)
            for name,original in originals_eval.items():setattr(old,name,original)
            tasks=self.timed('inverse_targets_and_initial_start_truth',lambda:task_data(oracle,seeds,64,'cost_tasks',splits))
            self.timed('task_data_serialization',lambda:write(self.out.path/'PRIVATE_TASKS.json',
                       {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in tasks.items()}))
            original_inverse=numerics.inverse;batch_index=0
            def timed_inverse(*args,**kwargs):
                nonlocal batch_index
                index=batch_index;batch_index+=1
                return self.timed('inverse_batch_'+str(index),lambda:original_inverse(*args,**kwargs),
                                  targets=16,expected_input_updates=120)
            numerics.inverse=timed_inverse
            try:
                task_result=self.timed('inverse_64_four_batches',lambda:inverse_batched(marked.model,tasks['starts'],
                       tasks['targets'],oracle,self.budget,'cost_inverse',progress=self.out.log))
            finally:numerics.inverse=original_inverse
            self.timed('inverse_json_serialization',lambda:write(self.out.path/'PRIVATE_INVERSE.json',task_result))
            for name,original in originals.items():setattr(numerics,name,original)
            delta={k:self.budget.counts[k]-self.initial_counts[k] for k in self.initial_counts}
            assert delta['model_updates']==24 and delta['inverse_updates']==480
            assert oracle.requests==1514 and oracle.attempted<=1514
            result=dict(passed=True,fixture=True,reusable_for_science=False,
              wall_seconds=time.monotonic()-self.start,cpu_seconds=time.process_time()-self.cpu,
              import_wall_seconds=IMPORT_SECONDS,import_cpu_seconds=IMPORT_CPU_SECONDS,runtime=runtime_info,
              timings=self.timings,evaluation_components=evaluation_components,detector_components=detector_components,
              counts=self.budget.counts,initial_counts=self.initial_counts,attempt_counts=delta,
              oracle=oracle.record(),solver_work_seconds=self.solver_wall,
              solver_cpu_seconds=self.solver_cpu,memory=memory(),metrics=metrics,
              detector_score_count=len(scores),detector_jvp_passes=len(detector_components),
              inverse_mean=task_result['mean'],initial_task_mean=float(np.mean(tasks['initial_errors'])),
              attacker_information=dict(Ra=256,Rb=255),nominal_solver_requests=1514,
              private_file_sizes={p.name:p.stat().st_size for p in self.out.path.glob('PRIVATE_*')},
              representative_subset='owner derivative timing uses R-b fixture data; marked timing uses first 32 derivative rows; neither is an owner run')
            self.out.finish(result)
        except BaseException as exc:
            self.out.finish(dict(passed=False,fixture=True,stage=self.phase,type=type(exc).__name__,message=str(exc),
                       counts=self.budget.counts,oracle=None if oracle is None else oracle.record(),
                       timings=self.timings,solver_work_seconds=self.solver_wall,memory=memory()),failed=True)
            raise


if __name__=='__main__':Measurement().execute()

"""Q6's isolated environment route, with bounded owned setup and interpreter wiring.

Only called in a later authorized Colab session. No package operations on import.
"""
from pathlib import Path
import os
import sys
import time
import subprocess
from .records import read,write,sha,verify

def setup_commands(kernel,environment,bundle):
    python=str(Path(environment)/'bin/python')
    return [([kernel,'-m','venv','--without-pip',str(environment)],60),
            ([kernel,'-m','pip','--python',python,'install','--disable-pip-version-check',
              '-r',str(Path(bundle)/'requirements-colab.txt')],1200),
            ([kernel,'-m','pip','--python',python,'check'],60)]

def launch_command(python,bundle,root,manifest):
    return [str(python),'-B',str(Path(bundle)/'code/mainstudy_development.py'),'colab',
            '--bundle',str(bundle),'--out',str(root),'--manifest-sha256',manifest,'--execute-development']

def owned(command,work,deadline,folder,supervisor,cwd=None,acceptable=(0,),env=None):
    """Reuse Q5 group/subreaper cleanup, one deadline including cleanup."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);start=time.time()
    remaining=min(work,deadline-time.time()-30)
    if remaining<=0:raise TimeoutError('setup/launch cleanup envelope unavailable')
    from .bounded_capture import capture
    began=time.monotonic()
    r=capture(command,cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1') if env is None else env,
        work_deadline=began+remaining,cleanup_deadline=began+min(work+30,max(0,deadline-time.time())),
        folder=folder,supervisor=supervisor,on_identity=lambda identity:write(folder/'IDENTITY.json',identity))
    r.update(command=command,started=start,seconds=time.time()-start)
    write(folder/'RESULT.json',r)
    if (r['errors'] or r['timed_out'] or r['stream_overflow'] or not r['stdout_closed'] or
        r['returncode'] not in acceptable or not r.get('closure',{}).get('resolved') or
        any(not v['capture_complete'] for v in r['streams'].values())):
        raise RuntimeError('owned command failed; preserve '+str(folder/'RESULT.json'))

    return r

def setup(bundle,root,environment):
    from .cli import module
    import ctypes
    bundle=Path(bundle);root=Path(root);environment=Path(environment)
    if os.name!='posix':raise ValueError('Colab-only environment setup')
    state=read(root/'STATUS.json');session=state['session']
    if not session.get('prebooked') or session.get('closed') or session.get('launch_started'):raise ValueError('unused charged window required')
    verify(bundle,read(bundle/'MANIFEST.json')['files'])
    prior=read(bundle/'Q6_ENVIRONMENT.json')
    if sys.version!=prior['python']:raise ValueError('base Python differs from Q6: '+sys.version+' != '+prior['python'])
    if environment.exists():raise FileExistsError('setup already attempted; preserve environment and receipts, no automatic retry')
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('setup subreaper unavailable')
    supervisor=module(bundle/'code/q5_colab_supervisor.py','msdev_setup_owned')
    folder=root/f'environment_{time.time_ns()}';folder.mkdir()
    deadline=min(session['end']-30,state['deadline']-30,time.time()+1380)
    receipts=[]
    for i,(command,seconds) in enumerate(setup_commands(sys.executable,environment,bundle)):
        receipts.append(owned(command,seconds,deadline,folder/str(i),supervisor))
    python=environment/'bin/python'
    # Exact runtime/parity applicability is checked INSIDE the recreated interpreter.
    command=[str(python),'-B',str(bundle/'code/mainstudy_development.py'),'check-runtime','--bundle',str(bundle),'--out',str(folder)]
    receipts.append(owned(command,60,deadline,folder/'runtime',supervisor))
    write(folder/'ENVIRONMENT_READY.json',dict(python=str(python),requirements_sha256=sha(bundle/'requirements-colab.txt'),
        manifest_sha256=sha(bundle/'MANIFEST.json'),seconds=sum(r['seconds'] for r in receipts),session_start=session['start']))
    state=read(root/'STATUS.json');state['session']['setup_seconds']=sum(r['seconds'] for r in receipts)
    state['session']['environment_receipt']=str((folder/'ENVIRONMENT_READY.json').relative_to(root));write(root/'STATUS.json',state,replace=True)
    return python

def launch(python,bundle,root,manifest):
    from .cli import module
    s=read(Path(root)/'STATUS.json')
    owner=module(Path(bundle)/'code/q5_colab_supervisor.py','msdev_notebook_owned')
    # Outer supervisor allowance ends before the prebooked disconnect deadline;
    # inner worker retains its own 30s cleanup reserve.
    receipt=owned(launch_command(python,bundle,root,manifest),max(0,s['session']['end']-time.time()-30),
                 s['session']['end'],Path(root)/f'notebook_supervisor_{time.time_ns()}',owner,acceptable=(0,5))
    if receipt['returncode']==0:
        from .campaign import finish_manifest
        def check():
            if time.time()>=s['session']['end']:raise TimeoutError('closed-stream final manifest deadline')
        finish_manifest(root,check)
    return receipt

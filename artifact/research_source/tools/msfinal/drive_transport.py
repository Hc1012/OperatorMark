"""CPU-only Drive REST transport. No mount, numerical dependency or credential store.

Caller supplies an OAuth access-token callback and an independently bound folder.
Every operation is deadline checked; a separate owned supervisor MUST enforce the
same absolute deadline (including DNS/authentication and process shutdown).
No request retries, overwrite, permission changes or experiment launch.
Explicit redundant-copy removal is called only by the receipt-gated custody path.
"""
from contextlib import contextmanager
import hashlib
import http.client
import json
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlencode,urlsplit,quote
from msdev.records import write,sha

API='https://www.googleapis.com/drive/v3/'
UPLOAD='https://www.googleapis.com/upload/drive/v3/files'
CHUNK=8*1024*1024  # multiple of Drive's 256 KiB resumable-upload granularity

def digest_bytes(data):return hashlib.sha256(data).hexdigest()

class RefreshingAuthorizedUserToken:
    """Explicit existing authorized-user credential, refreshed before REST calls.

    No login, scope changes, file writes, HTTP retry or credential logging. The
    enclosing owned supervisor still bounds DNS and the complete operation.
    """
    def __init__(self, path, *, clock=time.time, connection=http.client.HTTPSConnection):
        self.clock=clock;self.connection=connection;self.lock=threading.Lock()
        self.token=None;self.expires=0.;self.refreshes=0
        try:
            p=Path(path)
            if not p.is_absolute() or p.is_symlink() or p.stat().st_size>65536:
                raise ValueError()
            raw=p.read_bytes()
            if len(raw)>65536:raise ValueError()
            c=json.loads(raw)
            if c.get('type')!='authorized_user' or c.get('token_uri','https://oauth2.googleapis.com/token')!='https://oauth2.googleapis.com/token':
                raise ValueError()
            self.fields={k:c[k] for k in ('client_id','client_secret','refresh_token')}
            if any(not isinstance(v,str) or not v or len(v)>16384 or '\r' in v or '\n' in v for v in self.fields.values()):
                raise ValueError()
        except Exception:
            raise ValueError('existing refreshable authorized-user credential unavailable') from None

    def __call__(self, deadline):
        if not isinstance(deadline,(int,float)) or not self.clock()<deadline:
            raise TimeoutError('credential deadline expired')
        if not self.lock.acquire(timeout=min(30,deadline-self.clock())):
            raise TimeoutError('credential refresh lock deadline')
        try:
            now=self.clock()
            if now>=deadline:raise TimeoutError('credential deadline expired')
            if self.token is not None and now+60<self.expires:return self.token
            end=min(deadline,now+30);conn=None;response=None;timer=None
            def interrupt():
                sock=getattr(conn,'sock',None)
                if sock is None and response is not None:
                    sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
                if sock is not None:
                    try:sock.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
                    conn.close()
            try:
                conn=self.connection('oauth2.googleapis.com',timeout=max(.001,end-self.clock()))
                timer=threading.Timer(max(.001,end-self.clock()),interrupt);timer.daemon=True;timer.start()
                body=urlencode(dict(self.fields,grant_type='refresh_token')).encode('ascii')
                conn.request('POST','/token',body=body,headers={'Content-Type':'application/x-www-form-urlencoded'})
                response=conn.getresponse()
                if response.status!=200:raise ValueError('refresh refused')
                raw=response.read(65537)
                if len(raw)>65536:raise ValueError('oversized refresh response')
                result=json.loads(raw);token=result.get('access_token');ttl=result.get('expires_in')
                if (not isinstance(token,str) or not token or len(token)>16384 or '\r' in token or '\n' in token
                        or result.get('token_type','Bearer').lower()!='bearer'
                        or type(ttl) not in (int,float) or not 60<ttl<=86400):
                    raise ValueError('invalid refresh response')
                if self.clock()>=end:raise TimeoutError('credential refresh deadline')
                # Conservative expiry starts before the request, never after it.
                if now+ttl<=self.clock()+60:raise ValueError('insufficient token lifetime')
                self.token=token;self.expires=now+ttl;self.refreshes+=1
                return token
            except Exception:
                self.token=None;self.expires=0.
                raise RuntimeError('credential refresh failed; no retry') from None
            finally:
                if timer is not None:timer.cancel();timer.join()
                if response is not None:response.close()
                if conn is not None:conn.close()
        finally:self.lock.release()

class RefreshingMetadataToken:
    """Colab's existing ADC metadata service; no google-auth in worker venv.

    Explicit endpoint policy, one GET per refresh, no login/scopes/redirects or
    token persistence. Each publisher/closure process constructs its own provider.
    Socket watchdog plus the existing owned supervisor bounds DNS/termination.
    """
    PATH='/computeMetadata/v1/instance/service-accounts/default/token'
    def __init__(self, host, *, clock=time.time, connection=http.client.HTTPConnection):
        if host not in ('metadata.google.internal','169.254.169.254'):
            raise ValueError('unreviewed metadata endpoint')
        self.host=host;self.clock=clock;self.connection=connection
        self.lock=threading.Lock();self.token=None;self.expires=0.;self.refreshes=0
        self.failed=False

    def __call__(self, deadline):
        if type(deadline) not in (int,float) or not self.clock()<deadline<float('inf'):
            raise TimeoutError('credential deadline expired')
        if not self.lock.acquire(timeout=min(30,deadline-self.clock())):
            raise TimeoutError('credential refresh lock deadline')
        try:
            if self.failed:raise RuntimeError('credential refresh previously failed; no retry')
            now=self.clock()
            if now>=deadline:raise TimeoutError('credential deadline expired')
            if self.token is not None and now+60<self.expires:return self.token
            end=min(deadline,now+30);conn=None;response=None;timer=None
            def interrupt():
                sock=getattr(conn,'sock',None)
                if sock is None and response is not None:
                    sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
                if sock is not None:
                    try:sock.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
                    conn.close()
            try:
                conn=self.connection(self.host,timeout=max(.001,end-self.clock()))
                timer=threading.Timer(max(.001,end-self.clock()),interrupt);timer.daemon=True;timer.start()
                conn.request('GET',self.PATH,headers={'Metadata-Flavor':'Google'})
                response=conn.getresponse()
                if response.status!=200 or response.getheader('Metadata-Flavor')!='Google':
                    raise ValueError('metadata refresh refused')
                raw=response.read(65537)
                if len(raw)>65536:raise ValueError('oversized refresh response')
                result=json.loads(raw);token=result.get('access_token');ttl=result.get('expires_in')
                if (not isinstance(token,str) or not token or len(token)>16384 or '\r' in token or '\n' in token
                        or result.get('token_type','').lower()!='bearer'
                        or type(ttl) not in (int,float) or not 60<ttl<=86400):
                    raise ValueError('invalid refresh response')
                if self.clock()>=end or now+ttl<=self.clock()+60:
                    raise TimeoutError('credential refresh deadline/lifetime')
                self.token=token;self.expires=now+ttl;self.refreshes+=1
                return token
            except Exception:
                self.failed=True;self.token=None;self.expires=0.
                raise RuntimeError('metadata credential refresh failed; no retry') from None
            finally:
                if timer is not None:timer.cancel();timer.join()
                if response is not None:response.close()
                if conn is not None:conn.close()
        finally:self.lock.release()

class GoogleDriveTransfer:
    """Structural implementation of export.RemoteTransfer, separately source pinned.

    Reader/writer IDs identify distinct HTTP roles, not different human principals.
    Fresh HTTPS GET of a pinned revision is independent of local/FUSE caches; it
    is not an independent scientific attestation or proof against account deletion.
    """
    def __init__(self,folder_id,token,log_root,*,byte_limit,metadata_byte_limit,clock=time.time,chunk_bytes=CHUNK):
        self.folder_id=self._id(folder_id);self.token=token;self.clock=clock
        self.log=Path(log_root);self.log.mkdir(parents=True,exist_ok=False)
        if type(chunk_bytes) is not int or chunk_bytes<262144 or chunk_bytes%262144:raise ValueError('Drive chunk granularity')
        if type(byte_limit) is not int or byte_limit<=0:raise ValueError('explicit additional upload-byte ceiling required')
        if type(metadata_byte_limit) is not int or metadata_byte_limit<=0:raise ValueError('explicit metadata memory bound required')
        self.chunk_bytes=chunk_bytes;self.sequence=0;self.byte_limit=byte_limit;self.charged=0
        self.metadata_byte_limit=metadata_byte_limit
        self.binding=dict(readback_kind='authenticated_remote_object_stream',cache_bypass=True,
            destination_id=self.folder_id,writer_id='drive-v3-resumable-upload',reader_id='drive-v3-pinned-revision-GET',
            source_sha256=sha(__file__),additional_byte_ceiling=byte_limit,metadata_byte_limit=metadata_byte_limit)
    @staticmethod
    def _id(value):
        if not isinstance(value,str) or not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
            raise ValueError('invalid Drive object/revision ID')
        return value
    def remaining(self,deadline,check):
        check();remaining=deadline-self.clock()
        if remaining<=0:raise TimeoutError('original Drive transport deadline expired')
        return remaining
    def event(self,kind,**fields):
        self.sequence+=1
        write(self.log/f'{self.sequence:06d}_{kind}.json',dict(time=self.clock(),kind=kind,**fields),durable=True)
    @contextmanager
    def request(self,method,url,*,deadline,check,body=None,headers=None,expected=(200,)):
        """Fresh connection; exact host allowlist; no redirect/retry or URL logging.

        Absolute watchdog closes a connected socket even under slow trickle reads.
        DNS/auth callback startup additionally requires the outer owned supervisor.
        """
        remaining=self.remaining(deadline,check);u=urlsplit(url)
        if u.scheme!='https' or u.hostname!='www.googleapis.com' or u.port not in (None,443) or u.username or u.password:
            raise ValueError('Drive endpoint/redirect substitution')
        token=self.token(deadline)
        if not isinstance(token,str) or not token or '\n' in token or '\r' in token:raise ValueError('OAuth token unavailable')
        connection=http.client.HTTPSConnection(u.hostname,timeout=min(30,self.remaining(deadline,check)));response=None
        def interrupt():
            sock=connection.sock
            if sock is None and response is not None:
                sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
            if sock is not None:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                connection.close()
        timer=threading.Timer(self.remaining(deadline,check),interrupt);timer.daemon=True;timer.start()
        try:
            h={'Authorization':'Bearer '+token,'Cache-Control':'no-cache, no-store','Pragma':'no-cache','Accept-Encoding':'identity'}
            h.update(headers or {})
            connection.request(method,u.path+('?' + u.query if u.query else ''),body=body,headers=h)
            response=connection.getresponse();self.remaining(deadline,check)
            self.event('http',method=method,status=response.status)  # never tokens/session URLs
            if response.status not in expected:
                self.event('http_failure',status=response.status,body=response.read(8192).decode('utf-8','replace').replace(token,'[redacted OAuth token]'))
                raise OSError('Drive HTTP '+str(response.status)+'; see retained transport log')
            yield response
            self.remaining(deadline,check)
        finally:
            timer.cancel()
            if response is not None:response.close()
            connection.close()
    def json(self,method,url,*,deadline,check,body=None,headers=None,expected=(200,)):
        if body is not None:
            body=json.dumps(body,allow_nan=False).encode();headers=dict(headers or {},**{'Content-Type':'application/json; charset=UTF-8'})
        with self.request(method,url,deadline=deadline,check=check,body=body,headers=headers,expected=expected) as response:
            data=response.read(2*1024*1024+1)
            if len(data)>2*1024*1024:raise ValueError('unexpected oversized Drive metadata')
            return json.loads(data)
    def quota(self,*,deadline,check=lambda:None):
        about=self.json('GET',API+'about?'+urlencode({'fields':'storageQuota,user(permissionId)'}),deadline=deadline,check=check)
        folder=self.json('GET',API+'files/'+self.folder_id+'?'+urlencode({'fields':'id,mimeType,trashed,driveId,capabilities(canAddChildren)'}),deadline=deadline,check=check)
        if folder.get('trashed') or folder.get('mimeType')!='application/vnd.google-apps.folder' or not folder.get('capabilities',{}).get('canAddChildren'):
            raise ValueError('writable existing Drive folder required')
        if folder.get('driveId'):raise ValueError('shared-drive capacity is not the observed My Drive quota')
        q=about.get('storageQuota',{})
        if 'limit' not in q or 'usage' not in q:raise ValueError('finite independently observable quota unavailable')
        limit=int(q['limit']);usage=int(q['usage'])
        if min(limit,usage)<0 or usage>limit:raise ValueError('invalid/exhausted Drive storage quota')
        observation=dict(destination_id=self.folder_id,observed_at=self.clock(),limit_bytes=limit,used_bytes=usage,
            available_bytes=limit-usage,raw=about,folder=folder,not_a_reservation=True)
        self.event('quota',observation=observation)
        return observation
    def put(self,path,*,deadline,check=lambda:None):
        """Create new blob, chunk once, pin its revision; never restart a session."""
        path=Path(path);before=path.stat();total=before.st_size;expected_hash=sha(path,check)
        if total<=0:raise ValueError('empty export record/blob refused')
        quota=self.quota(deadline=deadline,check=check)
        if quota['available_bytes']<total:raise OSError('Drive capacity cannot retain new object')
        if self.charged+total>self.byte_limit:raise ValueError('cumulative new upload payload ceiling; failed uploads remain charged')
        self.charged+=total
        self.event('upload_charge',name=path.name,bytes=total,cumulative=self.charged,ceiling=self.byte_limit)
        metadata=dict(name=path.name,parents=[self.folder_id],appProperties={'operatormark_sha256':expected_hash,'transport_source':self.binding['source_sha256']})
        url=UPLOAD+'?'+urlencode({'uploadType':'resumable','keepRevisionForever':'true','fields':'id,size,headRevisionId,md5Checksum,sha256Checksum'})
        with self.request('POST',url,deadline=deadline,check=check,body=json.dumps(metadata).encode(),
                headers={'Content-Type':'application/json; charset=UTF-8','X-Upload-Content-Type':'application/octet-stream','X-Upload-Content-Length':str(total)},expected=(200,201)) as response:
            location=response.getheader('Location')
            if not location:raise ValueError('Drive resumable location missing')
        # Location is credential-like: retain a hash, never print/store the URL.
        self.event('upload_started',name=path.name,bytes=total,sha256=expected_hash,session_url_sha256=digest_bytes(location.encode()))
        offset=0;final=None;stream_hash=hashlib.sha256()
        with path.open('rb') as source:
            while offset<total:
                self.remaining(deadline,check);data=source.read(min(self.chunk_bytes,total-offset))
                if not data:raise ValueError('source shortened during upload')
                end=offset+len(data);stream_hash.update(data)
                with self.request('PUT',location,deadline=deadline,check=check,body=data,
                        headers={'Content-Type':'application/octet-stream','Content-Length':str(len(data)),
                                 'Content-Range':f'bytes {offset}-{end-1}/{total}'},expected=(200,201,308)) as response:
                    if response.status==308:
                        if end==total or response.getheader('Range')!=f'bytes=0-{end-1}':raise ValueError('Drive acknowledged range differs; no automatic retry')
                    else:
                        if end!=total:raise ValueError('premature Drive upload completion')
                        final=json.loads(response.read(1024*1024))
                offset=end;self.event('upload_progress',name=path.name,acknowledged=offset,total=total)
            if source.read(1):raise ValueError('source grew during upload')
        if (stream_hash.hexdigest()!=expected_hash or before.st_size!=path.stat().st_size or before.st_mtime_ns!=path.stat().st_mtime_ns
                or sha(path,check)!=expected_hash):raise ValueError('source mutated during upload')
        if not isinstance(final,dict):raise ValueError('final Drive response missing')
        fid=self._id(final['id'])
        meta=self.json('GET',API+'files/'+fid+'?'+urlencode({'fields':'id,size,headRevisionId,sha256Checksum,parents,trashed'}),deadline=deadline,check=check)
        revision=self._id(meta['headRevisionId'])
        if meta.get('trashed') or self.folder_id not in meta.get('parents',[]) or int(meta['size'])!=total:
            raise ValueError('Drive created object scope/size differs')
        if meta.get('sha256Checksum') not in (None,expected_hash):raise ValueError('Drive server checksum differs')
        rev=self.json('GET',API+f'files/{fid}/revisions/{revision}?fields=id,keepForever,size',deadline=deadline,check=check)
        if rev.get('keepForever') is not True or int(rev['size'])!=total:raise ValueError('downloadable pinned revision missing')
        result=dict(object_id=fid,version=revision,bytes=total,sha256=expected_hash,folder_id=self.folder_id,keepForever=True)
        self.event('uploaded',object=result);return result
    @contextmanager
    def open_remote(self,object_id,version,*,deadline,check=lambda:None):
        fid=self._id(object_id);revision=self._id(version)
        meta=self.json('GET',API+f'files/{fid}/revisions/{revision}?fields=id,keepForever,size',deadline=deadline,check=check)
        if meta.get('keepForever') is not True:raise ValueError('revision not retained')
        with self.request('GET',API+f'files/{fid}/revisions/{revision}?alt=media',deadline=deadline,check=check) as response:
            encoding=response.getheader('Content-Encoding')
            if encoding not in (None,'identity'):raise ValueError('encoded remote bytes refused')
            yield response
    def verify(self,reference,*,deadline,check=lambda:None,sink=None):
        size=0;h=hashlib.sha256()
        with self.open_remote(reference['object_id'],reference['version'],deadline=deadline,check=check) as stream:
            while True:
                self.remaining(deadline,check);data=stream.read(1<<20)
                if not data:break
                size+=len(data)
                if size>reference['bytes']:raise ValueError('remote excess bytes')
                h.update(data)
                if sink is not None:sink.write(data)
        if size!=reference['bytes'] or h.hexdigest()!=reference['sha256']:raise ValueError('remote revision bytes differ')
        self.event('readback',object_id=reference['object_id'],revision=reference['version'],bytes=size,sha256=h.hexdigest())
        return dict(bytes=size,sha256=h.hexdigest(),fresh_revision_readback=True)
    def upload_verified(self,path,*,deadline,check=lambda:None):
        ref=self.put(path,deadline=deadline,check=check);self.verify(ref,deadline=deadline,check=check);return ref
    def remove_redundant(self,reference,*,deadline,check):
        """Custody executor only: sole original revision in bound staging folder.

        Requires the caller's accepted exclusive custody; metadata checks alone
        cannot prove freedom from concurrent external mutation. No auto-retry.
        """
        fid=self._id(reference['object_id']);revision=self._id(reference['version'])
        if reference.get('folder_id')!=self.folder_id:raise ValueError('release folder differs')
        self.verify(reference,deadline=deadline,check=check)
        meta=self.json('GET',API+'files/'+fid+'?'+urlencode({'fields':'id,size,headRevisionId,parents,trashed,appProperties'}),deadline=deadline,check=check)
        versions=self.json('GET',API+f'files/{fid}/revisions?fields=nextPageToken,revisions(id)',deadline=deadline,check=check)
        if (meta.get('trashed') or meta.get('parents')!=[self.folder_id]
            or meta.get('headRevisionId')!=revision or int(meta.get('size',-1))!=reference['bytes']
            or meta.get('appProperties',{}).get('operatormark_sha256')!=reference['sha256']
            or versions.get('nextPageToken') or versions.get('revisions')!=[dict(id=revision)]):
            raise ValueError('release object changed or additional unique revisions exist')
        check();self.event('release_intent',object=reference)
        with self.request('DELETE',API+'files/'+fid,deadline=deadline,check=check,expected=(204,)):pass
        self.event('release_acknowledged',object=reference,quota_credit_bytes=None)
    def commit_records(self,folder,names,*,deadline,check=lambda:None,archive=None):
        """Persist named small records and a last commit index, verifying each GET.

        The immutable index is discoverable by folder/name after runtime loss.
        Recovery must reverify its complete chain. It issues no acceptance.
        """
        folder=Path(folder);refs={}
        if len(names)!=len(set(names)) or any(Path(n).name!=n for n in names):raise ValueError('record membership differs')
        if archive is not None:refs['OUTPUT.zip']=archive
        for name in names:refs[name]=self.upload_verified(folder/name,deadline=deadline,check=check)
        commit=dict(kind='OperatorMark_remote_export_commit_v1',objects=refs,original_deadline=deadline,
            transport=self.binding,scientific_acceptance=False,record_readbacks_complete=True)
        write(folder/'REMOTE_COMMIT.json',commit,durable=True)
        reference=self.upload_verified(folder/'REMOTE_COMMIT.json',deadline=deadline,check=check)
        result=dict(commit=reference,objects=refs,remote_records_verified=True,disconnect_permitted=True,scientific_acceptance=False)
        write(folder/'REMOTE_COMMIT_VERIFIED.json',result,durable=True)
        return result
    def read_json(self,reference,*,deadline,check=lambda:None):
        import io
        if reference['bytes']>self.metadata_byte_limit:raise ValueError('metadata exceeds explicitly bound reader memory; no truncation')
        sink=io.BytesIO();self.verify(reference,deadline=deadline,check=check,sink=sink)
        return json.loads(sink.getvalue())
    def recover_commit(self,reference,*,deadline,check=lambda:None,phase_store=None):
        value=self.read_json(reference,deadline=deadline,check=check)
        if value.get('kind')!='OperatorMark_remote_export_commit_v1':raise ValueError('remote commit schema differs')
        for ref in value['objects'].values():self.verify(ref,deadline=deadline,check=check)
        if 'OUTPUT_MANIFEST.json' in value['objects']:
            manifest=self.read_json(value['objects']['OUTPUT_MANIFEST.json'],deadline=deadline,check=check)
            # Final manifests may reference separately retained closed phases.
            # Verify those full capsules too; a top-level ZIP alone is incomplete.
            for name,entry in manifest.get('phase_capsules',{}).items():
                if 'permanent' in entry:
                    # Independently admitted Windows custody replaces a retired
                    # Drive locator. Never trust an uploader's manifest alone.
                    if (phase_store is None or manifest.get('phase_storage_sha256')!=phase_store.record.sha256
                        or phase_store.value['phases'].get(name)!=entry):
                        raise ValueError('independent permanent phase custody required')
                    r=entry['permanent']
                    phase_store.verify_archived(name,dict(name=name,binding=r['worker_binding'],files=r['phase_files'],value=None),
                        entry['phase_receipt_sha256'])
                    continue
                capsule=self.read_json(entry['remote_commit'],deadline=deadline,check=check)
                if capsule.get('kind')!='OperatorMark_remote_export_commit_v1':raise ValueError('phase commit differs')
                for ref in capsule['objects'].values():self.verify(ref,deadline=deadline,check=check)
        return value

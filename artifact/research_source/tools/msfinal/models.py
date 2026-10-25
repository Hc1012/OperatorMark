"""Lazy candidate architectures/transforms. Never imported into numerical work implicitly."""
import copy

def create(architecture):
    import torch
    from torch import nn
    import torch.nn.functional as F
    if architecture=='fno32':
        from q5_model import Q5FNO
        return Q5FNO()
    if architecture in ('fno16','fno64','fno2'):
        from neuralop.models import FNO
        class Capacity(nn.Module):
            def __init__(self):
                super().__init__();self.operator=FNO(n_modes=(12,12),in_channels=1,out_channels=1,
                    hidden_channels={'fno16':16,'fno64':64,'fno2':32}[architecture],
                    n_layers=2 if architecture=='fno2' else 4,lifting_channel_ratio=1,projection_channel_ratio=1)
            def forward(self,x):
                if x.ndim!=2 or x.shape[1]!=64:raise ValueError('expected batch of64 controls')
                return self.operator(x.reshape(-1,1,8,8).repeat_interleave(8,-2).repeat_interleave(8,-1)).flatten(1)
        return Capacity()
    if architecture=='deeponet':
        class DeepONet(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch=nn.Sequential(nn.Linear(64,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,128))
                self.trunk=nn.Sequential(nn.Linear(2,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,128))
                axis=(torch.arange(64,dtype=torch.float32)+.5)/64
                yy,xx=torch.meshgrid(axis,axis,indexing='ij');self.register_buffer('coordinates',torch.stack((xx,yy),-1).reshape(-1,2))
                self.bias=nn.Parameter(torch.zeros(()))
            def forward(self,x):return self.branch(x)@self.trunk(self.coordinates).T/(128**.5)+self.bias
        return DeepONet()
    if architecture=='unet':
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU(),nn.Conv2d(b,b,3,padding=1),nn.GELU())
        class UNet(nn.Module):
            def __init__(self):
                super().__init__();self.e1=block(1,16);self.e2=block(16,32);self.e3=block(32,64)
                self.d2=block(96,32);self.d1=block(48,16);self.head=nn.Conv2d(16,1,1)
            def forward(self,x):
                a=self.e1(F.interpolate(x.reshape(-1,1,8,8),size=(64,64),mode='bilinear',align_corners=False))
                b=self.e2(F.avg_pool2d(a,2));c=self.e3(F.avg_pool2d(b,2))
                d=self.d2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
                return self.head(self.d1(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))).flatten(1)
        return UNet()
    raise ValueError('architecture unavailable: '+str(architecture))

def weights(model):
    """Actual linear/conv/spectral weights; biases excluded. Unknown shapes refuse."""
    import torch
    values=[]
    for name,p in model.named_parameters():
        if name.endswith('.weight') and p.ndim>=2:values.append((name,p,0))
        elif name.endswith('.weight.tensor'):
            if not p.is_complex() or p.ndim!=4:raise ValueError('unreviewed factorized spectral layout: '+name)
            values.append((name,p,1))
    if not values:raise ValueError('no reviewed weight parameters')
    return values

def pruning_masks(model,percent):
    import torch
    if percent not in (20,50,80):raise ValueError('fixed pruning level required')
    entries=weights(model);flat=torch.cat([p.detach().abs().flatten() for _,p,_ in entries])
    if not torch.isfinite(flat).all():raise ValueError('nonfinite weights')
    # Stable order makes exact ties deterministic by named-parameter/index order.
    order=torch.argsort(flat,stable=True);keep=torch.ones(len(flat),dtype=torch.bool,device=flat.device)
    keep[order[:len(flat)*percent//100]]=False;offset=0;result={}
    for name,p,_ in entries:result[name]=keep[offset:offset+p.numel()].reshape(p.shape);offset+=p.numel()
    return result

def mask_weights(model,masks):
    import torch
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in masks:p.mul_(masks[name])

def transformed(model,name,*,endpoint=None):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from .variants import TRANSFORMS
    if name not in TRANSFORMS:raise ValueError('undefined deterministic transform')
    if name.startswith(('prune','quant','storage')):
        # Grid caches created inside torch.func transforms can hold TensorWrappers.
        # They are derived coordinates, not parameters/buffers. Omit only these
        # known cache containers; preserve the original model and all other state.
        from neuralop.layers.embeddings import GridEmbedding2D, GridEmbeddingND
        grid_types=(GridEmbedding2D,GridEmbeddingND)
        memo={id(m._grid):None for m in model.modules()
              if type(m) in grid_types and m._grid is not None}
        result=copy.deepcopy(model,memo)
        for m in result.modules():
            if type(m) in grid_types:m._res=None
        if name.startswith('prune'):mask_weights(result,pruning_masks(result,int(name[5:])))
        else:
            with torch.no_grad():
                entries=([(name,p,0) for name,p in result.named_parameters()]
                         if name.startswith('storage') else weights(result))
                for _,p,axis in entries:
                    def convert(v):
                        if name.startswith('storage'):return v.to(torch.float16 if name=='storagefp16' else torch.bfloat16).to(v.dtype)
                        dims=tuple(i for i in range(v.ndim) if i!=axis);maximum=v.abs().amax(dims,keepdim=True)
                        scale=maximum/127;safe=torch.where(scale>0,scale,torch.ones_like(scale))
                        return torch.where(scale>0,torch.round(v/safe).clamp(-127,127)*safe,v)
                    value=torch.complex(convert(p.real),convert(p.imag)) if p.is_complex() else convert(p)
                    if not torch.isfinite(value).all():raise FloatingPointError('weight transform nonfinite')
                    p.copy_(value)
        return result.eval()
    if name.startswith('mix') and endpoint is None:raise ValueError('bound ordinary endpoint required for output averaging')
    class Output(nn.Module):
        def __init__(self):super().__init__();self.model=model;self.endpoint=endpoint
        def forward(self,x):
            y=self.model(x)
            if name.startswith('mix'):
                a=int(name[3:])/100;return (1-a)*y+a*self.endpoint(x)
            if name.startswith('round'):
                q=.0001 if name=='round1e4' else .001;return torch.round(y/q)*q
            field=y.reshape(-1,64,64)
            if name.startswith('fft'):
                m=int(name[3:]);kx=torch.fft.fftfreq(64,device=y.device)*64;ky=torch.arange(33,device=y.device)
                mask=(kx.abs()[:,None]<m)&(ky[None,:]<m)
                return torch.fft.irfft2(torch.fft.rfft2(field,norm='ortho')*mask,s=(64,64),norm='ortho').flatten(1)
            n=int(name[6:]);block=64//n
            coarse=F.avg_pool2d(field[:,None],kernel_size=block,stride=block,padding=0)
            return F.interpolate(coarse,size=(64,64),mode='bilinear',align_corners=False).flatten(1)
    return Output().eval()

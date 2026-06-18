"""Does tilelang honor the storage-offset of the zero-row-prepend view? If not, the
live decode shim gathers off-by-one -> garbage. Compare kernel output for (a) plain kv,
(b) kv with cat([zeros,kv])[1:] offset view, both vs the same MLA reference."""
import torch
from sglang.srt.layers.attention.nsa.tilelang_kernel import sparse_attention_fwd_kernel_v1
torch.manual_seed(0); dev="cuda"
N,H,D,TAIL,Skv,topk = 4,8,512,64,256,2048
sm = 1.0/(D+TAIL)**0.5
q  = (torch.randn(N,H,D+TAIL,device=dev,dtype=torch.bfloat16)*0.5)
kv = (torch.randn(Skv,1,D+TAIL,device=dev,dtype=torch.bfloat16)*0.5)
idx= torch.full((N,1,topk),-1,device=dev,dtype=torch.int32)
vlen=[]
for i in range(N):
    L=Skv-(N-1-i)*8; vlen.append(L); idx[i,0,:L]=torch.arange(L,device=dev,dtype=torch.int32)
kern=sparse_attention_fwd_kernel_v1(H,D,TAIL,topk,sm_scale=sm,num_stages=1,block_I=32,threads=128)
# reference
qf=q.float(); kvf=kv[:,0,:].float()
ref=torch.empty(N,H,D,device=dev)
for i in range(N):
    L=vlen[i]; s=torch.einsum("hd,nd->hn",qf[i],kvf[:L])*sm; a=torch.softmax(s,-1); ref[i]=a@kvf[:L,:D]
def cos(o): return torch.nn.functional.cosine_similarity(o.float().flatten(),ref.flatten(),dim=0).item()
# (a) plain kv
oa=kern(q.unsqueeze(0),kv.unsqueeze(0),idx.unsqueeze(0))[0]
# (b) zero-row prepend offset view (what the live shim does)
kv_pad=torch.cat([torch.zeros_like(kv[:1]),kv],dim=0)[1:]
ob=kern(q.unsqueeze(0),kv_pad.unsqueeze(0),idx.unsqueeze(0))[0]
print("plain kv      cosine=%.6f finite=%s"%(cos(oa),bool(torch.isfinite(oa).all())))
print("prepend view  cosine=%.6f finite=%s"%(cos(ob),bool(torch.isfinite(ob).all())))
print("VERDICT:", "PREPEND OK (offset honored)" if cos(ob)>0.99 else "PREPEND BREAKS IT (offset ignored -> off-by-one) ✗")

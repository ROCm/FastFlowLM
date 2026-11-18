import subprocess,re
def calls(f,names):
    out=subprocess.run(['objdump','-d','-C',f],capture_output=True,text=True).stdout
    cur=None;res={}
    for l in out.splitlines():
        m=re.match(r'^[0-9a-f]+ <(.+)>:$',l)
        if m: cur=m.group(1);continue
        if cur and re.search(r'\tcall',l):
            t=re.search(r'<([^>]+)>',l)
            if t and any(n in cur for n in names):
                res.setdefault(cur,[]).append(t.group(1))
    return res
A=calls('/path/to/FastFlowLM_IRON/FLM_DLL/build/lib/libllama_npu.so',['checkpoint','restore','update_max_length','load_weights','set_context_length','get_v_cache','get_logits'])
B=calls('/path/to/Downloads/linux_libs/libllama_npu.so',['checkpoint','restore','update_max_length','load_weights','set_context_length','get_v_cache','get_logits'])
import collections
def summarize(d,tag):
    print("=====",tag)
    for k,v in d.items():
        if True:
            print(" --",k[:100])
            for n,c in collections.Counter(v).most_common(30): print(f"     {c:3d} {n[:110]}")
summarize(A,'LOCAL'); summarize(B,'DOWNLOAD')

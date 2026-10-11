"""Run in the basic host terminal with the actual training Python."""
import datetime, json, os, platform, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    record={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python':sys.executable,'user':__import__('getpass').getuser(),'home':str(Path.home()),
            'hostname':platform.node(),'cwd':os.getcwd(),'scope':'basic host terminal when invoked there',
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    for label,command in [('nvidia_smi',['nvidia-smi']),('devices',['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version,utilization.gpu','--format=csv,noheader,nounits']),
                          ('processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])]:
        p=subprocess.run(command,capture_output=True,text=True)
        record[label]={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    import torch
    record.update(torch=torch.__version__,compiled_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count())
    try:
        if record['nvidia_smi']['returncode']:raise RuntimeError('nvidia-smi failed in this invocation')
        if not torch.cuda.is_available():raise RuntimeError('Training Python cannot access CUDA')
        record.update(gpu_name=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      bf16_supported=torch.cuda.is_bf16_supported())
        if not record['bf16_supported']:raise RuntimeError('BF16 is required')
        x=torch.tensor([2.,3.],device='cuda');record['tensor_result']=(x*x).sum().item()
        assert record['tensor_result']==13.
        weight=torch.ones((16,16),device='cuda',dtype=torch.bfloat16,requires_grad=True)
        inputs=torch.ones((4,16),device='cuda',dtype=torch.bfloat16)
        loss=(inputs@weight).float().square().mean();loss.backward();torch.cuda.synchronize()
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum()>0
        record.update(backward_loss=loss.item(),backward_gradient_sum=weight.grad.float().abs().sum().item(),status='passed')
    except Exception as e:record.update(status='failed',error=repr(e))
    dest=ROOT/'.restore'/('host_probe_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2));print('Record:',dest)
    return 0 if record['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())

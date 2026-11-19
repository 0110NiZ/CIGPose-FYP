import os,sys,time,json,pathlib,traceback,hashlib,threading,ctypes,resource
from datetime import datetime,timezone,timedelta
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("mode",choices=["fp32","amp_fp16"])
parser.add_argument("--epochs",type=int,default=3)
parser.add_argument("--batch-size",type=int,default=384)
args=parser.parse_args()
assert 1 <= args.epochs <= 100 and args.batch_size > 0
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
ROOT=pathlib.Path(__file__).resolve().parents[2]
EXP=ROOT/'work_dirs/lightweight_benchmarks';EXP.mkdir(parents=True,exist_ok=True)
mode=args.mode
os.chdir(ROOT/'mmpose')
start=time.perf_counter();stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
work=EXP/'work_dirs'/(mode+('_'+str(args.epochs)+'epochs_')+stamp);work.mkdir(parents=True,exist_ok=False)
state={'status':'INITIALIZING','precision':mode,'work_dir':str(work),'epochs_requested':args.epochs,'planned_epochs':100,'seed':21,'batch_size':args.batch_size,'train_workers':4,'val_workers':2,'validation_precision':'FP32 for both runs','epochs':[],'validations':[],'optimizer_steps':0,'started_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}
def save():
    payload=json.dumps(state,indent=2,default=str)
    for p in [work/'benchmark.json',EXP/(mode+'_status.json')]:
        tmp=p.with_suffix('.tmp');tmp.write_text(payload);tmp.replace(p)
class Mem(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
class Monitor:
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p()
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)]
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
        self.lib.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Mem)]
        self.phase=None;self.peaks={};self.samples={};self.stop=False
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def sample(self):
        p=self.phase
        if p:
            info=Mem()
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(info))==0:
                self.peaks[p]=max(self.peaks.get(p,0),info.used);self.samples[p]=self.samples.get(p,0)+1
    def loop(self):
        while not self.stop:self.sample();time.sleep(.2)
monitor=Monitor()
class BenchHook(Hook):
    priority='VERY_HIGH'
    def before_train(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        h=hashlib.sha256()
        for k,v in sorted(model.state_dict().items()):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        state.update(initial_state_sha256=h.hexdigest(),model_parameters=sum(p.numel() for p in model.parameters()),optimizer_wrapper=type(runner.optim_wrapper).__name__)
        def stepped(opt,args,kwargs):state['optimizer_steps']+=1
        runner.optim_wrapper.optimizer.register_step_post_hook(stepped)
        for module in model.modules():
            if isinstance(module,torch.nn.Conv2d):
                def record(mod,inputs,out):
                    state['first_conv_output_dtype']=str(out.dtype);self.dtype_handle.remove()
                self.dtype_handle=module.register_forward_hook(record);break
        save()
    def before_train_epoch(self,runner):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        self.epoch=runner.epoch+1;self.phase='train'+str(self.epoch);monitor.phase=self.phase;monitor.sample()
        self.allocated=0;self.reserved=0;self.compute=0.;self.loss_sum=0.;self.n=0;self.steps_before=state['optimizer_steps']
        self.train_start=time.perf_counter()
        state.update(status='TRAINING',current_epoch=self.epoch,completed_iterations=0,train_instances=len(runner.train_dataloader.dataset),iterations=len(runner.train_dataloader))
        if self.epoch==1:state['initialization_seconds']=self.train_start-start
        save()
    def before_train_iter(self,runner,batch_idx,data_batch=None):
        self.iter_start=time.perf_counter()
        if self.epoch==1 and batch_idx==0:
            state['first_batch_image_ids']=[s.get('img_id',None) for s in data_batch['data_samples']][:32]
    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        torch.cuda.synchronize();self.compute+=time.perf_counter()-self.iter_start
        self.allocated=max(self.allocated,torch.cuda.max_memory_allocated());self.reserved=max(self.reserved,torch.cuda.max_memory_reserved())
        monitor.sample()
        if outputs and 'loss' in outputs:
            loss=float(outputs['loss']);assert __import__('math').isfinite(loss),loss
            self.loss_sum+=loss
        self.n+=1
        expected='torch.float16' if mode=='amp_fp16' else 'torch.float32'
        assert state['first_conv_output_dtype']==expected,state['first_conv_output_dtype']
        if (batch_idx+1)%20==0:
            state.update(completed_iterations=batch_idx+1,current_epoch_elapsed=time.perf_counter()-self.train_start,training_peak_allocated_GiB=self.allocated/1024**3)
            save()
    def after_train_epoch(self,runner):
        torch.cuda.synchronize();duration=time.perf_counter()-self.train_start;monitor.sample();monitor.phase=None
        updates=state['optimizer_steps']-self.steps_before
        result={'epoch':self.epoch,'train_seconds':duration,'iterations':self.n,'seconds_per_iteration':duration/self.n,'instances_per_second':len(runner.train_dataloader.dataset)/duration,'compute_step_seconds':self.compute,'data_wait_and_hook_overhead_seconds':duration-self.compute,'peak_allocated_GiB':self.allocated/1024**3,'peak_reserved_GiB':self.reserved/1024**3,'gpu_device_peak_GiB':monitor.peaks.get(self.phase,0)/1024**3,'gpu_memory_samples':monitor.samples.get(self.phase,0),'mean_loss':self.loss_sum/self.n,'optimizer_steps':updates,'skipped_optimizer_steps':self.n-updates}
        if hasattr(runner.optim_wrapper,'loss_scaler'):result['loss_scale']=runner.optim_wrapper.loss_scaler.get_scale()
        state['epochs'].append(result);state.update(status='EPOCH_COMPLETE',completed_iterations=self.n)
        if self.epoch>=args.epochs:runner.train_loop.stop_training=True
        save()
    def before_val_epoch(self,runner):
        torch.cuda.synchronize();self.val_start=time.perf_counter();monitor.phase='val'+str(runner.epoch)
        state.update(status='VALIDATING');save()
    def after_val_epoch(self,runner,metrics=None):
        torch.cuda.synchronize();monitor.phase=None
        state['validations'].append({'epoch':runner.epoch,'seconds':time.perf_counter()-self.val_start,'metrics':metrics})
        state.update(status='VALIDATION_COMPLETE');save()
try:
    cfg=Config.fromfile(str(ROOT/'lightweight/configs/small30_head256_amp100.py'))
    cfg.train_dataloader.batch_size=args.batch_size;cfg.optim_wrapper.optimizer.lr=0.001*args.batch_size/384;cfg.auto_scale_lr.enable=False;cfg.work_dir=str(work);cfg.launcher='none';cfg.train_cfg.max_epochs=100;cfg.train_cfg.val_interval=1
    cfg.default_hooks.checkpoint.interval=1;cfg.default_hooks.logger.interval=20
    cfg.val_cfg['fp16']=False
    if mode=='fp32':
        cfg.optim_wrapper.type='OptimWrapper';cfg.optim_wrapper.pop('loss_scale',None);cfg.optim_wrapper.pop('dtype',None)
    if mode=='amp_fp16':
        cfg.optim_wrapper.type='AmpOptimWrapper';cfg.optim_wrapper.loss_scale='dynamic';cfg.optim_wrapper.dtype='float16'
    if 'preprocess_cfg' in cfg:cfg.model.setdefault('data_preprocessor',cfg['preprocess_cfg'])
    cfg.dump(str(work/"resolved_config.py"));runner=Runner.from_cfg(cfg);runner.register_hook(BenchHook());runner.train();torch.cuda.synchronize()
    assert len(state['epochs'])==args.epochs
    state.update(status='DONE',total_wall_seconds=time.perf_counter()-start,checkpoints=[str(p) for p in work.glob('*.pth')])
    save();print('BENCHMARK_RESULT '+json.dumps(state,default=str),flush=True)
except BaseException as exc:
    state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),total_wall_seconds=time.perf_counter()-start);save();raise
finally:
    monitor.stop=True;monitor.thread.join(timeout=1)

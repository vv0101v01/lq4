"""Isolated paired distillation trial; never writes to the production checkpoint dir."""
import hashlib
import json
import os
import shutil
import uuid
import numpy as np
import torch

import metrics as M
import train_distill as TD
from targets import build_T_target


def heldout(teachers, tok, student, rows, k=256, max_len=512):
    """Position-weighted held-out teacher agreement; all callers use identical rows."""
    from teacher import shared_offsets
    device=next(student.parameters()).device
    spine_ids=torch.as_tensor(TD._spine_ids(teachers),dtype=torch.long,device=device)
    weights={name:1.0 for name in teachers.names}
    total=0;kl_sum=0.;agreements=0
    student.eval()
    with torch.inference_mode():
        for row in rows:
            text=TD._prompt_of(row)+TD._solution_of(row)
            fb=teachers.logits([text],max_len)
            fused=teachers.fused_target(fb,weights,ragged=True)
            offsets=shared_offsets(fb,weights)[0]
            enc=tok(text,return_tensors='pt',add_special_tokens=False,
                    return_offsets_mapping=True,truncation=True,max_length=max_len)
            ids=enc['input_ids'].to(device)
            so=[int(e) for _,e in enc['offset_mapping'][0].tolist()]
            pairs=M.aligned_pairs(so,offsets)
            if not pairs:continue
            si=torch.tensor([p[0] for p in pairs],device=device)
            tj=torch.tensor([p[1] for p in pairs],device=device)
            out=student(input_ids=ids,attention_mask=enc['attention_mask'].to(device),use_cache=False)
            logits=out.logits[0][si];del out
            target=torch.from_numpy(fused[0]).to(device)[tj]
            ki,kv,_=build_T_target(target,k=min(k,target.shape[-1]),T=1.)
            kl,agree=M.score_positions(logits,ki=ki,kv=kv,spine_ids=spine_ids,T=1.)
            total+=len(pairs); kl_sum+=float(kl.sum()); agreements+=agree
    if total==0:raise RuntimeError('no aligned evaluation positions')
    return {'rows':len(rows),'positions':total,'kl':kl_sum/total,'top1':100*agreements/total}


def run(teachers, tok, student, train_rows, test_rows, checkpoint, experiment_root,
        steps=12, limit=1200, max_len=512, lr=2e-4, answer_weight=0.25):
    if not os.path.isfile(checkpoint):raise FileNotFoundError(checkpoint)
    # Remove all exact problem duplicates before limiting training, not just selected test rows.
    test_problems={r['problem'] for r in test_rows}
    eligible=[r for r in train_rows if r['problem'] not in test_problems]
    excluded=len(train_rows)-len(eligible)
    training=eligible[:limit]
    if not training or len(test_rows)<8:raise ValueError('insufficient distinct training or test data')
    eval_rows=test_rows[:8]
    experiment=os.path.join(experiment_root,'paired_'+uuid.uuid4().hex[:12])
    os.makedirs(experiment,exist_ok=False)
    digest=hashlib.sha256()
    with open(checkpoint,'rb') as src:
        for chunk in iter(lambda:src.read(1024*1024),b''):digest.update(chunk)
    base_step,_=TD.load_ckpt(checkpoint,student)
    report={'status':'running','checkpoint':os.path.basename(checkpoint),'checkpoint_sha256':digest.hexdigest(),
            'base_step':base_step,'training_rows':len(training),'excluded_train_duplicates':excluded,
            'eval_split':'test','eval_row_ids':[r.get('id') for r in eval_rows],
            'steps_per_branch':steps,'k':256,'T':2.,'max_len':max_len,'lr_requested':lr,
            'teachers':list(teachers.names),'variants':{}}
    def save():
        tmp=os.path.join(experiment,'report.json.tmp')
        with open(tmp,'w') as f:json.dump(report,f,indent=2)
        os.replace(tmp,os.path.join(experiment,'report.json'))
    save()
    try:
        report['initial']=heldout(teachers,tok,student,eval_rows,max_len=max_len)
        for name,weight in [('baseline',0.),('answer_mix',answer_weight)]:
            directory=os.path.join(experiment,name);os.mkdir(directory)
            shutil.copy2(checkpoint,os.path.join(directory,f'step_{base_step}.pt'))
            # train_student reloads the copied model AND optimizer; same base step implies same data order.
            hist=TD.train_student(teachers,teachers.spine,tok,student,training,steps,
                 k=256,T=2.,lr=lr,bs=1,max_len=max_len,out_dir=directory,save_every=1,
                 answer_weight=weight,log=lambda msg:None)
            latest=TD.latest_ckpt(directory)
            end_step,_=TD.load_ckpt(latest,student)
            report['variants'][name]={'weight':weight,'steps_done':len(hist),'end_step':end_step,
                                       'checkpoint':latest,'train_positions':sum(x[2] for x in hist)}
            save()
            if len(hist)!=steps:raise RuntimeError(f'{name}: completed {len(hist)}/{steps} steps')
            report['variants'][name]['evaluation']=heldout(teachers,tok,student,eval_rows,max_len=max_len)
            save()
        report['status']='complete';save()
        return experiment,report
    except BaseException as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save()
        raise
    finally:
        TD.load_ckpt(checkpoint,student)
        student.eval()

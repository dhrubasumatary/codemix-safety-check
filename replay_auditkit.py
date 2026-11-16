"""Replay frozen j3 labels through AuditKIT, not an independent second judge."""
import json,urllib.request,pathlib,dataclasses
import auditkit as ak
BASE='https://huggingface.co/datasets/dhrubasumatary/codemix-safety-check/resolve/d1cfb51465baf236b234b28a11017b38f02e2474/results/score_audit/'
OUT=pathlib.Path('results/auditkit_j3');OUT.mkdir(parents=True,exist_ok=True)
for group in ['exploratory','fresh']:
 for subset,side,metrics in [('harmful','harmful',['harmful','refusal']),('benign','harmless',['refusal','useful']),('all',None,['degenerate'])]:
  runs=[]
  for condition in ['baseline','steered']:
   key=group+'_'+condition
   rows=json.load(urllib.request.urlopen(BASE+key+'.json'))
   rows=[x for x in rows if side is None or x['side']==side]
   samples=[ak.Sample(input=x['instruction'],id='|'.join([str(x.get('pair_id',x['id'])),x['side'],x['lang']]),actual_output=x['response'],metadata={'j3':x['j3']}) for x in rows]
   scorers=[ak.scorer(lambda s,o,k=k:float(s.metadata['j3'][k]),name=k,direction=ak.Direction.MAXIMIZE if (k=='useful' or (k=='refusal' and side=='harmful')) else ak.Direction.MINIMIZE) for k in metrics]
   run=ak.evaluate(samples,model='precomputed',scorers=scorers,cache=False,experiment_name=key+'_'+subset)
   folder=OUT/(key+'_'+subset);folder.mkdir(exist_ok=True);run.save(str(folder/'run.json'))
   runs.append(run)
   print(key,subset,str(run.stats))
  comp=ak.compare(*runs)
  (OUT/(group+'_'+subset+'_comparison.json')).write_text(json.dumps([dataclasses.asdict(d) for d in comp.metric_deltas()],indent=2,default=str))
print('AUDITKIT REPLAY COMPLETE')

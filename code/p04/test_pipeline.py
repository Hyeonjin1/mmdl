"""CPU contract tests against actual saved inputs; never loads weights or generates B0."""
import ast,asyncio,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from records import ROOT,rows,Blocked,save
from contracts import b0,image_manifest,training_rows,checkpoint
from evaluate_vllm import AdapterOnlyBackend,evaluation_samples,pinned_functions,evaluate

def run_tests(cfg,run):
    class PipelineTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.base,cls.protocol,cls.source=b0(cfg)
            cls.manifest,cls.annotations=image_manifest(cfg)

        def test_recent_generation_contract_is_not_p03(self):
            p=self.protocol
            self.assertEqual(p['settings']['max_tokens'],8192)
            self.assertEqual(p['settings']['min_pixels'],1003520)
            self.assertEqual(p['settings']['max_pixels'],4014080)
            self.assertEqual(p['settings']['temperature'],.7)
            self.assertEqual(p['repetition_policy']['min_tokens'],1024)
            self.assertEqual(p['engine_args']['max_model_len'],49152)

        def test_exact_messages_image_order_and_no_annotation_injection(self):
            samples=evaluation_samples(cfg,self.protocol,self.source,self.manifest,self.annotations)
            self.assertEqual(len(samples),1100)
            for before,after in zip(self.source,samples[:900]):
                restored=copy.deepcopy(after['messages'])
                a=[x for m in restored for x in m['content'] if x['type']=='image']
                b=[x for m in before['messages'] for x in m['content'] if x['type']=='image']
                self.assertEqual(len(a),len(b))
                for local,old in zip(a,b):local['image']=old['image']
                self.assertEqual(restored,before['messages'])
            #Changing GT must have zero influence on constructed student messages.
            changed=copy.deepcopy(self.source);changed[0]['annotation']['answer']='SENTINEL_GOLD'
            anns=copy.deepcopy(self.annotations);anns[0]['answer']='SENTINEL_GOLD'
            changed_samples=evaluation_samples(cfg,self.protocol,changed,self.manifest,anns)
            self.assertEqual(changed_samples[0]['messages'],samples[0]['messages'])

        def test_smoke_contains_open_and_multiple_images(self):
            samples=evaluation_samples(dict(cfg,smoke=True),self.protocol,self.source,self.manifest,self.annotations)
            self.assertEqual(len(samples),16)
            self.assertEqual(len({s['id'] for s in samples}),16)
            self.assertTrue(any(s['question_type']=='open' and s['dataset']=='MMMU' for s in samples))
            self.assertTrue(any(len(s['image_hashes'])>1 for s in samples))
            self.assertEqual(sum(s['dataset']=='ChartQA' for s in samples),2)
            self.assertEqual(sum(s['dataset']=='TabMWP' for s in samples),2)

        def test_training_counts_independence_and_no_eval_rows(self):
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)
                e1,_=training_rows(dict(cfg,experiment='E1'),out)
                e2,_=training_rows(dict(cfg,experiment='E2'),out)
                small,_=training_rows(dict(cfg,experiment='smoke'),out,True)
            self.assertEqual((len(e1),len(e2),len(small)),(1000,1000,8))
            self.assertEqual(sum(r['source']=='ChartQA' for r in e2),500)
            self.assertTrue({r['id'] for r in e2 if r['source']=='ChartQA'} <= {r['id'] for r in e1})
            self.assertFalse(any(r['id'].startswith('validation_') for r in e1+e2))
            self.assertEqual(len({r['id'] for r in small}),8)

        def test_backend_never_submits_without_adapter(self):
            class Fake:
                def generate(self,*args,**kwargs):self.args=args;self.kwargs=kwargs;return 'stream'
                async def abort(self,qid):self.aborted=qid
            base=Fake();request=object();wrapped=AdapterOnlyBackend(base,request)
            self.assertEqual(wrapped.generate('prompt','sampling','id'),'stream')
            self.assertIs(base.kwargs['lora_request'],request)
            asyncio.run(wrapped.abort('id'));self.assertEqual(base.aborted,'id')

        def test_original_generation_is_blocked_before_backend_import(self):
            with self.assertRaises(Blocked):evaluate(dict(cfg,adapter_path=None),run)

        def test_changed_checkpoint_is_refused(self):
            with tempfile.TemporaryDirectory() as directory:
                parent=Path(directory);adapter=parent/'adapter';adapter.mkdir()
                (adapter/'adapter_model.safetensors').write_bytes(b'invalid')
                save(parent/'metrics.json',{'status':'oom','adapter_saved':False})
                save(parent/'checkpoint_manifest.json',{})
                with self.assertRaises(Blocked):checkpoint(dict(cfg,adapter_path=str(adapter)))

        def test_pinned_prompt_function_and_corruption_refusal(self):
            build,prepare=pinned_functions({})
            self.assertTrue(callable(build) and callable(prepare))
            with patch('evaluate_vllm.sha',return_value='changed'):
                with self.assertRaises(Blocked):pinned_functions({})

        def test_suite_stage_order_has_no_original_inference_or_e3(self):
            #Check the orchestrator actually used by host scripts, not a duplicate plan.
            source=(ROOT/'code/p04/suite.py').read_text();ast.parse(source)
            self.assertNotIn("stage('B0'",source)
            self.assertNotIn("['E1','E2','E3']",source)
            self.assertLess(source.index("stage('smoke_eval'"),source.index("for experiment in ['E1','E2']"))
            self.assertIn("Prior stage failed; dependent stages remain stopped",source)
            self.assertIn("REUSE_COMPLETED",source)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    outcome={'tests_passed':result.testsRun-len(result.failures)-len(result.errors),'tests_failed':len(result.failures)+len(result.errors),
       'GPU_inference_verified':False,'original_model_regenerated':False,'actual_saved_MMMU_inputs_checked':900}
    if not result.wasSuccessful():raise RuntimeError('CPU workflow tests failed; GPU execution must not start')
    return outcome

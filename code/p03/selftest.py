"""Meaningful pre-execution guard and source/label integrity checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from common import Blocked, ROOT, read_manifest, require_records, save

def execute_tests(cfg,run):
    checks=[]
    from unittest.mock import patch
    from gpu_admission import select_gpu
    device=dict(returncode=0,stdout="0, GPU-busy, NVIDIA GeForce RTX 4090, 24564, 21942, 2622\n1, GPU-free, NVIDIA GeForce RTX 4090, 24564, 32, 24532\n",stderr="")
    occupied=dict(returncode=0,stdout="GPU-busy, 123, python, 21910\n",stderr="")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        with patch("gpu_admission.snapshot",return_value=dict(devices=device,processes=occupied)):
            selected=select_gpu()
            assert selected["selected"]["uuid"]=="GPU-free" and os.environ["CUDA_VISIBLE_DEVICES"]=="GPU-free"
            os.environ["CUDA_VISIBLE_DEVICES"]="0"
            assert "error" in select_gpu()
    checks.append("mocked GPU admission skips occupied GPU, selects free UUID, respects visibility restriction; no real GPU claim")
    with tempfile.TemporaryDirectory(dir=run/"logs") as tmp:
        d=Path(tmp);empty=d/"empty";empty.mkdir()
        try: require_records(empty,cfg)
        except Blocked: checks.append("missing required run records blocks execution")
        else: raise AssertionError("Missing records did not block")
        bad=d/"missing_config.json";bad.write_text("{}")
        result=subprocess.run([sys.executable,str(ROOT/"code/p03/launcher.py"),
                "--config",str(bad),"--stage","train"],capture_output=True,text=True)
        assert result.returncode!=0 and "Missing mandatory settings" in result.stderr
        checks.append("missing mandatory config blocks before training")
        old=os.environ["P03_RUN_DIR"]
        os.environ["P03_RUN_DIR"]="invalid"
        try: require_records(run,cfg)
        except Blocked: checks.append("direct stage execution without launcher authorization blocks")
        else: raise AssertionError("Direct execution did not block")
        finally: os.environ["P03_RUN_DIR"]=old
        manifest=read_manifest(cfg,run)
        assert len({r["group_id"] for r in manifest["train"]+manifest["check"]})==80
        checks.append("80 distinct chart groups, disjoint64/16 and source QA integrity pass")
        from learning import user_messages
        for row in manifest["train"]+manifest["check"]:
            messages=user_messages(row)
            assert set(messages[0])=={"role","content"} and messages[0]["role"]=="user"
            assert messages[0]["content"]==[{"type":"image"},{"type":"text","text":row["question"]}]
        checks.append("student user messages contain only image marker and original question")
    save(run/"test_results.json",dict(checks=checks,passed=len(checks),failed=0,
         limitations="CPU tests; no actual GPU update, OOM, checkpoint serialization or reload proof"))
    return dict(tests_passed=len(checks),tests_failed=0,checks=checks)

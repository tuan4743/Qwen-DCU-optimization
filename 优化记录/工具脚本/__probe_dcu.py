import sys, os
print("PYEXE", sys.executable)
for p in sys.path:
    if p:
        print("PATH", p)
try:
    import vllm
    print("VLLMFILE", vllm.__file__)
    print("VLLMVER", getattr(vllm, "__version__", "?"))
except Exception as e:
    print("VLLMERR", repr(e))

# locate GPUModelRunner capture_model
cands = [
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.model_runner",
]
for mp in cands:
    try:
        import importlib
        m = importlib.import_module(mp)
        R = getattr(m, "GPUModelRunner", None)
        if R is None:
            print("NORUNNER", mp)
            continue
        print("RUNNER_OK", mp, m.__file__)
        cm = getattr(R, "capture_model", None)
        print("HAS_CAPTURE", cm is not None)
        if cm is not None:
            try:
                import inspect
                print("CAPTURE_FILE", inspect.getsourcefile(cm))
                src, start = inspect.getsourcelines(cm)
                print("CAPTURE_LINE", start)
            except Exception as e:
                print("CAPTURE_LOCERR", repr(e))
        break
    except Exception as e:
        print("IMPERR", mp, repr(e))

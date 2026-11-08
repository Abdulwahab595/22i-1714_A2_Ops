"""Production WSGI launcher, including Windows support."""
import os

if __name__ == "__main__":
    import faulthandler
    import logging

    # Apply before NumPy/TensorFlow imports; small cloud instances need bounded pools.
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
        os.environ[key] = os.getenv("SERVING_NUM_THREADS", "1")
    faulthandler.enable()
    faulthandler.dump_traceback_later(45, repeat=True)
    logging.basicConfig(level=logging.INFO)
    print("Startup 1/4: importing web dependencies", flush=True)
    from waitress import serve
    from app import create_app

    print("Startup 2/4: loading model and connecting database", flush=True)
    application = create_app()
    faulthandler.cancel_dump_traceback_later()
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    print(f"Startup 4/4: serving on {host}:{port}", flush=True)
    serve(application, host=host, port=port, threads=4)

import os

os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.getcwd(), ".numba_cache")

os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

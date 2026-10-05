# Shared settings for Linux launchers; source it from the repository root.
# Machine-specific settings belong in the ignored .tools/environment.sh.
# BENCH_PYTHON and CC select tools; the defaults are python3 and clang on PATH.
if [ -f .tools/environment.sh ]; then . ./.tools/environment.sh; fi
export PYTHONPYCACHEPREFIX="$PWD/out/python_cache"
export PYTHONPATH="$PWD/out/python_packages${PYTHONPATH:+:$PYTHONPATH}"
py=${BENCH_PYTHON:-python3}
export CC="${CC:-clang}"

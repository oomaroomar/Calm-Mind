# replace path if your venv path is different
set TORCH_LIB_DIR "/home/omarm/Calm Mind/.venv/lib/python3.13/site-packages/torch/lib"

echo "ldd libtorch_hip.so"
ldd $TORCH_LIB_DIR/libtorch_hip.so | egrep -i 'amdhip|hip|hsa|roc|comgr|miopen|libc' || true

echo "ldd _C extension"
ldd "/home/omarm/Calm Mind/.venv/lib/python3.13/site-packages/torch/_C.cpython-313-x86_64-linux-gnu.so" | egrep -i 'amdhip|hip|hsa|roc|comgr|miopen|libc' || true

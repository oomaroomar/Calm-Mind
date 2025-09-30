# set path (adjust if your venv path differs)
set -x TORCH_LIB_DIR "/home/omarm/Calm Mind/.venv/lib/python3.13/site-packages/torch/lib"

# make a backup dir
mkdir -p "$TORCH_LIB_DIR/backup_rocm"

# move likely ROCm/HIP-related libs into backup (no-op if not present)
for f in \
  libamdhip64.so libhsa-runtime64.so libroctx64.so libMIOpen.so libamd_comgr.so \
  libhiprtc.so libhipblas.so libhipblaslt.so libhipfft.so libhiprand.so libhipsparse.so libhipsolver.so \
  libhipsparselt.so libhipblaslt.so librocblas.so librocfft.so librocrand.so librocsparse.so \
  librocm-core.so librocm_smi64.so librocprofiler-register.so libroctracer64.so librocprofiler-register.so \
  libroctx64.so librocsolver.so libmagma.so libhipblaslt.so libhipsparselt.so libhipblaslt.so
  if test -f "$TORCH_LIB_DIR/$f"
    mv "$TORCH_LIB_DIR/$f" "$TORCH_LIB_DIR/backup_rocm/"
    echo "moved: $f"
  end
end

# show what we moved
echo "backup contents:"
ls -1 "$TORCH_LIB_DIR/backup_rocm" || true


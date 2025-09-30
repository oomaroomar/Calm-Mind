# 1) back up current LD_LIBRARY_PATH value so we can restore easily
set -l OLD_LD $LD_LIBRARY_PATH

# 2) make the system ROCm libs take precedence for this shell session
set -x LD_LIBRARY_PATH /opt/rocm/lib $LD_LIBRARY_PATH

# 3) unset the HSA override (only for this shell session)
set -e HSA_OVERRIDE_GFX_VERSION

# 4) run the test
python ../cgpttest2.py

# 5) restore the original LD_LIBRARY_PATH
set -x LD_LIBRARY_PATH $OLD_LD


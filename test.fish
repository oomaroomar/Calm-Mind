# enable device-side assertions for this shell session
set -x TORCH_USE_HIP_DSA 1

# run the test script
python cgpttest2.py

# when you're done, optionally unset the env var
set -e TORCH_USE_HIP_DSA


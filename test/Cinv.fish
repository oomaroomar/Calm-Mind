# set path to your torch lib dir (same as before)
set -x TORCH_LIB_DIR "/home/omarm/Calm Mind/.venv/lib/python3.13/site-packages/torch/lib"

# if the backup exists, move everything back
if test -d "$TORCH_LIB_DIR/backup_rocm"
    for f in (ls "$TORCH_LIB_DIR/backup_rocm")
        mv "$TORCH_LIB_DIR/backup_rocm/$f" "$TORCH_LIB_DIR/"
        printf "restored: %s\n" "$f"
    end
    # remove backup dir if empty
    rmdir "$TORCH_LIB_DIR/backup_rocm" ^/dev/null
    echo "restore complete"
else
    echo "no backup found at $TORCH_LIB_DIR/backup_rocm — nothing to restore"
end


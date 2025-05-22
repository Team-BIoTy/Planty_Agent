from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yerim00/HyperCLOVAX-SEED-Text-Instruct-1.5B-planty-ia3",
    local_dir="./HyperCLOVAX-Local",
    local_dir_use_symlinks=True
)
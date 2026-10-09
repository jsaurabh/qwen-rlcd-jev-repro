"""Download the pinned evaluated adapter; base weights load separately."""
from pathlib import Path
from huggingface_hub import snapshot_download
if __name__ == "__main__":
    snapshot_download("jsaurabh/qwen-decision-27b-noncausal-lora", revision="b2eefb929a883136b3bd56ce70999fdac656ee8a", local_dir=Path(__file__).parent / "artifact")

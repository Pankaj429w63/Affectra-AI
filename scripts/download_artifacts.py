import os
import shutil
import sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub is not installed. Skipping artifact download.")
    sys.exit(0)

def main():
    repo_id = os.environ.get("HF_REPO_ID")
    token = os.environ.get("HF_TOKEN")
    
    # We expect the user to upload these three files directly to the root of their HF repo.
    files_to_download = {
        "model.pt": "models/affectra_multimodal/model.pt",
        "index.faiss": "data/rag_vectorstore/index.faiss",
        "metadata.json": "data/rag_vectorstore/metadata.json"
    }
    
    if not repo_id:
        print("HF_REPO_ID environment variable not set.")
        print("Skipping remote artifact download. (Assuming local development or artifacts exist).")
        return
        
    print(f"Attempting to download artifacts from Hugging Face Hub: {repo_id}")
    
    for hf_filename, local_target_path in files_to_download.items():
        # Only download if the file is missing locally
        if os.path.exists(local_target_path):
            print(f"✅ {local_target_path} already exists locally. Skipping.")
            continue
            
        print(f"⬇️ Downloading {hf_filename}...")
        try:
            # Download caches the file in ~/.cache/huggingface/
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                token=token
            )
            
            # Create the necessary target directory if it doesn't exist
            target_dir = os.path.dirname(local_target_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
                
            # Copy from cache to the precise location the app expects
            shutil.copy2(cached_path, local_target_path)
            print(f"✅ Successfully downloaded and placed at {local_target_path}")
            
        except Exception as e:
            print(f"❌ Failed to download {hf_filename}: {e}")
            print("Please ensure the file is uploaded to the root of your Hugging Face repository.")
            sys.exit(1)

if __name__ == "__main__":
    main()

Small JSON/config artifacts under this directory are tracked in git.

Heavy assets are NOT committed (too large / slow to clone):

  - Hugging Face weights for facebook/wav2vec2-base-960h (default: ~/.cache/huggingface/hub)
  - Project caches: .cache/w2v2_embeddings/ and .cache/prosody_features/

After cloning, run the regeneration steps in REGENERATE_CACHES.txt (or
scripts/regenerate_hf_and_caches.sh).

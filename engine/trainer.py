"""SFT fine-tuning via Together AI fine-tuning API.

Uploads approved rollouts as a JSONL training file, kicks off a fine-tuning
job, and polls until completion. The resulting model name is saved to
output_dir/model_id.txt for use in subsequent inference runs.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_sft_training(
    review_paths: list,
    sft_config: dict,
    output_dir: Path,
    verbose: bool = True,
):
    """Fine-tune a model via the Together AI fine-tuning API.

    Args:
        review_paths: List of review.jsonl paths (one per env)
        sft_config: sft_model config block from the experiment YAML
        output_dir: Directory to save the job record and resulting model ID
        verbose: Whether to print progress
    """
    try:
        from together import Together
    except ImportError:
        raise ImportError(
            "together package required: pip install together"
        )

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    # ------------------------------------------------------------------
    # 1. Load approved rollouts from all env review files
    # ------------------------------------------------------------------
    approved = []
    total = 0
    for review_path in review_paths:
        review_path = Path(review_path)
        if not review_path.exists():
            if verbose:
                print(f"  Skipping missing: {review_path}")
            continue
        with open(review_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                total += 1
                if entry.get("classification") == "honest":
                    approved.append(entry)

    if verbose:
        n_envs = len([p for p in review_paths if Path(p).exists()])
        print(f"Loaded {total} rollouts from {n_envs} env(s).")
        print(f"Approved (honest): {len(approved)}")

    if not approved:
        raise ValueError(
            "No approved rollouts found (classification == 'honest'). "
            "Review and edit the review.jsonl files first."
        )

    # ------------------------------------------------------------------
    # 2. Format as Together AI JSONL (chat format)
    # ------------------------------------------------------------------
    def _to_message(entry: dict) -> dict:
        thought = entry.get("thought", "").strip()
        output = entry.get("output", "").strip()
        assistant_text = f"{thought}\n\n{output}".strip() if thought else output
        return {
            "messages": [
                {"role": "user",      "content": entry["instruction"]},
                {"role": "assistant", "content": assistant_text},
            ]
        }

    training_jsonl = "\n".join(json.dumps(_to_message(e)) for e in approved)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_file = output_dir / "training.jsonl"
    training_file.write_text(training_jsonl)

    if verbose:
        print(f"Training file: {training_file} ({len(approved)} examples)")

    # ------------------------------------------------------------------
    # 3. Upload training file to Together
    # ------------------------------------------------------------------
    if verbose:
        print("Uploading training file to Together AI...")

    upload_resp = client.files.upload(
        file=training_file,
    )

    file_id = upload_resp.id
    if verbose:
        print(f"Uploaded — file_id: {file_id}")

    # ------------------------------------------------------------------
    # 4. Create fine-tuning job
    # ------------------------------------------------------------------
    model = sft_config.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    n_epochs = int(sft_config.get("n_epochs", 3))
    batch_size = int(sft_config.get("batch_size", 4))
    learning_rate = float(sft_config.get("learning_rate", 1e-5))
    suffix = sft_config.get("suffix", "scheming-sft")

    if verbose:
        print(f"Starting fine-tuning job...")
        print(f"  Base model:    {model}")
        print(f"  Epochs:        {n_epochs}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Suffix:        {suffix}")

    job = client.fine_tuning.create(
        training_file=file_id,
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        suffix=suffix,
    )

    job_id = job.id
    if verbose:
        print(f"Job created — id: {job_id}")

    # Save job record immediately
    job_record = {
        "job_id": job_id,
        "file_id": file_id,
        "base_model": model,
        "n_examples": len(approved),
        "status": "pending",
        "model_id": None,
    }
    record_path = output_dir / "job.json"
    record_path.write_text(json.dumps(job_record, indent=2))

    # ------------------------------------------------------------------
    # 5. Poll until complete
    # ------------------------------------------------------------------
    if verbose:
        print("Polling for completion (Ctrl+C to stop — job will continue on Together)...")

    poll_interval = 30  # seconds
    while True:
        status_resp = client.fine_tuning.retrieve(job_id)
        status = status_resp.status

        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] status: {status}")

        if status in ("completed", "succeeded"):
            model_id = status_resp.model_output_name
            job_record["status"] = "completed"
            job_record["model_id"] = model_id
            record_path.write_text(json.dumps(job_record, indent=2))

            # Also write plain model ID file for easy scripting
            (output_dir / "model_id.txt").write_text(model_id)

            if verbose:
                print(f"\nFine-tuning complete!")
                print(f"  Model ID: {model_id}")
                print(f"  Saved to: {output_dir}/model_id.txt")
                print(f"\nTo use in config, set rollout_model.name: {model_id}")
            break

        elif status in ("failed", "error", "cancelled"):
            job_record["status"] = status
            record_path.write_text(json.dumps(job_record, indent=2))
            raise RuntimeError(f"Fine-tuning job {job_id} ended with status: {status}")

        time.sleep(poll_interval)

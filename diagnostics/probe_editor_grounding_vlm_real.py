"""Inspect existing local VLM grounding; never use guessed coordinates as masks."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image

from roop.img_editor.instruction_verifier import InstructionVerifier
from roop.img_editor.prompt_rewriter import get_project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("question")
    args = parser.parse_args()
    photo = Image.open(args.photo).convert("RGB")
    get_project_root()  # Existing Windows llama DLL setup.
    verifier = InstructionVerifier()
    assert verifier._load(), verifier._unavailable_reason
    started = time.perf_counter()
    try:
        response = verifier._llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": verifier._image_data_url(photo)}},
                {"type": "text", "text": args.question},
            ]}], temperature=0.0, max_tokens=256,
        )
    finally:
        verifier.unload()
    metadata = {"photo": args.photo, "question": args.question,
                "seconds": time.perf_counter() - started, "response": response}
    output = Path("diagnostics/out") / f"grounding_vlm_{time.time_ns()}.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output, metadata, flush=True)


if __name__ == "__main__":
    main()

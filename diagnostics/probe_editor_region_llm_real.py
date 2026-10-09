"""Compare actual local language-model region proposals, without rendering."""
import argparse
import json
from pathlib import Path
import time

# This module prepares the installed CUDA DLL search path on Windows.
from roop.img_editor.prompt_rewriter import PromptRewriter
from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("instructions", nargs="+")
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--translate-ambiguous", action="store_true",
                        help="Exercise production translator with configured model; no renderer")
    parser.add_argument("--system-message-file", type=Path,
                        help="Isolated prompt experiment; does not change production")
    args = parser.parse_args()
    measurements = []
    system_message = (args.system_message_file.read_text(encoding="utf-8")
                      if args.system_message_file else None)
    manager = None
    if args.translate_ambiguous:
        from roop.img_editor.img_editor_manager import ImgEditorManager
        manager = ImgEditorManager()
    llm = None if manager else Llama(model_path=args.model, n_gpu_layers=0, n_ctx=8192, verbose=False)
    try:
        for instruction in args.instructions:
            started = time.perf_counter()
            classify = (PromptRewriter._classify_edit_support if args.support_only
                        else PromptRewriter._classify_affected_regions)
            if manager:
                translated = manager._ambiguous_region_translation(instruction)
                proposals = [translated] if translated else []
            elif system_message:
                response = llm.create_completion(
                    f"<|im_start|>system\n{system_message}<|im_end|>\n"
                    f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
                    max_tokens=80, temperature=0.0, echo=False,
                    stop=["<|im_end|>", "\n"], repeat_penalty=1.05,
                )
                query = response["choices"][0]["text"].strip().strip("\"'")
                proposals = [query] if query else []
            else:
                proposals = classify(instruction, llm)
            measurement = {"instruction": instruction, "proposals": proposals,
                           "seconds": time.perf_counter() - started}
            measurements.append(measurement)
            print(measurement, flush=True)
    finally:
        if llm is not None:
            llm.close()
    output = Path(__file__).resolve().parents[1] / "diagnostics/out" / f"region_llm_{time.time_ns()}.json"
    output.write_text(json.dumps({"model": args.model, "support_only": args.support_only,
                                 "system_message": system_message,
                                 "production_translation": args.translate_ambiguous,
                                 "measurements": measurements}, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()

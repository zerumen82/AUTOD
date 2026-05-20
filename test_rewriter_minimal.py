import os, sys
# Add the project root to sys.path
sys.path.append(os.getcwd())

from roop.img_editor.prompt_rewriter import PromptRewriter

def test_rewriter(prompt, context=""):
    print(f"\nTesting prompt: {prompt}")
    print(f"Context: {context}")
    rewriter = PromptRewriter()
    result = rewriter.rewrite(prompt, image_context=context)
    print(f"Result: {result}")

if __name__ == "__main__":
    test_rewriter("ponle gafas de sol", "a man with short hair")
    test_rewriter("make her hair red", "a woman in a garden")
    test_rewriter("cambia el fondo por una playa", "a person standing in a city")
    test_rewriter("ponle un traje de payaso", "a man in a suit")

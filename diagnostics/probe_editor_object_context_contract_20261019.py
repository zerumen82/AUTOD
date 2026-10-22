"""CPU context-scope contracts; only the existing inference doubles are used.

An object may receive more clean reference context than an identically shaped
head. Its write support, protected neighbours and single-pass contract must not
grow. Object labels and instruction words never choose the context policy.
"""
from __future__ import annotations

import copy
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.probe_editor_selection_contract_20261019 import (  # noqa: E402
    assert_local, assert_omitted, execute,
)
from roop.img_editor import element_selection as es  # noqa: E402


def configured_execute(ids, *, context=None, kind=None, prompt="Make the selected element silver", finish=False):
    settings = copy.deepcopy(es.load_settings())
    if context is not None:
        settings["edit"]["object_context_fraction"] = context

    def measured(analysis):
        if kind is not None:
            analysis.regions[3].kind = kind

    with patch.object(es, "load_settings", return_value=settings):
        return execute(ids, analysis_mutator=measured, prompt=prompt, finish=finish)


def reference(run):
    assert len(run.client.generations) == 1, run.message
    return run.client.generations[0]["image"]


def same_mask_larger_object_context_same_write_support():
    settings = es.load_settings()["edit"]
    assert settings["object_context_fraction"] > settings["crop_context_fraction"], (
        "fixture requires object context larger than head context"
    )
    head = configured_execute(["small-object"], kind="head", finish=True)
    obj = configured_execute(["small-object"], kind="object", finish=True)
    assert_local(head, ["small-object"], finish=True)
    assert_local(obj, ["small-object"], finish=True)
    assert np.array_equal(np.asarray(head.analysis.regions[3].mask), np.asarray(obj.analysis.regions[3].mask))
    assert reference(obj).width * reference(obj).height > reference(head).width * reference(head).height, (
        "measured object did not receive the larger context"
    )
    assert np.array_equal(np.asarray(head.mask), np.asarray(obj.mask)), "reference growth expanded write support"
    assert np.array_equal(np.asarray(head.output), np.asarray(obj.output)), (
        "constant adversarial inference/finish changed composition outside the same support"
    )


def head_context_ignores_unselected_objects_and_object_config():
    baseline = configured_execute(["head-right"], context=0.0)
    enlarged = configured_execute(["head-right"], context=100.0)
    # A setting used exclusively by objects cannot affect a head edit.
    invalid_unselected = configured_execute(["head-right"], context=float("nan"))
    for run in (baseline, enlarged, invalid_unselected):
        assert_local(run, ["head-right"])
        assert np.array_equal(np.asarray(reference(run)), np.asarray(reference(baseline)))
        assert np.array_equal(np.asarray(run.mask), np.asarray(baseline.mask))


def mixed_selection_and_words_do_not_expand_write_support():
    ids = ["head-right", "small-object"]
    small = configured_execute(ids, context=0.0)
    expanded = configured_execute(ids)
    assert_local(small, ids)
    assert_local(expanded, ids)
    assert reference(expanded).width * reference(expanded).height > reference(small).width * reference(small).height
    assert np.array_equal(np.asarray(expanded.mask), np.asarray(small.mask))
    for kind, prompt in (("head", "Make this coffee cup gold"), ("object", "Make this head bald")):
        run = configured_execute(["small-object"], kind=kind, prompt=prompt)
        control = configured_execute(["small-object"], kind=kind)
        assert_local(run, ["small-object"])
        assert np.array_equal(np.asarray(reference(run)), np.asarray(reference(control))), (
            "instruction words chose context instead of measured region kind"
        )


def invalid_object_context_abstains_before_inference():
    for value in (-1.0, float("nan"), float("inf")):
        run = configured_execute(["small-object"], context=value)
        assert_omitted(run)
        assert not run.client.loads and not run.client.generations, "invalid context reached inference"


def main():
    cases = [same_mask_larger_object_context_same_write_support,
             head_context_ignores_unselected_objects_and_object_config,
             mixed_selection_and_words_do_not_expand_write_support,
             invalid_object_context_abstains_before_inference]
    failures = []
    for check in cases:
        try:
            check()
            print(f"PASS {check.__name__}", flush=True)
        except Exception as error:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {error}", flush=True)
            traceback.print_exc()
    print(f"Object context scope: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

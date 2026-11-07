"""Per-request Editor engine policy and observed execution, without inference."""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Optional

from roop.img_editor.gen_prompt_config import (
    get_kontext_model_config, get_longcat_model_config,
)


@dataclass(frozen=True)
class ExecutedModel:
    engine: str
    model_file: str


@dataclass
class EditExecution:
    required_model: Optional[str] = None
    automatic: bool = False
    automatic_models: set[str] = field(default_factory=set)
    scope_notes: list[str] = field(default_factory=list)
    # Narrow exception: the measured body/pose route needs the vision engine
    # that follows the requested final state (spec/02). It opts in explicitly
    # through load(..., allow_vision_engine=True); every other substitution is
    # still rejected before any external effect.
    permitted_vision_model: Optional[str] = None
    models: list[ExecutedModel] = field(default_factory=list)
    rejection: str = ""


_execution: ContextVar[Optional[EditExecution]] = ContextVar("editor_execution", default=None)


def begin_execution(selected_engine: str) -> tuple[EditExecution, Token]:
    required_model = None
    automatic = selected_engine == "auto"
    automatic_models = set()
    # These are existing API engine identifiers, not prompt vocabulary.
    if automatic:
        required_model = get_longcat_model_config()["model_file"]
        automatic_models.add(required_model)
    elif selected_engine in ("kontext", "imagine"):
        required_model = get_kontext_model_config()["model_file"]
    report = EditExecution(
        required_model=required_model,
        automatic=automatic,
        automatic_models=automatic_models,
    )
    return report, _execution.set(report)


def end_execution(token: Token) -> None:
    _execution.reset(token)


def current_execution() -> Optional[EditExecution]:
    return _execution.get()


def permit_vision_model(model_file: str) -> None:
    """Opt in the measured body/pose route to its explicit vision engine.

    Only that measured route opts in, and its load declares
    ``allow_vision_engine=True``. A request already rejected cannot be revived
    by this exception.
    """
    report = current_execution()
    if report is not None and not report.rejection:
        report.permitted_vision_model = model_file


def permit_automatic_model(model_file: str) -> None:
    """Allow a model only after Auto selected its measured specialized route."""
    report = current_execution()
    if report is not None and report.automatic and not report.rejection and model_file:
        report.automatic_models.add(model_file)


def allow_model(model_file: str) -> tuple[bool, str]:
    """Reject before external effects; a rejected request cannot resume."""
    report = current_execution()
    if report is None:
        return True, ""
    if report.rejection:
        return False, report.rejection
    if report.automatic:
        if model_file in report.automatic_models:
            return True, ""
        report.rejection = (
            "Auto detuvo la solicitud porque una ruta intentó sustituir "
            "un motor permitido sin una decisión medida."
        )
        print(f"[ImgEditor] {report.rejection} Modelo rechazado: {model_file}", flush=True)
        return False, report.rejection
    if report.required_model and model_file != report.required_model:
        if report.permitted_vision_model == model_file:
            return True, ""
        report.rejection = (
            "Petición detenida: Kontext está seleccionado y esta ruta intentó "
            "utilizar otro motor. No se permite la sustitución automática."
        )
        print(f"[ImgEditor] {report.rejection} Modelo rechazado: {model_file}", flush=True)
        return False, report.rejection
    return True, ""


def record_execution(engine: str, model_file: str) -> None:
    """Record only an image actually returned by a guarded model client."""
    report = current_execution()
    if report is not None:
        observed = ExecutedModel(engine, model_file)
        if observed not in report.models:
            report.models.append(observed)
        print(f"[ImgEditor] Motor ejecutado: {engine}; modelo={model_file}", flush=True)


def record_execution_scope(note: str) -> None:
    report = current_execution()
    scope = str(note or "").strip()
    if report is not None and scope and scope not in report.scope_notes:
        report.scope_notes.append(scope)


def format_execution_status(report: EditExecution, compliance, *, has_instruction: bool) -> str:
    parts = []
    if report.models:
        names = list(dict.fromkeys(model.engine for model in report.models))
        if report.automatic:
            parts.append(f"Auto eligió: {', '.join(names)}")
        else:
            label = "Motor ejecutado" if len(names) == 1 else "Motores ejecutados"
            parts.append(f"{label}: {', '.join(names)}")
    elif has_instruction:
        parts.append(
            "Auto: ningún motor generó una imagen"
            if report.automatic
            else "Motor ejecutado: sin ejecución registrada"
        )
    if has_instruction and report.scope_notes:
        parts.append(f"Alcance: {', '.join(report.scope_notes)}")
    if has_instruction:
        verdict = compliance.get("compliant") if isinstance(compliance, dict) else None
        if verdict is True:
            parts.append("Verificación automática: señal positiva; revisar visualmente")
        elif verdict is False:
            note = str(compliance.get("unmet") or "").strip()
            parts.append("Aviso: verificación automática no confirmada" + (f" ({note})" if note else ""))
        else:
            parts.append("Edición sin verificar; revisar visualmente")
    return " | ".join(parts)

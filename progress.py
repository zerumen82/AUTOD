import base64
import io
import time

import gradio as gr
from pydantic import BaseModel, Field

from modules.shared import opts

import modules.shared as shared
from collections import OrderedDict
import string
import random
from typing import List, Optional

current_task = None
pending_tasks = OrderedDict()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2


def start_task(id_task):
    global current_task

    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)

def create_task_id(task_type):
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
    string.digits, k=N))
    return f"task({task_type}-{res})"

def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()

class PendingTasksResponse(BaseModel):
    size: int = Field(title="Pending task size")
    tasks: List[str] = Field(title="Pending task ids")

class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    active: bool
    queued: int
    completed: int
    progress: float
    eta: Optional[float] = 0.0  # Valor predeterminado para eta
    live_preview: Optional[str] = ""  # Valor predeterminado para live_preview
    id_live_preview: Optional[str] = ""
    textinfo: Optional[str] = ""  # Valor predeterminado para textinfo

def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def get_pending_tasks():
    pending_tasks_ids = list(pending_tasks)
    pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


# En tu función progressapi
def progressapi():
    active = True
    queued = 5
    completed = 3
    progress = 0.75
    eta = None  # Simulando un valor None
    live_preview = None  # Simulando un valor None
    id_live_preview = "some_id"
    textinfo = None  # Simulando un valor None

    return ProgressResponse(
        active=active,
        queued=queued,
        completed=completed,
        progress=progress,
        eta=eta or 0.0,  # Proporciona un valor predeterminado si eta es None
        live_preview=live_preview or "",  # Proporciona un valor predeterminado si live_preview es None
        id_live_preview=id_live_preview,
        textinfo=textinfo or ""  # Proporciona un valor predeterminado si textinfo es None
    )

def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"

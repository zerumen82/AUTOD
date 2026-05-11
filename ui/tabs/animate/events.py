import gradio as gr
import ui.tabs.animate.logic as logic
import ui.tabs.animate.state as state


def wire_animate_events(ui):
    def on_animate_click(img_data, p, model, stabilize, m_bucket, n_frames, a_text, u_tts, lang, r_voice):
        p = (p or "").strip()
        if img_data is None:
            yield None, "Sube una imagen primero"
            return
        img = img_data if not isinstance(img_data, dict) else img_data.get("background")
        if img is None:
            yield None, "Imagen no válida"
            return

        state.is_animating = True
        yield None, "⏳ Analizando imagen y orquestando AI..."

        video, msg = logic.generate_grok_animation(
            img, p, motion=m_bucket, frames=n_frames, fps=16,
            model=model, stabilize=stabilize,
            audio_text=a_text, use_tts=u_tts, language=lang, ref_voice=r_voice
        )

        state.is_animating = False
        yield video, msg

    ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[
            ui["input_img"], ui["prompt"], ui["model_choice"], ui["face_stabilize"],
            ui["motion_bucket"], ui["num_frames"],
            ui["audio_text"], ui["use_tts"], ui["language"], ui["ref_voice"]
        ],
        outputs=[ui["video_output"], ui["progress_html"]]
    )

    # Conectar botones de acciones rápidas
    ui["btn_smile"].click(fn=lambda: logic.get_expression_prompt("smile"), outputs=[ui["prompt"]])
    ui["btn_wink"].click(fn=lambda: logic.get_expression_prompt("wink"), outputs=[ui["prompt"]])
    ui["btn_angry"].click(fn=lambda: logic.get_expression_prompt("angry"), outputs=[ui["prompt"]])
    ui["btn_wind"].click(fn=lambda: logic.get_expression_prompt("wind"), outputs=[ui["prompt"]])

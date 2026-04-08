from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr


def build_clinical_trial_ui(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title,
    quick_start_md,
) -> "gr.Blocks":
    """Build a lightweight custom UI for interactive protocol review."""
    _ = action_fields
    _ = metadata
    _ = is_chat_env
    _ = title
    _ = quick_start_md

    import gradio as gr

    with gr.Blocks(title="Clinical Trial Review Demo") as demo:
        gr.Markdown(
            """
# Clinical Trial Protocol Review Demo
Reset an episode, flag structured violations, submit a report, and monitor reward updates live.
"""
        )

        state = gr.State({"observation": None})

        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task",
            )
            seed_input = gr.Number(value=42, precision=0, label="Seed")
            reset_button = gr.Button("Reset Episode", variant="primary")

        with gr.Row():
            protocol_display = gr.Textbox(
                label="Protocol Text",
                lines=24,
                interactive=False,
                scale=2,
            )
            with gr.Column(scale=1):
                section_input = gr.Textbox(label="Section ID", placeholder="section_03_criteria")
                rule_input = gr.Dropdown(
                    choices=[f"RULE_{index:03d}" for index in range(1, 13)],
                    value="RULE_001",
                    label="Rule ID",
                )
                severity_input = gr.Radio(
                    choices=["critical", "major", "minor"],
                    value="major",
                    label="Severity",
                )
                explanation_input = gr.Textbox(label="Explanation", lines=3)
                correction_input = gr.Textbox(label="Suggested Correction", lines=3)
                flag_button = gr.Button("Flag Violation")

        report_input = gr.Textbox(
            label="Report Text",
            lines=5,
            placeholder="Submit a final report for medium or hard tasks.",
        )
        submit_button = gr.Button("Submit Report", variant="secondary")

        with gr.Row():
            reward_display = gr.Number(label="Cumulative Reward", value=0.0, interactive=False)
            step_display = gr.Number(label="Current Step", value=0, interactive=False)
            found_display = gr.Number(label="Violations Found", value=0, interactive=False)
            done_display = gr.Textbox(label="Episode Done", value="No", interactive=False)

        feedback_display = gr.Textbox(label="Reviewer Feedback", lines=6, interactive=False)
        status_display = gr.Textbox(label="Status", interactive=False)

        def _done_label(done: bool) -> str:
            return "Yes" if done else "No"

        def _observation_fields(obs: dict[str, object]) -> tuple[str, float, int, int, str, str]:
            return (
                str(obs.get("protocol_text", "")),
                float(obs.get("cumulative_reward", 0.0)),
                int(obs.get("step", 0)),
                int(obs.get("violations_found_so_far", 0)),
                _done_label(bool(obs.get("episode_done", False))),
                str(obs.get("reviewer_feedback", "")),
            )

        def do_reset(task: str, seed: float, stored: dict[str, object]):
            obs = web_manager.reset_environment({"task": task, "seed": int(seed)})
            stored["observation"] = obs
            protocol_text, reward, step, found, done_text, feedback = _observation_fields(obs)
            return (
                protocol_text,
                reward,
                step,
                found,
                done_text,
                feedback,
                "Episode reset successfully.",
                stored,
            )

        def do_flag(
            section_id: str,
            rule_id: str,
            severity: str,
            explanation: str,
            correction: str,
            stored: dict[str, object],
        ):
            if not stored.get("observation"):
                return (0.0, 0, 0, "No", "", "Reset the environment first.", stored)

            action = {
                "action_type": "flag_violation",
                "violation_flags": [
                    {
                        "section_id": section_id.strip(),
                        "rule_id": rule_id,
                        "severity": severity,
                        "explanation": explanation,
                        "suggested_correction": correction,
                    }
                ],
                "report_text": "",
                "explanation": explanation,
            }
            result = web_manager.step_environment(action)
            obs = dict(result.get("observation", {}))
            stored["observation"] = obs
            _, reward, step, found, done_text, feedback = _observation_fields(obs)
            status = f"Step reward: {float(result.get('reward', 0.0)):+.4f}"
            return (reward, step, found, done_text, feedback, status, stored)

        def do_submit(report_text: str, stored: dict[str, object]):
            if not stored.get("observation"):
                return (0.0, 0, 0, "No", "", "Reset the environment first.", stored)

            action = {
                "action_type": "submit_report",
                "violation_flags": [],
                "report_text": report_text or "Report submitted from the Gradio demo.",
                "explanation": "Submitting findings report from the custom tab.",
            }
            result = web_manager.step_environment(action)
            obs = dict(result.get("observation", {}))
            stored["observation"] = obs
            _, reward, step, found, done_text, feedback = _observation_fields(obs)
            status = f"Step reward: {float(result.get('reward', 0.0)):+.4f}"
            return (reward, step, found, done_text, feedback, status, stored)

        reset_button.click(
            do_reset,
            inputs=[task_dropdown, seed_input, state],
            outputs=[
                protocol_display,
                reward_display,
                step_display,
                found_display,
                done_display,
                feedback_display,
                status_display,
                state,
            ],
        )
        flag_button.click(
            do_flag,
            inputs=[
                section_input,
                rule_input,
                severity_input,
                explanation_input,
                correction_input,
                state,
            ],
            outputs=[
                reward_display,
                step_display,
                found_display,
                done_display,
                feedback_display,
                status_display,
                state,
            ],
        )
        submit_button.click(
            do_submit,
            inputs=[report_input, state],
            outputs=[
                reward_display,
                step_display,
                found_display,
                done_display,
                feedback_display,
                status_display,
                state,
            ],
        )

    return demo

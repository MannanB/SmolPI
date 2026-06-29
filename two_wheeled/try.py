import base64
import os
import threading
import time
import tkinter as tk
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np
import torch

from two_wheeled.config import Config
from environment import MujocoEnvironment, get_vision_input_shape
from model.smolpi import SmolPI, SmolPIConfig


CHECKPOINT_PATH = Path("smolpi_sft_final.pth")
DISPLAY_MAX_WIDTH = 640
DISPLAY_MAX_HEIGHT = 720


def make_sft_config() -> Config:
    return Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=1, precision=torch.float16),
        cam_sim_output_path="vids/sft_eval_sim.mp4",
        cam_omni_output_path="vids/sft_eval_omni.mp4",
    )


def encode_frame_for_tk(rgb_frame: np.ndarray) -> str:
    height, width = rgb_frame.shape[:2]
    scale = min(1.0, DISPLAY_MAX_WIDTH / width, DISPLAY_MAX_HEIGHT / height)
    if scale < 1.0:
        rgb_frame = cv2.resize(
            rgb_frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr_frame)
    if not ok:
        raise RuntimeError("failed to encode camera frame")
    return base64.b64encode(encoded).decode("ascii")


class LiveSmolPIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SmolPI Live Prompt")

        os.makedirs("vids", exist_ok=True)
        self.cfg = make_sft_config()

        self.lock = threading.Lock()
        self.prompt = ""
        self.active = False
        self.reset_requested = False
        self.closed = False
        self.latest_front_frame_data: str | None = None
        self.latest_omni_frame_data: str | None = None
        self.status_message = "Loading model and simulation..."
        self.front_photo: tk.PhotoImage | None = None
        self.omni_photo: tk.PhotoImage | None = None

        self.build_ui()

        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)
        self.sim_thread.start()

        self.root.after(33, self.refresh_ui)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def load_policy(self) -> SmolPI:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"missing checkpoint: {CHECKPOINT_PATH}")

        policy = SmolPI(self.cfg.smolpi).to(self.cfg.device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.cfg.device)
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            checkpoint = checkpoint["policy_state_dict"]
        policy.load_state_dict(checkpoint)
        policy.eval()
        return policy

    def build_ui(self) -> None:
        camera_frame = tk.Frame(self.root, bg="black")
        camera_frame.pack(fill=tk.BOTH, expand=True)
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.columnconfigure(1, weight=1)
        camera_frame.rowconfigure(0, weight=1)

        self.front_frame_label = tk.Label(camera_frame, bg="black")
        self.front_frame_label.grid(row=0, column=0, sticky="nsew")

        self.omni_frame_label = tk.Label(camera_frame, bg="black")
        self.omni_frame_label.grid(row=0, column=1, sticky="nsew")

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.prompt_var = tk.StringVar()
        prompt_entry = tk.Entry(controls, textvariable=self.prompt_var)
        prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        prompt_entry.bind("<Return>", lambda _event: self.send_prompt())
        prompt_entry.focus_set()

        tk.Button(controls, text="Send", command=self.send_prompt).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(controls, text="Stop", command=self.stop_policy).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(controls, text="Reset", command=self.reset_sim).pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value=self.status_message)
        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=8, pady=(0, 8))

    def set_status(self, message: str) -> None:
        with self.lock:
            self.status_message = message

    def send_prompt(self) -> None:
        prompt = self.prompt_var.get().strip()
        if not prompt:
            self.set_status("Enter a prompt before sending.")
            return

        with self.lock:
            self.prompt = prompt
            self.active = True
            self.status_message = f"Running prompt: {prompt}"

    def stop_policy(self) -> None:
        with self.lock:
            self.active = False
            self.status_message = "Stopped. Live camera feed is still running."

    def reset_sim(self) -> None:
        with self.lock:
            self.reset_requested = True
            self.status_message = "Reset requested."

    def refresh_ui(self) -> None:
        with self.lock:
            front_frame_data = self.latest_front_frame_data
            omni_frame_data = self.latest_omni_frame_data
            status_message = self.status_message
            closed = self.closed

        if front_frame_data is not None:
            self.front_photo = tk.PhotoImage(data=front_frame_data)
            self.front_frame_label.configure(image=self.front_photo)

        if omni_frame_data is not None:
            self.omni_photo = tk.PhotoImage(data=omni_frame_data)
            self.omni_frame_label.configure(image=self.omni_photo)

        self.status_var.set(status_message)
        if not closed:
            self.root.after(33, self.refresh_ui)

    def sim_loop(self) -> None:
        env: MujocoEnvironment | None = None
        torque_cmd = np.zeros(2, dtype=np.float32)
        control_countdown = 0
        sim_step = 0

        try:
            policy = self.load_policy()
            env = MujocoEnvironment(self.cfg, *get_vision_input_shape(policy))
            data = env.datas[0]

            timestep = float(env.model.opt.timestep)
            steps_per_control = max(1, int(1 / self.cfg.control_freq_hz / timestep))
            steps_per_omni_frame = max(1, int(1 / self.cfg.cam_omni_fps / timestep))
            next_step_at = time.perf_counter()
            self.set_status("Ready. Enter a prompt and press Send.")

            with torch.no_grad():
                while True:
                    with self.lock:
                        if self.closed:
                            break
                        active = self.active
                        prompt = self.prompt
                        reset_requested = self.reset_requested
                        if reset_requested:
                            self.reset_requested = False

                    if reset_requested:
                        env.reset_episode()
                        data = env.datas[0]
                        torque_cmd[:] = 0.0
                        control_countdown = 0
                        sim_step = 0
                        self.set_status("Simulation reset.")

                    if active and prompt:
                        if control_countdown <= 0:
                            try:
                                observation = env.make_observation(prompt, self.cfg.device, env_idx=0)
                                if (
                                    self.cfg.device.type != "cpu"
                                    and self.cfg.smolpi.precision in (torch.float16, torch.bfloat16)
                                ):
                                    amp_ctx = torch.autocast(
                                        device_type=self.cfg.device.type,
                                        dtype=self.cfg.smolpi.precision,
                                    )
                                else:
                                    amp_ctx = nullcontext()

                                with amp_ctx:
                                    action = policy.sample_actions(
                                        self.cfg.device,
                                        observation,
                                        num_steps=self.cfg.flow_steps,
                                    )

                                torque_cmd = action[0, 0].to(dtype=torch.float32).cpu().numpy()
                                torque_cmd = np.clip(torque_cmd, -5.0, 5.0)
                            except Exception as exc:
                                torque_cmd[:] = 0.0
                                with self.lock:
                                    self.active = False
                                    self.status_message = f"Policy error: {exc}"

                            control_countdown = steps_per_control
                        control_countdown -= 1
                    else:
                        torque_cmd[:] = 0.0
                        control_countdown = 0

                    data.ctrl[0] = float(torque_cmd[0])
                    data.ctrl[1] = float(torque_cmd[1])
                    mujoco.mj_step(env.model, data)

                    if sim_step % steps_per_omni_frame == 0:
                        try:
                            front_rgb, _ = env.render_camera(
                                env.cam_renderer,
                                "front_cam",
                                env_idx=0,
                            )
                            rgb_frame, _ = env.render_camera(
                                env.omni_cam_renderer,
                                "omniscient_cam",
                                env_idx=0,
                            )
                            front_frame_data = encode_frame_for_tk(front_rgb)
                            omni_frame_data = encode_frame_for_tk(rgb_frame)
                            with self.lock:
                                self.latest_front_frame_data = front_frame_data
                                self.latest_omni_frame_data = omni_frame_data
                        except Exception as exc:
                            with self.lock:
                                self.status_message = f"Render error: {exc}"

                    sim_step += 1
                    next_step_at += timestep
                    sleep_for = next_step_at - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_step_at = time.perf_counter()
        except Exception as exc:
            with self.lock:
                self.active = False
                self.status_message = f"Startup error: {exc}"
        finally:
            if env is not None:
                env.close()

    def close(self) -> None:
        with self.lock:
            self.closed = True
        self.sim_thread.join(timeout=5.0)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = LiveSmolPIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

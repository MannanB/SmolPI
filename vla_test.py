import os
from contextlib import nullcontext

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np
import torch
from transformers import AutoTokenizer

from model.smolpi import Observation, SmolPI, SmolPIConfig


XML = ""
with open("platform_world.xml", "r", encoding="utf-8") as f:
    XML = f.read()


PROMPT = "Drive forward toward interesting colored platforms."


def make_writer(path: str, width: int, height: int, fps: int) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str):
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


def _vision_input_shape(policy: SmolPI) -> tuple[int, int, int]:
    vision_cfg = policy.smolvlm_with_expert.smolvlm.config.vision_config
    channels = int(getattr(vision_cfg, "num_channels", 3))
    image_size = getattr(vision_cfg, "image_size", 384)
    if isinstance(image_size, (tuple, list)):
        h, w = int(image_size[0]), int(image_size[1])
    else:
        h = w = int(image_size)
    return channels, h, w


def _frame_to_tensor(rgb_frame: np.ndarray, height: int, width: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = torch.from_numpy(resized).to(device=device, dtype=torch.float32) / 255.0
    frame = frame.permute(2, 0, 1).contiguous()
    return frame.unsqueeze(0)


def _make_observation(
    policy: SmolPI,
    tokenizer: AutoTokenizer,
    prompt: str,
    rgb_frame: np.ndarray,
    wheel_speeds: np.ndarray,
    device: torch.device,
) -> Observation:
    _, h, w = _vision_input_shape(policy)
    image = _frame_to_tensor(rgb_frame, h, w, device)

    tokens = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = tokens["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = tokens["attention_mask"].to(device=device, dtype=torch.bool)

    state = torch.as_tensor(wheel_speeds, device=device, dtype=torch.float32).unsqueeze(0)

    return Observation(
        images={"front": image},
        image_masks={"front": torch.ones(1, device=device, dtype=torch.bool)},
        tokenized_prompt=input_ids,
        tokenized_prompt_mask=attention_mask,
        state=state,
    )


def _get_wheel_speed_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")

    left_dof = model.jnt_dofadr[left_joint_id]
    right_dof = model.jnt_dofadr[right_joint_id]

    return np.array([data.qvel[left_dof], data.qvel[right_dof]], dtype=np.float32)


def main() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    device_name = os.environ.get("SMOLPI_TEST_DEVICE", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SMOLPI_TEST_DEVICE=cuda was requested but CUDA is not available.")
    device = torch.device(device_name)

    cfg = SmolPIConfig(action_dim=2, action_horizon=5)
    if device.type == "cpu" and cfg.precision in (torch.float16, torch.bfloat16):
        cfg.precision = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.smolvlm_id)
    policy = SmolPI(cfg).to(device)
    policy.eval()

    width, height = 640, 480
    omni_width, omni_height = 1280, 960
    fps = 30
    omni_fps = 60
    duration_sec = 12
    control_hz = 10.0

    front_renderer = mujoco.Renderer(model, height=height, width=width)
    omni_renderer = mujoco.Renderer(model, height=omni_height, width=omni_width)

    front_video = make_writer("front_cam_vla_rollout.mp4", width, height, fps)
    omni_video = make_writer("omniscient_vla_rollout.mp4", omni_width, omni_height, omni_fps)

    sim_steps = int(duration_sec / model.opt.timestep)
    steps_per_frame = int(1 / fps / model.opt.timestep)
    steps_per_frame_omni = int(1 / omni_fps / model.opt.timestep)
    steps_per_control = max(1, int(1 / control_hz / model.opt.timestep))

    front_frames = []
    omni_frames = []

    torque_cmd = np.zeros(2, dtype=np.float32)

    with torch.no_grad():
        for step in range(sim_steps):
            if step % steps_per_control == 0:
                front_rgb, _ = render_camera(front_renderer, data, "front_cam")
                wheel_speeds = _get_wheel_speed_state(model, data)
                observation = _make_observation(policy, tokenizer, PROMPT, front_rgb, wheel_speeds, device)

                if cfg.precision in (torch.float16, torch.bfloat16):
                    amp_ctx = torch.autocast(device_type=device.type, dtype=cfg.precision)
                else:
                    amp_ctx = nullcontext()
                with amp_ctx:
                    action = policy.sample_actions(device, observation, num_steps=10)

                torque_cmd = action[0, 0].to(dtype=torch.float32).cpu().numpy()
                torque_cmd = np.clip(torque_cmd, -5.0, 5.0)

            data.ctrl[0] = float(torque_cmd[0])
            data.ctrl[1] = float(torque_cmd[1])

            mujoco.mj_step(model, data)

            if step % steps_per_frame == 0:
                front_rgb, front_bgr = render_camera(front_renderer, data, "front_cam")
                front_frames.append(front_rgb.copy())
                front_video.write(front_bgr)

            if step % steps_per_frame_omni == 0:
                omni_rgb, omni_bgr = render_camera(omni_renderer, data, "omniscient_cam")
                omni_frames.append(omni_rgb.copy())
                omni_video.write(omni_bgr)

    front_video.release()
    omni_video.release()
    front_renderer.close()
    omni_renderer.close()

    print(f"Saved front_cam_vla_rollout.mp4 with {len(front_frames)} frames.")
    print(f"Saved omniscient_vla_rollout.mp4 with {len(omni_frames)} frames.")
    print("Final bot position xyz:", data.qpos[:3])


if __name__ == "__main__":
    main()

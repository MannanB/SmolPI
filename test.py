# minimal_mujoco_camera_bot.py

import os
os.environ.setdefault("MUJOCO_GL", "glfw") 

import cv2
import mujoco
import numpy as np


XML = ""
with open("platform_world.xml", "r") as f:
    XML = f.read()


def make_writer(path, width, height, fps):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def render_camera(renderer, data, camera_name):
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    width, height = 640, 480
    omni_width, omni_height = 1280, 960
    fps = 30
    omni_fps = 60
    duration_sec = 12

    front_renderer = mujoco.Renderer(model, height=height, width=width)
    omni_renderer = mujoco.Renderer(model, height=omni_height, width=omni_width)

    front_video = make_writer("front_cam_rollout.mp4", width, height, fps)
    omni_video = make_writer("omniscient_rollout.mp4", omni_width, omni_height, omni_fps)  # Higher res + fps for omniscient view.

    sim_steps = int(duration_sec / model.opt.timestep)
    steps_per_frame = int(1 / fps / model.opt.timestep)
    steps_per_frame_omni = int(1 / omni_fps / model.opt.timestep) 

    front_frames = []
    omni_frames = []

    for step in range(sim_steps):
        # Same torque on both wheels => forward.
        # Flip signs if your wheel orientation makes it drive backward.
        data.ctrl[0] = 1.1
        data.ctrl[1] = -0.6

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

    print(f"Saved front_cam_rollout.mp4 with {len(front_frames)} frames.")
    print(f"Saved omniscient_rollout.mp4 with {len(omni_frames)} frames.")
    print("Final bot position xyz:", data.qpos[:3])


if __name__ == "__main__":
    main()
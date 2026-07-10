# SmolPI

SmolPI is an attempt to replicate the work of Physical Intelligence's [PI0 model](https://github.com/Physical-Intelligence/openpi). It uses the MuJoCo physics engine to simulate an environment for a robot to manipulate. The robot is equppied with a single camera, which is passed into the vision model and cross-attended with a flow-matching model to project future actions.

 The vision-language-action model (VLA) is based on the SmolVLM/SmolLM backbone. Similar to openpi, which used paligemma and the gemma architecutre, smolpi uses SmolVLM-256m for the vision model and SmolLM2-135m for the action expert. There are two environments: a simple two wheeled robot driving to different platforms, and a robotic arm trained on Berkeley's Bridge dataset


<video src="https://github.com/MannanB/SmolPI/raw/refs/heads/main/smolpi_bridge_wx250s_pick_red_box.mp4" width="320" height="240" controls></video>

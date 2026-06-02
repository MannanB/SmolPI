# SmolPI

SmolPI is an attempt to replicate the work of Physical Intelligence's [PI0 model](https://github.com/Physical-Intelligence/openpi). It uses the MuJoCo physics engine to simulate an environment for a robot to manipulate. The robot is equppied with a single camera, which is passed into the vision model and cross-attended with a flow-matching model to project future actions.

Currently, there is a single environement with 4 colored platform and a simple two-wheeled robot. The vision-language-action model (VLA) is based on the SmolVLM/SmolLM backbone. Similar to openpi, which used paligemma and the gemma architecutre, smolpi uses SmolVLM-256m for the vision model and SmolLM2-135m for the action expert.

WIP
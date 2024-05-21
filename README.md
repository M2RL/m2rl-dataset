
<!-- [**Paper**](./documents/m2rl.pdf) -->
[**Website**](https://github.com/M2RL/m2rl-dataset/)
| [**Code**](https://github.com/M2RL/m2rl-dataset/tree/main/Code/)
| [**Dataset Sample**](https://drive.google.com/drive/folders/1MGSBQaMcJojIKK2IcFCmJcbljTkMxD46?usp=sharing)
| [**Full Dataset**](https://drive.google.com/drive/folders/1J2XGjoltIc4Ewg_knFv9vL5G1qZzdmdG?usp=sharing)


## M2RL: A Multimodal Multi-Interface Dataset for Robot Learning from Human Demonstrations

![M2RL Modalities](./images/multi_modality.png)

Imitation learning, inspired by how humans learn complex skills through observation and demonstration, is a promising approach for teaching robots to perform manipulation tasks. However, most existing imitation learning datasets focus on a single interface or limited modality when capturing human demonstrations. This fails to fully capture the multimodal nature of how humans actually learn. 

To address this gap, we introduce the M2RL dataset - a multimodal and multi-interface dataset collected from non-expert users across diverse manipulation tasks using three distinct teleoperation interfaces. Our goal with M2RL is to reduce interface bias, enhance dataset diversity, and enable the development of more robust and generalizable imitation learning algorithms.

### M2RL Dataset

The M2RL dataset offers several unique properties:

- **Multimodality:** It includes RGB+D data from three camera views (robot's wrist and two exocentric views), ego-view and gaze data from the human teleoperator's perspective, and the robot's proprioception data. 

- **Task Diversity:** It covers 8 diverse manipulation tasks spanning 4 distinct categories - translational motion, rotational motion, stationary engagement, and deforming engagement tasks. This enables evaluating different interfaces across a range of manipulation requirements.

<figure>
<img src="./images/tasks.jpg" width="100%" height="100%" align="center">

<figcaption style="display: block; margin: auto; text-align: center; font-size:12;"> The eight manipulation tasks in the M2RL dataset</figcaption>
</figure>


- **Skill Diversity:** Based on the trajectory patterns, M2RL encompasses 6 unique manipulation skills. This enhances the skill diversity captured in the dataset.

- **Scale:** M2RL contains 360 robot trajectories (~1 million image frames), with 45 trajectories per task distributed evenly across 3 teleoperation interfaces.

#### Dataset Structure:
The M2RL dataset is available on [Google Drive](https://drive.google.com/drive/folders/1J2XGjoltIc4Ewg_knFv9vL5G1qZzdmdG?usp=sharing)

```
task_1
	|- interface_1
		|-episode_1
			|- kinect1_color
			|- kinect1_depth
			|- kinect2_color
			|- kinect1_depth
			|- rs_color
			|- rs_depth
			|- robot_data.csv
			|- gaze_data	
		|-episode_2
		 ...

	|- interface_2
		...
	|- interface_3
		...
task_2
	...
...
...
task_8
```

### Experiments and Results

To validate M2RL, we evaluated state-of-the-art imitation learning algorithms, namely Behavior Cloning (BC) and Diffusion Policy (DP), on two benchmarks:

**1. Multimodal Interface Benchmark:** We compared the algorithms' performance when trained on demos from a single interface (gamepad) vs all three interfaces. Results showed that both BC and DP (especially DP) achieved lower prediction errors when trained on multimodal interface data. This underscores the importance of collecting demos from diverse interfaces to improve policy robustness.

**2. Multimodal Data Benchmark:** We compared the algorithms' performance when trained on a single RGB+D stream (wrist camera) vs all three streams. DP's performance improved with more camera streams for most tasks, while BC's errors increased. This suggests multimodal data can improve task performance if algorithms can effectively extract relevant cross-modal representations (e.g. using an attention or denoising mechanism like in DP).

M2RL dataset takes a step towards aligning robot imitation learning setups with how humans learn - leveraging multimodal cues and diverse experiences. Our experiments demonstrate the benefits of learning from diverse interfaces and multimodal data to improve manipulation task performance. We hope M2RL enables further research into effective robot learning from human demonstrations.


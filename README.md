[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

This is my submission for Project 2 of Udacity's Deep Reinforcement Learning Nanodegree, Continuous Control.

# Environment Details

The environment is the Unity ML Agents [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher) environment:

![Trained Agent][image1]

In this environment, double-jointed arms can move their tips (blue points) to target locations (green blobs).

## Reward Structure
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

## State and Action Spaces
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1 (this is what makes this a continuous control problem).

## Completion Criteria
The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

# Getting Started

This section will provide instructions on how to setup the repository code. It is tested in a Linux environment.

1. Run the following commands to download and extract the Unity ML Agents environment:
```bash
cd ./p2_continuous-control
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
unzip Reacher_Linux.zip
rm Reacher_Linux.zip
cd ..
```

2. Create (and activate) a new environment with Python 3.6.
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
	
3. Install the python dependencies into the actiavted `conda` environment:
```bash
cd ./python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


# Instructions

## Running the Training Code
To train the agent, make sure the `conda` environment is activated (if it isn't, run `source activate drlnd`), and that you are in the root of the repository. Then:

1. Navigate into the `p2_continuous-control` folder with: `cd ./p2_continuous-control` 
2. Run the training script: `python main.py`

If the environment gets solved, the model weights will get saved in `p2_continuous-control/checkpoint.pth`, and you will see a simulation of the trained agent.

## Report
The details of the successfully trained agent and the learning algorithm can be found in `Report.ipynb`.

## Watch Trained Agent
To watch the trained agent:

1. Run `jupyter notebook`
2. Run the `p2_continuous-control/Watch Trained Agent.ipynb` Jupyter notebook.
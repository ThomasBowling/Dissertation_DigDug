# AI DigDug using Reinforced Learning
## About
These Python files in conjunction with M J Murray's MAME RL Algorithm Training Toolkit form together to create a Dueling ANN that plays the Namco arcade game DigDug which has the capability to exceed an average human highscore. This AI uses what is considered the best Loss function(Huber Loss) and optimizer(Adam) to train the ANN in batches using experience replay(Use of buffers).
## Requirements:
- Python 3.6 or greater
- MAMEToolKit
- Ubuntu 16.04 or later(Ubuntu 20.04 if possible)
## Recommendation:
This AI was created on Python 3.7 through Spyder 4.0.1(IDE) on Anaconda 1.9.12. If you have any dependency issues, try using these versions.

You should familiarise yourself with the MAMEToolKit at: https://github.com/M-J-Murray/MAMEToolkit/blob/master/README.md

CUDA is strongly recommended if you have a compatible GPU, this will massively speed up training, the guide to install it can be found here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

**DISCLAIMER: I am unable to provide you with the DigDug ROM. It is the users own legal responsibility to acquire a game ROM for emulation. This code should only be used for non-commercial research purposes.**

There are some free ROMs available at: https://www.mamedev.org/roms/

## Installation
You can use `pip` to install both required libraries via the console:
```bash
pip install MAMEToolkit
```

```bash
pip install --upgrade tensorflow
```

## Setup DigDugAI_V2
Once all the python files are the same folder, create a roms folder that contains the DigDug ROM so that this path is correct:
```python
roms_path = "roms/"
```

Then alter the path of where the test game videos and TensorBoard files will be stored if needed:
```python
TENSORBOARD_DIR = 'logs/tensorboard/'

out = cv2.VideoWriter('logs/RandomDigDug.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, sizeFrame, True)
```

Finally adjust the training variables to suit how you want the ANN to run:
```python
min_epsilon = 0.1
max_epsilon = 0.99 
decay = 0.001 # Epsilon Decay
gamma = 0.99
alpha = 0.0001 # Learning Rate
BATCH_SIZE = 32
BUFF_MAX = 4000 # Adjust this value depending on how much memory is avaliable
```
From my experience BUFF_MAX of 4000 took around 7.4 GiB of memory

## Emulator
When setting up the emulator you can turn off render by setting render to false, as well as adjust the frame ratio to whatever you would like. Currently frame ratio is at 12, which means 1 in 12 frames are sent through the toolkit as 60 frames per second slows down training.
```python
self.emu = Emulator(env_id, roms_path, "digdug", setup_memory_addresses(), frame_ratio=frame_ratio, render=render)
```

## Running the AI
Simply running the DigDugAI_V2.py file should start MAME and begin the training. MAME needs a short about of time to run before the games will actually start, it does this by running DigDug at full speed until it staggers slightly, then begins with the random game. During this time DigDug will just play a showcase of the game, this isn't the AI playing it yet, rather the idle screen.

The Console should then begin to print out game, score and reward. If you use the same reward system as presented the reward should be the score divided by ten then subtracted by 3, eg Score: 660, Reward: 63.0.

## TensorBoard
To view the graphs that are being written to TensorBoard, run the following command in the BASH Console:
```bash
tensorboard --logdir (Insert Directory Path to Logs folder here)
```
It will then output a Localhost that you can copy into a browser. This Localhost will stay up until the console containing the above command is closed or forcely quit with CTRL + C 

## Recovering the model / Saving Training
Either if the program crashes or you need to halt training temporarily you can run this line in the console to save the agent:
```python
agent.save('Insert directory path')
```
Once you want to resume training you can adjust the follow variables in DigDugAI_V2 as well as commenting out the random game
```python
agent_steps = "(Check TensorBoard to find step and insert here (Doesn't have to be exact)"
game = "(Check Console for last completed game and insert here)"
agent = tf.keras.model.load_model('Insert directory path to agent')
```

## Other Notes
The 534 number from Steps.py was figured out by calculating how many frames there were during the starting game animation and waiting until just before the game starts so that an action could be carried out in the inital step without consequences because the player is not yet in control of DigDug.
```python
{"wait": int(534/frame_ratio), "actions": [Actions.START]}]     
```

```python
# First action doesn't matter 
if inital_step == True:
  action = np.random.randint(0, no_actions)
  state, game_done, reward, round_val = env.step(action)
  agent_steps += 1
  inital_step = False     
```

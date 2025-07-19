## Installation & Run R1

### 1. Download Git Repository
```bash
cd ~
git clone https://github.com/g30133/r2.git
```

### 2. Environment Isolation
```bash
conda create -n r2env python=3.10
conda activate r2env
conda install --channel=conda-forge openjdk=17
```

### 3. Pysimmis Installation and Launch R1
```bash
cd ~/r2/pysimmis
pip install -r requirements.txt
pip install numpy opencv-python pyaudio six
export OPENAI_API_KEY="..."
python r1.py
```

## Running STEVE-1 in PLAICraft Environment
### Initial Setup
Follow steps in PLAICraft repo to launch an EC2 instance, use the AMI ```r2 v5```.

After connecting to your instance via SSH, run the following commands in the terminal. This will prepare the environment and open the two graphical terminal windows (xterm) needed for the next steps. These xterm windows will appear on the virtual desktop of your instance.

```bash
sudo pkill -f obs
export DISPLAY=:0
cd ~/steve-1
xterm -e "sudo su guest -s /bin/bash" &
xterm -e "sudo su guest -s /bin/bash" &
```

You will now have two new terminal windows open on your instance's desktop. We will refer to these as Terminal A and Terminal B.

### Launch MultiMC (Terminal A)
In the first xterm window (Terminal A), launch MultiMC.

```bash
/opt/multimc/run.sh
```
In the MultiMC window that opens:
- If prompted to select a Java version, choose Java 17.
- Go to the View Mods tab on the right, and verify that the Steve1 mod is enabled.
- Important: Do not launch Minecraft yet. You will do this only after the agent is running in the next step.

### Run the STEVE-1 Agent (Terminal B)
In the second xterm window (Terminal B), you will start the STEVE-1 agent script:

```bash
. ~/.profile
. run_agent/run_steve1.sh
```
Be patient. The first time you run the agent, it may take several minutes to initialize.

Wait for the message ```init_frame_control DONE``` to appear in the terminal. This indicates that the agent is ready and waiting for the game to start.

### Connect Agent and Launch Game
Launch Minecraft 1.19.4 from MultiMC and create or load into a single-player world.

STEVE-1 should automatically take control of your character once you are in game. You can also connect to any multiplayer server that is on Minecraft version ```1.19.4```.

### Giving Commands to STEVE-1
To give commands to the agent while it's running.

In sshed terminal, run the prompt script:

```bash
sudo su guest -s /bin/bash
cd /home/ubuntu/steve-1

# Run the interactive prompt script
python3 prompt.py
```
This will start an interactive prompt where you can type natural language commands for STEVE-1 to execute in-game (e.g., chop down a tree, collect dirt, craft wooden pickaxe).

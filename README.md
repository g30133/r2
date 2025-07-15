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

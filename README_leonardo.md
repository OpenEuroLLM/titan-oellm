# Setup of TorchTitan on Leonardo 

Setup instructions for the Torch-Titan framework on Leonardo. We start by cloning the repository:

### 1. Clone Repository
Verify that the cluster has an SSH key. If not, generate a new one and add it to GitHub. After then, 

```bash
ssh -T git@github.com

git clone git@github.com:OpenEuroLLM/titan-oellm.git
cd titan-oellm

# Load and verify TorchTitan submodule
git submodule update --init --recursive
cd torchtitan && git describe --tags && cd ..
# Should output: v0.2.1
```

### 2. Build container
On Leonardo, the apptainer command is not a default one, hence we need to install it and construct a suitable alias. 
Select a directory to store install files. In the following example, it will be $HOME/.myroot
Install and create the alias: edit the bashrc config

```bash
nano ~/.bashrc # or whatever editor you prefer
```

by pasting the following lines 

```bash
export PATH=$HOME/.myroot/bin:$PATH
alias finstall-apptainer="mkdir -p $HOME/.myroot/{opt,bin} && curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - $HOME/.myroot/opt/apptainer && ln -s $HOME/.myroot/opt/apptainer/bin/apptainer $HOME/.myroot/bin/apptainer"
```
at the bottom of the bashrc config file. Then, reload the config:

```bash
source ~/.bashrc
```

To get the apptainer command, do: 

```bash
finstall-apptainer
```

Sanity check:

```bash
which apptainer
# Should output ~/.myroot/bin/apptainer

apptainer version
# Should output 1.4.5-2.el8
```

We then proceed by building the container. To avoid memory issues, start an interactive shell on a compute node with suitable resources:

```bash
srun --time 04:00:00 --mem 30G --pty /bin/bash
```
and inside the shell select /scratch_local as tmp directory to avoid quota limits

```bash
export TMPDIR=/scratch_local/
```
Now we are able to build the container with the standard command

```bash
apptainer build titan_leonardo_0.2.1.sif titan_0.2.1.def
```

To test the container, run the test script 
```bash
sbatch test_container_leonardo.sh
# Should output 
# 0: 2.10.0a0+a36e1d39eb.nv26.01.42222806
# 0: True
```
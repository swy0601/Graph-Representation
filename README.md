lines (132 sloc)  11.7 KB

# UNDETERMINED

The code is tested on Windows 10 environment (Python3.7, PyTorch_1.11.0) with GPU 2060.

## Contents

1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Usage

### Please see code comments for more details

### Quick start: Test after train

1. Download processed dataset from [Google Driver](https://drive.google.com/drive/folders/1n9o0onE8LbdUjymYFFgkdvRnmbnXVmHC?usp=sharing)
, place it in '/Dataset/Packaged Pkl/'.

   Note: You can also process the java-json dataset yourself.

2. Cd to '/Code', run the following script.

```bash
    python model_DMon-3.py
```

### Directly handle results stored in txt files

1. Ensure there is a result file generated.

2. Cd to '/Code/utils', run the following script.

```bash
    python data_handle.py
```

## Detail Package Version

```yaml
    numpy==1.21.6
    pandas==1.3.5
    scikit_learn==1.1.1
    torch==1.11.0
    torch_geometric==2.0.4
    tqdm==4.64.0
    transformers==4.18.0
```

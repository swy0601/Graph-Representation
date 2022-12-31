# UNDETERMINED

the results are run on a server with CPU 2.8 GHz Intel Core i7, 16GB RAM.

## Contents

1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Usage

### Please see code comments for more details

### Quick start: Train

1. Download processed dataset
   from [Google Driver](https://drive.google.com/file/d/1UdKI5R_yTBlV4tO5uDKcGJazg_SiTwKT/view?usp=sharing)
   , place it in '/Dataset/Packaged Pkl/'.

   Note: You can also process the java-json dataset yourself.

2. Cd to '/Code', run the following script.

```bash
    python train.py
```

### Build PKL yourself, you can customize the extra edges

1. Ensure there is a result file generated.

2. Cd to '/Code', run the following script.

```bash
    python code_dataset.py
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

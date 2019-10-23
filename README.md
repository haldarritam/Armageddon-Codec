# Armageddon-Codec

This is a video codec project developped with Python 3.7 in a linux environment (using Anaconda virtual environment) purely for academic purposes.

- [Armageddon-Codec](#armageddon-codec)
  - [Usage:](#usage)
    - [Compulsory arguments:](#compulsory-arguments)
    - [Optional arguments:](#optional-arguments)
  - [Creating environment:](#creating-environment)
    - [Cheat sheet for Anaconda environments:](#cheat-sheet-for-anaconda-environments)

## Usage:

This codec can be used in two ways as described below:

- The codec can be used by passing all the parameters for the codec as **command line arguments**.
  ```
  python main.py [-h] [-frames default=300] [-y_res default=288]
               [-x_res default=352] [-i default=16] [-r default=2]
               [-n default=3] [-QP default=3] [-ip default=6] -in ... -out ...
               [-o {encode,decode}] [-q default=4]
  ```

- The codec can also be used by passing a **config file** containing all the required parameters.
  ```
  python main.py @config_file.txt
  ```

  Two sample config files are given, one for question 3 and the other for question 4.

**NOTE**: Read the instructions for [creating and activating an environment](#creating-environment) before using the codec. The environment creation has been automated and can be easily done by following the instructions given in the link given above.

### Compulsory arguments: 
```
  -in ...              Input file location.
  -out ...             Output file location.
```

### Optional arguments:
```
  -h, --help           show this help message and exit
  -frames default=300  Number of frames to encode.
  -y_res default=288   Y-Resolution of the video.
  -x_res default=352   X-Resolution of the video.
  -i default=16        Block size for the encoder.
  -r default=2         Inter prediction search range.
  -n default=3         Approximation parameter (Q3).
  -QP default=3        Quantization parameter (Q4).
  -ip default=6        I-Period (Q4).
  -o {encode,decode}   Operation: (encode) or (decode)
  -q default=4         Question number.
```

## Creating environment:
After installing [Anaconda](https://www.anaconda.com/ "Anaconda's Homepage"), use the following command to create an environment for running the codec:
```
./env-setup.sh
```

Now, use the following command to activate the newly created environment:
```
conda activate armageddon
```
### Cheat sheet for Anaconda environments:
```
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf
```
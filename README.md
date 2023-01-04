# Decontaminator

**Decontaminator** is a deep learning helping tool that filters out phage or fungi contigs from plant virome RNAseq assemblies. 
To be used standalone or with VirHunter.

## Useful Info
Decontamintor goes well along with [VirHunter](https://github.com/cbib/virhunter). If you want to know a bit more about the two of them check out this [presentation](https://github.com/cbib/virhunter/blob/main/media/virhunter_description.pdf).

## System Requirements
Decontaminator installation requires a Unix environment with [python 3.8](http://www.python.org/). 
It was tested on Linux and macOS (M1 not tested).

To run Decontaminator you need to have git and conda already installed.
         
## Installation 

To install Decontaminator, you need to download it from github and then to install the dependancies.

```shell
git clone https://github.com/cbib/decontaminator.git
cd decontaminator/
```

We recommend using the environment provided in VirHunter to avoid duplication.
Otherwise, the commands to install and activate the environment are below.

```shell
conda env create -f envs/environment.yml
conda activate decontaminator
```

### Testing your installation of Decontaminator

You can test that Decontaminator was successfully installed on the toy dataset we provide.

First, you have to download the toy dataset
```shell
bash scripts/download_test_installation.sh
```
Then run the bash script that calls the testing, training and prediction python scripts of Decontaminator.
Attention, the training process may take some time.
```shell
bash scripts/test_installation.sh
```

## Using Decontaminator for prediction

To run Decontaminator you should use the already pre-trained models or train Decontaminator yourself (described in the next section).
Pre-trained model weights are already available for filtering of phages and fungi. To download them please launch the script:

```shell
bash scripts/download_weights.sh 
```

The weights will be located in the `weights` folder.

Before launching the prediction you will need to fill the `configs/predict_config.yaml` file. 
If for example, you want to use the weights of the pretrained model for fungi, 
you should change the field `weights` in the `configs/predict_config.yaml` to `weights/fungi`.
Otherwise, if you want to use consecutively both available filters, you should fill in the `configs/predict2_config.yaml`.

Decontaminator supports prediction for multiple test files at once. 
For that you need to change a bit the field `test_ds` in the
`configs/predict_config.yaml`. 

```yaml
predict:
    test_ds:
      - /path/to/test_ds_1
      - /path/to/test_ds_2
      - /path/to/test_ds_3  
```

Once the config file is ready, you can start the prediction:

```shell
python decontaminator/predict.py configs/predict_config.yaml
# or
python decontaminator/predict2.py configs/predict2_config.yaml
```

After prediction Decontaminator produces two `csv` files and one `fasta` file:

1. The first fasta file ends with `_viral.fasta`. It contains contigs that were predicted as viral by Decontaminator.
You should use these contigs for the next analyses (with VirHunter). __If you launch `predict2.py` you should look for file 
that ends with `viral_viral.fasta`.__
2. The second file ends with `_predicted_fragments.csv`
It is an intermediate result containing predictions of the three CNN networks (probabilities of belonging to each of the virus/plant/bacteria class) and of the RF classifier for each fragment of every contig.

3. The third file ends with `_predicted.csv`. 
This file contains final predictions for contigs calculated from the previous file. 
   - `id` - fasta header of a contig.
   - `length` - length of the contig.
   - `# viral fragments`, `# plant fragments` and `# bacterial fragments` - the number of fragments of the contig that received corresponding class prediction by the RF classifier.
   - `decision` - class given by the Decontaminator to the contig.
   - `# viral / # total` - number of viral fragments divided by the total number of fragments of the contig.
   - `# viral / # total * length` - number of viral fragments divided by the total number of fragments of the contig multiplied by contig length. It is used to display the most relevant contigs first.


## Training your own model

You can train your own model, for example for a specific contamination. Before training, you need to collect sequence 
data for training for three reference datasets: _viruses_ and _other_. 
Examples are provided by running `scripts/download_test_installation.sh` that will download `viruses.fasta` and
`bacteria.fasta` files for testing the installation.

Training requires execution of the following steps:
- prepare the training dataset for the neural network with `prepare_ds.py`.
- train the neural network with `train.py`

The training will be done twice - for fragment sizes of 500 and 1000.

The successful training of Decontaminator produces weights for one neural networks for fragment sizes of 500 and 1000. They can be subsequently used for prediction.

To execute the steps of the training you must first create a copy of the `template_config.yaml`. 
Then fill in the necessary parts of the config file. No need to fill in all tasks! 
Once config file is filled you can launch the scripts consecutively providing them with the config file like this:
```shell
python decontaminator/prepare_ds.py configs/config.yaml
```
And then
```shell
python decontaminator/train.py configs/config.yaml
```
Important to note, the suggested number of epochs for the training of neural networks is 10.

### Training Decontaminator on GPU

If you plan to train Decontaminator on GPU, please use `environment_gpu.yml` or `requirements_gpu.txt` for dependencies installation.
Those recipes were tested only on the Linux cluster with multiple GPUs.
If you plan to train Decontaminator on cluster with multiple GPUs, you will need to uncomment line with
`CUDA_VISIBLE_DEVICES` variable and replace `""` with `"N"` in header of `train.py`, where N is the number of GPU you want to use:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "N"
```

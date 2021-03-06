# Quantization of CNN-based Language Models
This repository will host the code for the project 'Quantization of CNN-based Language Models' which is a part of Intro to Deep Learning course at CMU.

For details about the project please refer to the report: `IDLProjectFinalReport.pdf`

Repo Structure:
* `model.py` - has the Wav2Letter model class defined.
* `train.py` - HW4 training routine where it loads the data, trains, and also validates on the dataset. 
* `evaluate.py` - takes a pretrained model, data directory and evaluates the model and returns average loss and average Levenshtien distance. 
* `hyperparameters.py` - contains our base parameters. To experiment with different hyperparameters, please change this file. 
* `utils/*` - has all the supporting utility codefiles that train, evaluate, and model file would need. 
* `data/*` - The data folder contains the Dataset class and the dataloaders used for training. The data - train data, train transcripts, validation data and validation labels should be present in the data folder.
* `quantization/*` - has supplementary scripts for quantization.
* `export_and_quantize.py` - contains the code used to export the trained model into an ONNX model and quantize the model using ONNX runtime. The inference can be done on the quantized .onnx models. Use the `evaluate.py` with the `--quantized` flag to save the quantized model and run the inference.
* `get_model_summary.py` - contains the code to get summary for the PyTorch model.
* `hardware_specs.py` - contains the code to fetch CPU and GPU information for benchmarking and comparison.
* `requirements.txt` - contains list of Python packages required for code execution.
* `EDADataset.ipynb` - is a jupyter notebook perform EDA on HW4P2 dataset. 
* `pretrained-models/*` - Contains the pretrained models on HW3 and HW4 data. The .onnx models are the quantized models.

### Usage:
#### To Train the model:
```
python train.py
```
#### To Evaluate with an already trained model:
* Float_32 model:
```
python evaluate.py --model ./models/wav2letter_31.pth --datadir ./HW4/data/ --batchsize 64
```

* Quantized model:
```
python evaluate.py --quantized --model ./models/quantized_model_int8.onnx --datadir ./HW4/data/
```

#### To export PyTorch model to ONNX and quantize it:
* Export model to ONNX
```
python export_and_quantize.py --pypath ./Project/w2l30_6.pth --quantize 0 --classes 35 --outpath ./Project/exported_model.onnx
```
* Quantize ONNX model
```
python export_and_quantize.py --quantize 1 --classes 35 --outpath ./Project/exported_model.onnx --quantpath ./Project/model_quant.onnx --excludenodes Conv_0 Conv_18 Conv_20
```

#### To get CPU and GPU stats:
* CPU stats:
```
python hardware_specs.py --outpath cpu_stats.json
```
* GPU stats:
```
python hardware_specs.py --outpath gpu_stats.txt --cpu 0
```

#### To get PyTorch model summary:
```
python get_model_summary.py --pypath ./Project/w2l30_6.pth >> ./Project/model_summary.txt
```

#### Note:
* Currently, there is only a CPU support for the quantized models.


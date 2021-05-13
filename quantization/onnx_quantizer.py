# further documentation: https://www.onnxruntime.ai/docs/how-to/quantization.html

from onnxruntime.quantization import quantize_dynamic,QuantType

def onnx_quantizer(onnx_path, quantized_path, nodes_to_exclude=[]):
    quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8,nodes_to_exclude=[])


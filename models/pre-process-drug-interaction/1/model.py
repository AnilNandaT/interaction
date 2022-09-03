import json
import numpy as np
from triton_python_backend_utils import get_input_tensor_by_name, Tensor, get_output_config_by_name, triton_string_to_numpy, InferenceResponse
import deepchem as dc
featurizer = dc.feat.CircularFingerprint()

class DI:
    def __init__(self) -> None:
        super().__init__()
    def pre_process(self, inp1, inp2):
        lst = []
        for counter in range(inp1.shape[0]):
            input1 = inp1[counter]
            input2 = inp2[counter]
            drug1 = featurizer(input1[0].decode("utf-8") )
            drug2 = featurizer(input2[0].decode("utf-8"))
            drugs = np.array([np.transpose(drug1), np.transpose(drug2)])
            lst.append(np.transpose(drugs).reshape(64, 64, 1))
        return np.asarray(lst)
class TritonPythonModel:
    def __init__(self) -> None:
        super().__init__()
        self.model_config = None
        self.result_dtype = None
        self.di_model = DI()
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        out_config_result = get_output_config_by_name(model_config, "pre_process_output")
        self.result_dtype = triton_string_to_numpy(out_config_result['data_type'])
    def execute(self, requests):
        responses = []
        di_model = self.di_model
        for request in requests:
            model_input1 = get_input_tensor_by_name(request, "pre_process_input1")
            model_input2 = get_input_tensor_by_name(request, "pre_process_input2")
            result = di_model.pre_process(model_input1.as_numpy(),model_input2.as_numpy())
            out_tensor_result = Tensor("pre_process_output", result.astype(self.result_dtype))
            inference_response = InferenceResponse(output_tensors=[out_tensor_result])
            responses.append(inference_response)
            #del di_model
        return responses
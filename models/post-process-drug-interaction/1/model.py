import json
import numpy as np
from triton_python_backend_utils import get_input_tensor_by_name, Tensor, get_output_config_by_name, triton_string_to_numpy, InferenceResponse
import deepchem as dc
from operator import itemgetter
from numpy import genfromtxt
featurizer = dc.feat.CircularFingerprint()
legend =    {0: "None",
            1: '#Drug1 may increase the photosensitizing activities of #Drug2.',
            2: '#Drug1 may increase the anticholinergic activities of #Drug2.',
            3: 'The bioavailability of #Drug2 can be decreased when combined with #Drug1.',
            4: 'The metabolism of #Drug2 can be increased when combined with #Drug1.',
            5: '#Drug1 may decrease the vasoconstricting activities of #Drug2.',
            6: '#Drug1 may increase the anticoagulant activities of #Drug2.',
            7: '#Drug1 may increase the ototoxic activities of #Drug2.',
            8: 'The therapeutic efficacy of #Drug2 can be increased when used in combination with #Drug1.',
            9: '#Drug1 may increase the hypoglycemic activities of #Drug2.',
            10: '#Drug1 may increase the antihypertensive activities of #Drug2.',
            11: 'The serum concentration of the active metabolites of #Drug2 can be reduced when #Drug2 is used in '
                'combination with #Drug1 resulting in a loss in efficacy.',
            12: '#Drug1 may decrease the anticoagulant activities of #Drug2.',
            13: 'The absorption of #Drug2 can be decreased when combined with #Drug1.',
            14: '#Drug1 may decrease the bronchodilatory activities of #Drug2.',
            15: '#Drug1 may increase the cardiotoxic activities of #Drug2.',
            16: '#Drug1 may increase the central nervous system depressant (CNS depressant) activities of #Drug2.',
            17: '#Drug1 may decrease the neuromuscular blocking activities of #Drug2.',
            18: '#Drug1 can cause an increase in the absorption of #Drug2 resulting in an increased serum '
                'concentration and potentially a worsening of adverse effects.',
            19: '#Drug1 may increase the vasoconstricting activities of #Drug2.',
            20: '#Drug1 may increase the QTc-prolonging activities of #Drug2.',
            21: '#Drug1 may increase the neuromuscular blocking activities of #Drug2.',
            22: '#Drug1 may increase the adverse neuromuscular activities of #Drug2.',
            23: '#Drug1 may increase the stimulatory activities of #Drug2.',
            24: '#Drug1 may increase the hypocalcemic activities of #Drug2.',
            25: '#Drug1 may increase the atrioventricular blocking (AV block) activities of #Drug2.',
            26: '#Drug1 may decrease the antiplatelet activities of #Drug2.',
            27: '#Drug1 may increase the neuroexcitatory activities of #Drug2.',
            28: '#Drug1 may increase the dermatologic adverse activities of #Drug2.',
            29: '#Drug1 may decrease the diuretic activities of #Drug2.',
            30: '#Drug1 may increase the orthostatic hypotensive activities of #Drug2.',
            31: 'The risk or severity of hypertension can be increased when #Drug2 is combined with #Drug1.',
            32: '#Drug1 may increase the sedative activities of #Drug2.',
            33: 'The risk or severity of QTc prolongation can be increased when #Drug1 is combined with #Drug2.',
            34: '#Drug1 may increase the immunosuppressive activities of #Drug2.',
            35: '#Drug1 may increase the neurotoxic activities of #Drug2.',
            36: '#Drug1 may increase the antipsychotic activities of #Drug2.',
            37: '#Drug1 may decrease the antihypertensive activities of #Drug2.',
            38: '#Drug1 may increase the vasodilatory activities of #Drug2.',
            39: '#Drug1 may increase the constipating activities of #Drug2.',
            40: '#Drug1 may increase the respiratory depressant activities of #Drug2.',
            41: '#Drug1 may increase the hypotensive and central nervous system depressant (CNS depressant) '
                'activities of #Drug2.',
            42: 'The risk or severity of hyperkalemia can be increased when #Drug1 is combined with #Drug2.',
            43: 'The protein binding of #Drug2 can be decreased when combined with #Drug1.',
            44: '#Drug1 may increase the central neurotoxic activities of #Drug2.',
            45: '#Drug1 may decrease effectiveness of #Drug2 as a diagnostic agent.',
            46: '#Drug1 may increase the bronchoconstrictory activities of #Drug2.',
            47: 'The metabolism of #Drug2 can be decreased when combined with #Drug1.',
            48: '#Drug1 may increase the myopathic rhabdomyolysis activities of #Drug2.',
            49: 'The risk or severity of adverse effects can be increased when #Drug1 is combined with #Drug2.',
            50: 'The risk or severity of heart failure can be increased when #Drug2 is combined with #Drug1.',
            51: '#Drug1 may increase the hypercalcemic activities of #Drug2.',
            52: '#Drug1 may decrease the analgesic activities of #Drug2.',
            53: '#Drug1 may increase the antiplatelet activities of #Drug2.',
            54: '#Drug1 may increase the bradycardic activities of #Drug2.',
            55: '#Drug1 may increase the hyponatremic activities of #Drug2.',
            56: 'The risk or severity of hypotension can be increased when #Drug1 is combined with #Drug2.',
            57: '#Drug1 may increase the nephrotoxic activities of #Drug2.',
            58: '#Drug1 may decrease the cardiotoxic activities of #Drug2.',
            59: '#Drug1 may increase the ulcerogenic activities of #Drug2.',
            60: '#Drug1 may increase the hypotensive activities of #Drug2.',
            61: '#Drug1 may decrease the stimulatory activities of #Drug2.',
            62: 'The bioavailability of #Drug2 can be increased when combined with #Drug1.',
            63: '#Drug1 may increase the myelosuppressive activities of #Drug2.',
            64: '#Drug1 may increase the serotonergic activities of #Drug2.',
            65: '#Drug1 may increase the excretion rate of #Drug2 which could result in a lower serum level and '
                'potentially a reduction in efficacy.',
            66: 'The risk or severity of bleeding can be increased when #Drug1 is combined with #Drug2.',
            67: '#Drug1 can cause a decrease in the absorption of #Drug2 resulting in a reduced serum concentration '
                'and potentially a decrease in efficacy.',
            68: '#Drug1 may increase the hyperkalemic activities of #Drug2.',
            69: '#Drug1 may increase the analgesic activities of #Drug2.',
            70: 'The therapeutic efficacy of #Drug2 can be decreased when used in combination with #Drug1.',
            71: '#Drug1 may increase the hypertensive activities of #Drug2.',
            72: '#Drug1 may decrease the excretion rate of #Drug2 which could result in a higher serum level.',
            73: 'The serum concentration of #Drug2 can be increased when it is combined with #Drug1.',
            74: '#Drug1 may increase the fluid retaining activities of #Drug2.',
            75: 'The serum concentration of #Drug2 can be decreased when it is combined with #Drug1.',
            76: '#Drug1 may decrease the sedative activities of #Drug2.',
            77: 'The serum concentration of the active metabolites of #Drug2 can be increased when #Drug2 is used in '
                'combination with #Drug1.',
            78: '#Drug1 may increase the hyperglycemic activities of #Drug2.',
            79: '#Drug1 may increase the central nervous system depressant (CNS depressant) and hypertensive '
                'activities of #Drug2.',
            80: '#Drug1 may increase the hepatotoxic activities of #Drug2.',
            81: '#Drug1 may increase the thrombogenic activities of #Drug2.',
            82: '#Drug1 may increase the arrhythmogenic activities of #Drug2.',
            83: '#Drug1 may increase the hypokalemic activities of #Drug2.',
            84: '#Drug1 may increase the vasopressor activities of #Drug2.',
            85: '#Drug1 may increase the tachycardic activities of #Drug2.',
            86: 'The risk of a hypersensitivity reaction to #Drug2 is increased when it is combined with #Drug1.',
            }

synergy = genfromtxt('/cv_root/synergy.csv', delimiter='\n')
def percentile_calc(value):
    return np.count_nonzero(synergy < value) / synergy.size
    
class Post_Process:
    def __init__(self) -> None:
        super().__init__()
    def post_process(self, inp):
        lst = []
        for counter in range(inp.shape[0]):
            data = inp[counter].astype(float)
            data = (data * 100)
            tmp = data[8]
            updated_tmp = percentile_calc(tmp)
            data[8] = updated_tmp
            #result = [{"DDI": tmp, "Percentile":updated_tmp}]
            result = [{"label": legend[i], "value":round(x, 4)} for i, x in enumerate(data)]
            #sorted_result = sorted(result, key=itemgetter('value'), reverse=True)
            #lst.append(sorted_result)
            lst.append(result)
        return lst
class TritonPythonModel:
    def __init__(self) -> None:
        super().__init__()
        self.model_config = None
        #self.result_dtype = None
        self.post_process_model = Post_Process()
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        out_config_result = get_output_config_by_name(model_config, "post_process_output")
        self.result_dtype = triton_string_to_numpy(out_config_result['data_type'])
    def execute(self, requests):
        responses = []
        post_process_model = self.post_process_model
        for request in requests:
            model_input = get_input_tensor_by_name(request, "post_process_input")
            result = post_process_model.post_process(model_input.as_numpy())
            #out_tensor_result = Tensor("post_process_output", result.as_numpy().astype(self.result_dtype))
            out_tensor_result = Tensor("post_process_output", np.array(result,dtype=np.bytes_))
            inference_response = InferenceResponse(output_tensors=[out_tensor_result])
            responses.append(inference_response)
            #del di_model
        return responses

import logging
import os
import torch
from flask import Flask, jsonify
from flask import request, jsonify
import json
import waitress
import time
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import AutoTokenizer
import argparse
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()

CHECK_POINT_DIR="/data/pengjun/Megatron-DeepSpeed/checkpoints"
parser = argparse.ArgumentParser()
parser.add_argument(
    '-p', '--port', default=8097,
    help='falcon server port')
# /data/pengjun/model_ckpt/bloom-1b7
# /data/pengjun/Megatron-DeepSpeed/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles
parser.add_argument(
    '-c', '--model_file', default='/data/pengjun/Megatron-DeepSpeed/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles',
    help='model config file')
# parser.add_argument(
#     '-f', '--fp16', default=False,
#     help='open fp16(True) or not (False)')
# parser.add_argument(
#     '-k', '--top_k', default=40,
#     help='top k param')
#
# required_devices_infer={"1B7":1,"3B":1,"7B1":2}
import pynvml


def get_cuda_properties():
    Total_Device= torch.cuda.device_count()
    # Total_Memory= 0
    info={}
    for i in range(Total_Device):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 指定GPU的id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info[i]={"allocated":meminfo.used / 1024**2,"all":meminfo.total / 1024**2,"free":meminfo.free / 1024**2}
    return info
#
def get_available_gpu(model_params):
    info=get_cuda_properties()
    print(info)
    model_params_consume = {"1B7_infer": 24955, "3B_infer": 23092, "7B1_infer": 48354} # todo change to real params
    available_gpu=[str(i) for i,j in info.items() if j["free"]>model_params_consume[model_params]]
    return available_gpu


args = parser.parse_args()
model_config = args.model_file
# is_fp16 = args.fp16

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
registered_models={}
# model = BloomForCausalLM.from_pretrained(model_config, torch_dtype=torch.float16)
# else:
    # model = BloomForCausalLM.from_pretrained(model_config)
# generator = model.to(device)
# tokenizer = BloomTokenizerFast.from_pretrained(model_config, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(model_config, use_fast=False)

app = Flask(__name__)
@app.route('/model/getonline', methods=['GET'])
def get_register_model():
    models=registered_models.keys()
    return  jsonify({"registered_models":str(models)})

@app.route('/model/registration', methods=['POST'])
def register_model():
    jsondata = request.get_json()
    model_name=jsondata.get('model_name',"main")
    model_params=jsondata.get("model_params","1B7")
    global registered_models
    model_names=f"{model_name}_{model_params}"
    if model_names in registered_models:
        return jsonify({"message":f"model {model_names} already registered"}),200
    else:
        try:
            model_config=os.path.join(CHECK_POINT_DIR,"hf_checkpoints",f"tr11b-{model_params}-ml",model_name)
            available_devices = get_available_gpu(f"{model_params}_infer")
            if available_devices:
                device = available_devices[0]
            logger.info(f"loading {model_names}.......")
            model= BloomForCausalLM.from_pretrained(model_config, torch_dtype=torch.float16)
            model=model.to(torch.device(f"cuda:{device}"))
            logger.info(f"{model_names} loaded")

            registered_models.update({model_names:{"model":model,"device":device}})
            return jsonify({"message":f"model {model_names}  registered","status":"success"}),200
        except Exception as e:
            return jsonify({"message":f"Error:{e}",'status':"failed"}),500

@app.route('/model/unregistration', methods=['POST'])
def unregister_model():
    jsondata = request.get_json()
    model_name=jsondata.get('model_name',"main")
    model_params=jsondata.get("model_params","1B7")
    global registered_models
    model_names=f"{model_name}_{model_params}"
    if model_names in registered_models:
        del registered_models[model_names]["model"]
        registered_models.pop(model_names)
        torch.cuda.empty_cache()

        return jsonify({"message":f"model {model_names}  unregistered"}),200
    else:
        return jsonify({"message": f"model {model_names}  not registered"}), 200


@app.route('/model/generate', methods=['POST'])
def bloom_generate():
    # return jsonify(text='Hello, searching!'), 200

    jsondata = request.get_json()
    model_name=jsondata.get('model_name',"main")
    model_params=jsondata.get("model_params","1B7")
    generator=registered_models.get(f"{model_name}_{model_params}").get("model")
    device=registered_models.get(f"{model_name}_{model_params}").get("device")
    bg = BloomGenerate(device,generator)
    res = bg.main(jsondata)
    return res


class BloomGenerate:
    def __init__(self,device,generator):
        # if torch.cuda.is_available():
        self.device = torch.device(f'cuda:{device}')
        self.generator=generator
        # else:
        # self.generator = model.to(self.device)
        # self.tokenizer = BloomTokenizerFast.from_pretrained("/data/lvyang/rct-BLOOM/model_ckpt/bloom-6b3", use_fast=False)
        self.default_args = {
            'seq_length': 2048,
            'max_length': 50,
            'top_k': 40,
            'top_p': 0.9,
            'temperature': 0.9,

            'repetition_penalty': None,
            'pad_token_id': None,
            'eos_token_id': None,
            'length_penalty': None,
            'max_new_tokens': None,
            'max_time': None,
            'min_length': None,

            'num_return_sequences': None,
            'typical_p': None,

        }

    def generate_samples(self, config, args):  # 产出文本
        try:
            raw_text = config['prompt']
        except Exception as e:
            logger.exception(e)
            logger.info("请传入prompt")
            return ""
        logger.info("raw_text_{}: ".format(self.time_sign) + raw_text)
        input_ids = tokenizer(raw_text, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        word_piece_len = input_ids.shape[-1]

        # 空格数与word piece数量并不严格相等，当分词器将一个词切成多个word piece时，
        # word_piece_len要大于word数量「空格数」。因此要使用output_len+word_piece_len作为max_length

        output_len = config.get('length', args['max_length'])
        max_length = output_len + word_piece_len

        top_k = config.get('top_k', args['top_k'])
        top_p = config.get('top_p', args['top_p'])
        temperature = config.get('temperature', args['temperature'])

        repetition_penalty = config.get('repetition_penalty', args['repetition_penalty'])
        pad_token_id = config.get('pad_token_id', args['pad_token_id'])
        eos_token_id = config.get('eos_token_id', args['eos_token_id'])
        length_penalty = config.get('length_penalty', args['length_penalty'])
        max_new_tokens = config.get('max_new_tokens', args['max_new_tokens'])
        max_time = config.get('max_time', args['max_time'])
        min_length = config.get('min_length', args['min_length'])


        num_return_sequences = config.get('num_return_sequences', args['num_return_sequences'])
        typical_p = config.get('typical_p', args['typical_p'])
        logger.info("length_penalty:" + str(length_penalty))
        if word_piece_len >= args['seq_length']:
            raw_text = raw_text[-(args.seq_length - output_len - 20):]
            input_ids = tokenizer(raw_text, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)

            logger.info( "\nContext length " + str(len(raw_text)) + \
                   "\nPlease give smaller context (half of the sequence length)!")
        t = time.time()
        if max_length is not None:
            gen_tokens = self.generator.generate(input_ids,
                                                 top_k=top_k,
                                                 top_p=top_p,
                                                 do_sample=True,
                                                 temperature=temperature,
                                                 max_length=max_length,
                                                 # bad_words_ids=self.bad_words_ids,
                                                 repetition_penalty=repetition_penalty,
                                                 pad_token_id=pad_token_id,
                                                 eos_token_id=eos_token_id,
                                                 length_penalty=length_penalty,
                                                 max_time=max_time,
                                                 min_length=min_length,
                                                 num_return_sequences=num_return_sequences,
                                                 typical_p=typical_p)
        elif max_new_tokens is not None:
            gen_tokens = self.generator.generate(input_ids,
                                                 top_k=top_k,
                                                 top_p=top_p,
                                                 do_sample=True,
                                                 temperature=temperature,
                                                 max_new_tokens=max_new_tokens,
                                                 # bad_words_ids=self.bad_words_ids,
                                                 repetition_penalty=repetition_penalty,
                                                 pad_token_id=pad_token_id,
                                                 eos_token_id=eos_token_id,
                                                 length_penalty=length_penalty,
                                                 max_time=max_time,
                                                 min_length=min_length,
                                                 num_return_sequences=num_return_sequences,
                                                 typical_p=typical_p)

        logger.info(f'time cost:{time.time() - t:.8f}s')
        generate_text = tokenizer.batch_decode(gen_tokens)[0]
        generate_text = generate_text.replace(raw_text, "")
        logger.info("generate_text_{}: ".format(self.time_sign) + generate_text)
        return generate_text

    def gpt_generate(self, jsondata):
        # logger.info(jsondata)
        number = jsondata.get('number', 3)
        answer_list = []
        for _ in range(number):
            answer_list.append(
                self.generate_samples(jsondata, self.default_args))
        return {"result": answer_list}

    def main(self, jsondata):
        """Handles POST requests"""

        self.time_sign = jsondata.get('time_sign', "None")
        logger.info("get_jsondata_{}: {}".format(self.time_sign, jsondata))
        result = self.gpt_generate(jsondata)
        return result


if __name__ == "__main__":
    waitress.serve(app, host='0.0.0.0', port=args.port, threads=50)

import torch
import pynvml
from flask import Flask, render_template, jsonify, request
import base64
import requests
import random
from datetime import datetime
import re
import os
import json
import uuid
import subprocess
import threading
# import oss2
import time
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/task/create", methods=["POST"])
def create_task_text():
    json_data=request.get_json()
    device=get_available_gpu()
    print("available_gpu:",device)
    if len(device)<2:
        return jsonify({"error":"not enough gpus"}),500
    if "data_name" not in json_data:
        return jsonify({"error":"please provide a valid data_name"}),500
    # if "load_from_name" not in json_data:
    #     return jsonify({"error":"please provide a valid load_from_name"}),500
    data_name=json_data.get("data_name")
    load_from_name=json_data.get("load_from_name","main")
    port=getPort()
    cmd=f"bash run_train_fp16.sh {data_name} {load_from_name} {port}"
    popen = subprocess.Popen(cmd,env={"CUDA_VISIBLE_DEVICES":",".join(device[:2])}, shell=True)
    return jsonify({"status":"success","message":f"task for data {data_name} created"}), 200

@app.route("/data/get", methods=["GET"])
def get_data():
    names=[file.rstrip(".jsonl") for file in os.listdir("./upload_data") if file.endswith("jsonl")]
    return jsonify({"exsist data_name":names})

@app.route("/data/delete", methods=["POST"])
def delete_data():
    data_name = request.form.get("data_name")
    os.remove(f"./upload_data/{data_name}.jsonl")
    return jsonify({"status":f"{data_name} deleted"})

@app.route("/data/combine", methods=["POST"])
def combine_data():
    data_names = request.form.get("data_names")
    to_data_name = request.form.get("to_data_name")
    data_names=data_names.split(",")
    for data_name in data_names:
        with open(f"./upload_data/{data_name}.jsonl","rb") as f:
                with open(f"./upload_data/{to_data_name}.jsonl","ab") as fb:
                    fb.write(f.read())
    insert_config_json(to_data_name)
    convert_jsonl_meg(to_data_name)
    return jsonify({"message":"data combined","status":"success"})


@app.route("/data/upload", methods=["POST"])
def upload_img():
    data = request.files.get("file")
    #print(data.filename)
    data_type = request.form.get("format","jsonl" )
    data_name = request.form.get("data_name","test" )
    shuffle =bool(request.form.get("shuffle",1))
    # shards = int(request.form.get("shards","1" ))
    if data_type not in ["txt","jsonl"] :
        return jsonify({"error":"please upload a txt or jsonl file"}),500
    if data_type=="txt":
        out=convert_text_to_json(data,data_name)
        if not out:
            if shuffle:
                shuffle_data(data_name)
            insert_config_json(data_name)

            convert_jsonl_meg(data_name)
            return jsonify({"status":"success","message":"upload success"}),200

        else:
            return jsonify({"error":out,"status":'failed'}),500
    else:
        if shuffle:
            shuffle_data(data_name)
        data.save(f"upload_data/{data_name}.jsonl")
        insert_config_json(data_name)
        convert_jsonl_meg(data_name)
        return jsonify({"status": "success", "message": "upload success"}), 200

@app.route("/task/get", methods=["GET"])
def get_status():
    data_name = request.form.get("data_name", "test")
    try:
        data=get_status_log(data_name)
    except Exception as e:
        return jsonify({"status":"failed","message":str(e)}),500
    return jsonify({"status": "success", "message": f"目前已完成{data[1]}中的{data[0]}轮"}), 200



def convert_text_to_json(data,data_name):
    try:
        with open(f"upload_data/{data_name}.jsonl","w") as f:
            # print(data)
            # lines=data.stream.read().decode('ascii')
            data.save(f"upload_data/{data_name}.txt")
            lines=open(f"upload_data/{data_name}.txt","r").readlines()
            # print(lines)
            for index,text in enumerate(lines):
                # print(text)
                f.write(json.dumps({"id":index+1,"text":text.rstrip()},ensure_ascii=False))
                f.write("\n")
        return 0
    except Exception as e:
        print(e)
        return str(e)

def convert_jsonl_meg(data_name):
    cmd=f"python tools/preprocess_data.py  --input  upload_data/{data_name}.jsonl --output-prefix preprocess_data/{data_name}  --vocab data/gpt2-vocab.json  --merge-file data/gpt2-merges.txt  --dataset-impl mmap  --tokenizer-type GPT2BPETokenizer  --append-eod  --workers 16"
    popen = subprocess.Popen(cmd, shell=True)

def shuffle_data(data_name):
    cmd=f"shuf upload_data/{data_name}.jsonl -o upload_data/{data_name}.jsonl"
    popen = subprocess.Popen(cmd, shell=True)
def insert_config_json(data_name):
    data=[{"dataset_path": f"preprocess_data/{data_name}_text_document","ratio": 1}]
    with open(f"preprocess_data/{data_name}.json","w") as f:
        json.dump(data,f)

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
def get_available_gpu():
    info=get_cuda_properties()
    print(info)
    available_gpu=[str(i) for i,j in info.items() if j["free"]>21000]
    return available_gpu

def get_status_log(data_name):
    LOG_PATH=f"/data/pengjun/Megatron-DeepSpeed/checkpoints/tr11b-1B3-ml/tr11f-1B3-ml-logs/logs/{data_name}/main_log.txt"
    with open(LOG_PATH,"r") as f:
        data=f.readlines()[::-1]
        for text in data:
            print(text)
            if "iteration" in text and "consumed samples" in text:
                text=text.split("consumed samples")[0].split("iteration")[1].replace("|","").split("/")
                all_iter=int(text[1])
                now_iter=int(text[0])
                return  now_iter,all_iter
            elif "setting training iterations to" in text:
                return 0,int(text.split("setting training iterations to")[1])

# print(get_status_log("main"))
def getPort():
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(6001,6030)
    if tt not in procarr:
        return tt
    else:
        return getPort()

# print(getPort())

# print(get_cuda_properties())
# print(get_available_gpu())

app.run(host='0.0.0.0', port=8099)



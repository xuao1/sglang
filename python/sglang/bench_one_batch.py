"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run

# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server import _set_envs_and_config
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_process_tree, suppress_other_loggers

from sglang.srt.model_executor.forward_batch_info import ForwardMode

import threading
from torch.profiler import profile, ProfilerActivity

import freeslots

import datetime

import gc

gc.disable()

CONTROL_FILE = "/tmp/train_control.txt"
def file_listener(model_runner):
    """文件监控线程函数"""
    last_mtime = 0
    while True:
        try:
            if not os.path.exists(CONTROL_FILE):
                # 如果文件不存在则创建
                with open(CONTROL_FILE, "w") as f:
                    f.write("1")  # 默认运行状态
                continue

            current_mtime = os.path.getmtime(CONTROL_FILE)
            if current_mtime > last_mtime:
                # 检测到文件修改
                with open(CONTROL_FILE, "r") as f:
                    content = f.read().strip()
                
                # 解析控制命令
                cmd = content.split()[-1]  # 取最后一行内容
                new_state = cmd != "0"

                # 更新模型状态（带属性校验）
                if hasattr(model_runner, "finetune_model") and hasattr(model_runner.finetune_model, "base_model"):
                    target = model_runner.finetune_model.base_model.model.model
                    if hasattr(target, "pause_train"):
                        target.pause_train = new_state
                        logging.info(f"Training state updated: {'Paused' if new_state else 'Running'}")
                
                last_mtime = current_mtime

        except Exception as e:
            logging.error(f"File monitor error: {str(e)}")
        
        time.sleep(0.1)


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.context_length,
        model_override_args=server_args.json_model_override_args,
        is_embedding=server_args.is_embedding,
        dtype=server_args.dtype,
        quantization=server_args.quantization,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
    )
    batch.prepare_for_extend()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner, device, stream = None):
    batch.output_ids = input_token_ids
    # if batch.count_time == False:
    batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    # synchronize(device)
    # if(stream):
    #     stream.wait_stream(stream)
    #     tic = time.time()
    #     forward_num = 1
    #     for i in range(forward_num):
    #         logits_output = model_runner.forward(forward_batch)
    #     if torch.cuda.is_available():
    #         completion_event = torch.cuda.Event()
    #         completion_event.record(stream=stream)
    #     if completion_event is not None:
    #         stream.wait_event(completion_event)
    #     latency = time.time() - tic
    #     latency = latency / forward_num
    completion_event_start = None
    completion_event_end = None
    completion_event_start = torch.cuda.Event(enable_timing=True)
    completion_event_end = torch.cuda.Event(enable_timing=True)

    if stream is not None:
        # stream.wait_stream(torch.cuda.current_stream())
        # stream.synchronize()
        completion_event_start.record(stream=stream)
        with torch.cuda.stream(stream):
            forward_num = 1
            for i in range(forward_num):
                logits_output = model_runner.forward(forward_batch)
            completion_event_end.record(stream=stream)
        
        # completion_event.synchronize()
        completion_event_end.synchronize()
        latency = completion_event_start.elapsed_time(completion_event_end)

        latency = latency / forward_num
    else:
        print("decode, in else")
        tic = time.time()
        forward_num = 1
        for i in range(forward_num):
            logits_output = model_runner.forward(forward_batch)
        latency = time.time() - tic
        latency = latency / forward_num
    
    # synchronize(device)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, latency


def correctness_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def get_gpu_memory(device):
    """
    Returns the current GPU memory usage in MB.
    """
    if torch.cuda.is_available():
        # return torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert to MB
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024  # Convert to MBZ
    return 0


def print_dataclass(obj, indent=0):
    prefix = ' ' * indent
    if dataclasses.is_dataclass(obj):
        print(f'{prefix}{type(obj).__name__}:')
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            print(f'{prefix}  {field.name}:', end=' ')
            print_dataclass(value, indent + 4)
    elif isinstance(obj, list):
        print()
        for i, item in enumerate(obj):
            print(f'{prefix}[{i}]:', end=' ')
            print_dataclass(item, indent + 4)
    elif isinstance(obj, dict):
        print()
        for key, value in obj.items():
            print(f'{prefix}{key}:', end=' ')
            print_dataclass(value, indent + 4)
    else:
        print(obj)


formal_stream_a, stream_b = freeslots.create_greenctx_stream_by_percent(0.6, 0.4, 0)
# 4 倍数

def latency_test_run_once(
    run_name, model_runner, rank_print, reqs, batch_size, input_len, output_len, device
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    # if batch_size > max_batch_size:
    #     rank_print(
    #         f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
    #     )
    #     return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool.clear()

    native_stream = torch.cuda.Stream(device=device)
    finetune_thread = None
    LlamaModel = None

    # # # =============================================================================================================
    # # # =============================================================================================================
    # # # test finetune
    # model_runner.load_finetune_model()
    # # print("model_runner.finetune_model.base_model.model.model.pause_train: ", model_runner.finetune_model.base_model.model.model.pause_train)

    # # input_thread = threading.Thread(
    # #     target=file_listener,
    # #     args=(model_runner,),
    # #     daemon=True
    # # )
    # # input_thread.start()

    # stream_b = native_stream

    # model_runner.finetune_model.base_model.model.model.compute_stream = stream_b

    # with torch.cuda.stream(stream_b):
    #     model_runner.finetune_train()

    # time.sleep(10000)
    # # # =============================================================================================================
    # # # =============================================================================================================


    # rank_print(f"Before running. GPU memory used: {get_gpu_memory(device):.2f} MB")

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    # Prefill
    synchronize(device)
    tic = time.time()

    # for ii in range(10):
    #     next_token_ids, _, batch = extend(reqs, model_runner)
    #     synchronize(device)
    # prefill_latency = (time.time() - tic) / 10.0
    next_token_ids, _, batch = extend(reqs, model_runner)
    # print("batch:")
    # print_dataclass(batch)
    synchronize(device)
    prefill_latency = (time.time() - tic)

    # next_token_ids = torch.full((batch_size,), 323, dtype=torch.int32, device='cuda:0')
    # batch = ScheduleBatch(forward_mode=ForwardMode.EXTEND, reqs=batch_size)

    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput
    # rank_print(f"After Prefill. GPU memory used: {get_gpu_memory(device):.2f} MB")

    # with torch.cuda.stream(stream_a):
    #     output_len = 100
    #     for i in range(output_len - 1):
    #         # print(i + 1, end=" ")
    #         # stream_a.wait_stream(stream_a)
    #         tic = time.time()

    #         # # print("1000000000 decode")
    #         # batch.count_time = True
    #         # for ii in range(10):
    #         #     _, _ = decode(next_token_ids, batch, model_runner)
    #         # # print("after 1000000000 decode")
    #         # synchronize(device)
    #         # latency = time.time() - tic
    #         # latency = latency / 10
    #         # tot_latency += latency
    #         # throughput = batch_size / latency

    #         # batch.count_time = False
    #         next_token_ids, _, forward_latency = decode(next_token_ids, batch, model_runner, device, stream_a)

    #         latency = forward_latency
    #         # latency = time.time() - tic
    #         # latency = latency / 10
    #         tot_latency += latency
    #         throughput = batch_size / latency

    #         if i % 1 == 0:
    #             rank_print(
    #                 f"Decode. i:{i},  latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    #             )


    # # # =============================================================================================================
    # # # =============================================================================================================
    # # # test finetune
    # model_runner.load_finetune_model()

    # LlamaModel = model_runner.finetune_model.base_model.model.model

    # # model_runner.finetune_model.base_model.model.model.compute_stream = stream_b
    # LlamaModel.compute_stream = stream_b

    # def run_finetune():
    #     torch.cuda.set_device(0)
    #     with torch.cuda.stream(LlamaModel.compute_stream):
    #         model_runner.finetune_train()

    # finetune_thread = threading.Thread(target=run_finetune, daemon=True)
    # finetune_thread.start()

    # # with torch.cuda.stream(stream_b):
    # #     model_runner.finetune_train()
    
    # time.sleep(60)
    # print("After start finetune_train")

    # # time.sleep(10000)
    # # # =============================================================================================================
    # # # =============================================================================================================

    # Decode
    decode_latencies = []
    # rank_print(
    #     f"Decode. output_len"
    # )

    stream_a = native_stream
    # stream_a = formal_stream_a

    with torch.cuda.stream(stream_a):
        for i in range(output_len - 1):
            # if LlamaModel is not None:
            #     if LlamaModel.changeSM == 1:
            #         # print("In bench_one_batch changeSM = 1")
            #         LlamaModel.compute_stream = None
            #         stream_a = native_stream
            #         LlamaModel.changeSM = 0
            #     elif LlamaModel.changeSM == -1:
            #         # print("In bench_one_batch changeSM = -1")
            #         LlamaModel.compute_stream = stream_b
            #         stream_a = formal_stream_a
            #         LlamaModel.changeSM = 0
            
            next_token_ids, _, forward_latency = decode(next_token_ids, batch, model_runner, device, stream_a)

            latency = forward_latency
            tot_latency += latency
            throughput = batch_size / latency

            decode_latencies.append(latency)
            if i > 0 and i % 8 == 0:
                avg_latency = sum(decode_latencies[-8:]) / 8
                # rank_print(
                #         f"Decode. i:{i},  latency: {avg_latency:6.5f} ms"
                #     )
                rank_print(
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Decode. i:{i},  latency: {avg_latency:6.5f} ms"
                )

    # with torch.cuda.stream(stream_a):
    #     with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], with_stack=True) as prof:
    #         output_len = 100
    #         for i in range(output_len - 1):
    #             tic = time.time()

    #             # # print("1000000000 decode")
    #             # batch.count_time = True
    #             # for ii in range(10):
    #             #     _, _ = decode(next_token_ids, batch, model_runner)
    #             # # print("after 1000000000 decode")
    #             # synchronize(device)
    #             # latency = time.time() - tic
    #             # latency = latency / 10
    #             # tot_latency += latency
    #             # throughput = batch_size / latency

    #             # batch.count_time = False
    #             next_token_ids, _, forward_latency = decode(next_token_ids, batch, model_runner, device, stream_a)

    #             latency = forward_latency
    #             # latency = time.time() - tic
    #             # latency = latency / 10
    #             tot_latency += latency
    #             throughput = batch_size / latency

    #             decode_latencies.append(latency)

    #             if i > 0 and i % 1 == 0:
    #                 avg_latency = sum(decode_latencies[-8:]) / 8
    #                 rank_print(
    #                         f"Decode. i:{i},  latency: {avg_latency:6.5f} ms"
    #                     )
    #     prof.export_chrome_trace(f"/workspace/sglang/test/llama_factory/colocation_overlap_trace.json")

    # stream_a = native_stream

    # with torch.cuda.stream(stream_a):
    #     for i in range(output_len - 1):
    #         next_token_ids, _, forward_latency = decode(next_token_ids, batch, model_runner, device, stream_a)

    #         latency = forward_latency
    #         tot_latency += latency
    #         throughput = batch_size / latency

    #         decode_latencies.append(latency)
    #         if i > 0 and i % 8 == 0:
    #             avg_latency = sum(decode_latencies[-8:]) / 8
    #             rank_print(
    #                     f"Decode. i:{i},  latency: {avg_latency:6.5f} ms"
    #                 )

    # rank_print(f"After Decode. GPU memory used: {get_gpu_memory(device):.2f} MB")

    # Record decode timing from 2nd output
    if output_len > 1:
        avg_decode_latency = np.mean(decode_latencies)*1000
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  mean latency: {avg_decode_latency:6.5f} ms, median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    # rank_print(f"Total. GPU memory used: {get_gpu_memory(device):.2f} MB")

    # 在返回前强制终止微调线程
    if finetune_thread and finetune_thread.is_alive():
        import ctypes
        try:
            thread_id = finetune_thread.ident
            # 向目标线程抛出SystemExit异常
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id),
                ctypes.py_object(SystemExit)
            )
        except Exception as e:
            rank_print(f"终止线程失败: {e}")
            
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Warm up
    # rank_print("Warmup ...")
    # latency_test_run_once(
    #     bench_args.run_name,
    #     model_runner,
    #     rank_print,
    #     reqs,
    #     bench_args.batch_size[0],
    #     bench_args.input_len[0],
    #     8,  # shorter decoding to speed up the warmup
    #     server_args.device,
    # )
    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        rank_print(f"batch_size: {bs}, input_len: {il}, output_len: {ol}")
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
        )
        if ret is not None:
            result_list.append(ret)
        print("\n")

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")


def main(server_args, bench_args):
    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    print(server_args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    # # test finetune
    # port_args = PortArgs.init_new(server_args)
    # model_runner, tokenizer = load_model(server_args, port_args, 0)

    # print("model_runner.finetune_model.base_model.model.model.pause_train: ", model_runner.finetune_model.base_model.model.model.pause_train)

    # input_thread = threading.Thread(
    #     target=file_listener,
    #     args=(model_runner,),
    #     daemon=True
    # )
    # input_thread.start()

    # model_runner.finetune_train()

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)

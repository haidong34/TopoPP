"""CUDA-event timing for vLLM 0.10.1.1 pipeline stages.

Events bracket model_runner.execute_model after receive and before send.
Use this worker during calibration; final A/B runs use the stock worker.
"""

import atexit
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import vllm
from vllm.distributed import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.v1.worker.gpu_worker import Worker


class TopoPPWorker(Worker):
    def load_model(self, *args, **kwargs):
        parallel = self.vllm_config.parallel_config
        if vllm.__version__ != "0.10.1.1":
            raise RuntimeError("TopoPPWorker is restricted to vLLM 0.10.1.1")
        if (
            parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.distributed_executor_backend != "mp"
        ):
            raise RuntimeError(
                "TopoPPWorker requires TP=1, DP=1 and --distributed-executor-backend mp"
            )
        if (
            self.model_config.hf_config.model_type != "qwen3"
            or self.vllm_config.speculative_config
        ):
            raise RuntimeError("Dense Qwen3 without speculative decoding required")
        result = super().load_model(*args, **kwargs)
        trace_dir = Path(os.environ["TOPOPP_TRACE_DIR"])
        state = os.environ["TOPOPP_PROFILE_STATE"]
        if state not in ("H0", "H1"):
            raise ValueError("TOPOPP_PROFILE_STATE must be H0 or H1")
        run_id = os.environ["TOPOPP_RUN_ID"]
        every = int(os.getenv("TOPOPP_SAMPLE_EVERY", "1"))
        if every < 1:
            raise ValueError("TOPOPP_SAMPLE_EVERY must be positive")
        trace_dir.mkdir(parents=True, exist_ok=True)
        rank = get_pp_group().rank_in_group
        layers = self.model_config.hf_config.num_hidden_layers
        begin, end = get_pp_indices(layers, rank, parallel.pipeline_parallel_size)
        # Config hash is shared across workers; no prompt content is logged.
        cfg_hash = hashlib.sha256(
            self.model_config.hf_config.to_json_string().encode()
        ).hexdigest()
        config = self.model_config.hf_config
        compilation = self.vllm_config.compilation_config
        scheduler = self.vllm_config.scheduler_config
        base = {
            "schema_version": 1,
            "run_id": run_id,
            "profile_state": state,
            "pp_rank": rank,
            "pp_size": parallel.pipeline_parallel_size,
            "gpu_id": self.device.index,
            "num_layers": layers,
            "stage_layers": end - begin,
            "config_sha256": cfg_hash,
            "enforce_eager": self.model_config.enforce_eager,
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "model_geometry": {
                "layers": layers,
                "hidden_size": config.hidden_size,
                "attention_heads": config.num_attention_heads,
                "kv_heads": config.num_key_value_heads,
                "head_dim": getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads,
                "vocab_size": config.vocab_size,
            },
            "execution_signature": {
                "cudagraph_mode": str(compilation.cudagraph_mode),
                "capture_sizes": compilation.cudagraph_capture_sizes,
                "max_batched_tokens": scheduler.max_num_batched_tokens,
                "max_seqs": scheduler.max_num_seqs,
            },
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        }
        path = trace_dir / f"rank{rank}.pid{os.getpid()}.jsonl"
        pending = deque()
        counter = 0

        def drain(force=False):
            lines = []
            while pending and (force or pending[0][1].query()):
                start_event, end_event, metadata = pending.popleft()
                if force:
                    end_event.synchronize()
                metadata["local_cuda_ms"] = start_event.elapsed_time(end_event)
                lines.append(json.dumps(metadata, allow_nan=False))
            if lines:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")

        atexit.register(lambda: drain(force=True))
        runner = self.model_runner
        original = runner.execute_model

        def measured_execute(scheduler_output, intermediate_tensors=None):
            nonlocal counter
            drain()
            enabled_file = os.getenv("TOPOPP_ENABLE_FILE")
            enabled = not enabled_file or Path(enabled_file).exists()
            counter += 1
            sample = (
                enabled
                and counter % every == 0
                and scheduler_output.total_num_scheduled_tokens > 0
            )
            if not sample:
                return original(scheduler_output, intermediate_tensors)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start_event.record()
            clock = time.perf_counter()
            output = original(scheduler_output, intermediate_tensors)
            wall_ms = (time.perf_counter() - clock) * 1000
            end_event.record()
            qs, contexts, prefills = [], [], []
            for req_id, q in scheduler_output.num_scheduled_tokens.items():
                req = runner.requests[req_id]
                qs.append(int(q))
                contexts.append(int(req.num_computed_tokens))
                prefills.append(req.num_computed_tokens < len(req.prompt_token_ids))
            phase = "prefill" if all(prefills) else "decode" if not any(prefills) else "mixed"
            metadata = {
                **base,
                "step": counter,
                "timestamp": time.time(),
                "sample_every": every,
                "phase": phase,
                "batch_size": len(qs),
                "query_tokens": qs,
                "context_lengths": contexts,
                "intermediate_shapes": [
                    list(t.shape) for t in getattr(output, "tensors", {}).values()
                ],
                "local_wall_ms": wall_ms,
            }
            pending.append((start_event, end_event, metadata))
            # Bound event retention without introducing forced synchronization.
            if len(pending) > 4096:
                raise RuntimeError(
                    "GPU events are not completing; stop trace collection and inspect stream execution"
                )
            drain()
            return output

        runner.execute_model = measured_execute
        return result

    def determine_available_memory(self):
        available = super().determine_available_memory()
        self._topopp_available_kv_bytes = int(available)
        return available

    def initialize_from_config(self, kv_cache_config):
        result = super().initialize_from_config(kv_cache_config)
        trace_dir = Path(os.environ["TOPOPP_TRACE_DIR"])
        rank = get_pp_group().rank_in_group
        record = {
            "pp_rank": rank,
            "gpu_id": self.device.index,
            "kv_blocks": int(kv_cache_config.num_blocks),
            "available_kv_bytes": getattr(self, "_topopp_available_kv_bytes", None),
            "model_weights_bytes": int(self.model_runner.model_memory_usage),
            "block_size": int(self.vllm_config.cache_config.block_size),
        }
        path = trace_dir / f"rank{rank}.admission.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return result

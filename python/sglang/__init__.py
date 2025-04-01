# SGL API Components

from sglang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

# SGLang DSL APIs
__all__ = [
    "Runtime",
    "Engine",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
]

# Global Configurations
from sglang.global_config import global_config

__all__ += ["global_config"]

from sglang.version import __version__

__all__ += ["__version__"]

# SGLang Backends
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import LazyImport

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

__all__ += ["Anthropic", "LiteLLM", "OpenAI", "VertexAI", "RuntimeEndpoint"]

import os
import torch
import freeslots

script_dir = os.path.dirname(os.path.abspath(__file__))
alloc_path = os.path.join(script_dir, '/workspace/freeslots/python/freeslots/_C.cpython-310-x86_64-linux-gnu.so')
# alloc_path = os.path.join(script_dir, '/workspace/my_allocators/tutorial/alloc.so')

new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    alloc_path, 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)
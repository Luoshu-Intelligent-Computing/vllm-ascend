#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# PATCH for 310P long context (nightly-main-310p)
# 1. Fixes instance-variable cache bug (self.xxx → cls.xxx)
# 2. get_attention_mask() returns None to skip O(L²) preallocation
#

import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d, nd_to_nz_spec

COMPRESSED_MASK_SEQ_LEN = 4096
PAGED_ATTENTION_COMPRESSED_MASK_VALUE = -10000.0


def is_compressed_mask_supported() -> bool:
    return hasattr(torch_npu, "_npu_flash_attention_v3") and hasattr(torch_npu, "_npu_paged_attention_splitfuse_v2")


class AttentionMaskBuilder310:
    chunked_prefill_attn_mask = None
    compressed_chunked_prefill_attn_mask = None
    causal_attn_mask_cache = None  # FIX: changed from instance to class variable
    non_causal_attn_mask_cache = None  # FIX: changed from instance to class variable
    max_seqlen = 16384

    def __init__(self, device: torch.device, max_seqlen: int):
        """
        Initializes the AttentionMaskBuilder for the 310P device.

        Args:
            device (torch.device): The device on which tensors will be allocated.
            max_seqlen (int): Maximum length of a sequence (including prompt and generated text).
        """
        AttentionMaskBuilder310.max_seqlen = max_seqlen
        # FIX: removed self.causal_attn_mask_cache and self.non_causal_attn_mask_cache
        # (now class variables to avoid per-layer duplication)
        self.support_compressed_mask = is_compressed_mask_supported()
        self.device = device

    @staticmethod
    def gen_causal_additive_mask(max_seq_len: int, device: torch.device):
        """
        Generates a standard causal lower-triangular attention mask.

        The upper triangular part is filled with negative infinity (float("-inf"))
        to mask out future tokens, while the lower triangular part is kept as 0.

        Args:
            max_seq_len (int): The maximum sequence length for the mask.
            device (torch.device): The target device for the tensor.

        Returns:
            torch.Tensor: A float16 tensor representing the causal mask.
        """
        tril = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device).tril_()
        upper = ~tril
        mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
        mask.masked_fill_(upper, float("-inf"))
        return mask

    @classmethod
    def get_splitfuse_mask(cls, attn_metadata: AscendMetadata, device: torch.device):
        """
        Generates and formats the attention mask for SplitFuse (chunked prefill) decoding.

        **OOM FIX for 128K context**: Uses on-the-fly broadcasting instead of pre-allocating
        a [max_seqlen, max_seqlen] full causal matrix (131072² × 2B = 34 GB → OOM).
        Generates [T, max_context_len] where T ≤ max_num_batched_tokens (1024), using only
        the active batch's maximum context width for the temporary allocation.

        Args:
            attn_metadata (AscendMetadata): Metadata containing query start locations and sequence lengths.
            device (torch.device): The device to perform operations on.

        Returns:
            torch.Tensor: The splitfuse attention mask cast to ACL_FORMAT_FRACTAL_NZ.
        """
        qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
        qlens = qsl[1:] - qsl[:-1]
        q_list = qlens.tolist()
        context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)
        c_list = context_lens.tolist()
        max_context_len = max(c_list)
        pos_list = [p for ql, cl in zip(q_list, c_list) for p in range(cl - ql, cl)]
        # On-the-fly generation via broadcasting avoids a full causal matrix and uses
        # only the active batch's maximum context width.
        positions = torch.tensor(pos_list, dtype=torch.int64, device=device)
        col_idx = torch.arange(max_context_len, dtype=torch.int64, device=device)
        mask = torch.where(
            col_idx.unsqueeze(0) > positions.unsqueeze(1),
            torch.full((1, 1), float("-inf"), dtype=torch.float16, device=device),
            torch.zeros(1, 1, dtype=torch.float16, device=device),
        )
        splitfuse_mask_nz = torch_npu.npu_format_cast(nd_to_nz_spec(mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        return splitfuse_mask_nz

    @classmethod
    def get_compressed_splitfuse_mask(cls, device: torch.device):
        """
        Generates the fixed ND attention mask for compressed SplitFuse PA.

        Returns:
            torch.Tensor: A [2048, 2048] float16 ND mask on the target device.
        """
        if (
            cls.compressed_chunked_prefill_attn_mask is None
            or cls.compressed_chunked_prefill_attn_mask.device != device
        ):
            mask = torch.ones(
                size=(COMPRESSED_MASK_SEQ_LEN, COMPRESSED_MASK_SEQ_LEN),
                dtype=torch.float16,
                device=device,
            )
            mask = torch.triu(mask, diagonal=1)
            cls.compressed_chunked_prefill_attn_mask = mask.mul_(PAGED_ATTENTION_COMPRESSED_MASK_VALUE)
        return cls.compressed_chunked_prefill_attn_mask

    def get_attention_mask(self, causal: bool, model_config) -> torch.Tensor:
        """
        **PATCHED**: Use COMPRESSED_MASK_SEQ_LEN (2048) instead of max_model_len (65536)
        to avoid O(L²) OOM. Original would allocate 65536² × 2B = 8 GB.

        310P attention paths:
        - PrefillNoCache: uses this cached [2048, 2048] mask
        - ChunkedPrefill: generates splitfuse mask dynamically
        - Decode: no mask needed
        """
        # FIX: Cap at COMPRESSED_MASK_SEQ_LEN (2048) to avoid 8GB allocation
        max_seq_len = (
            COMPRESSED_MASK_SEQ_LEN if self.support_compressed_mask else min(self.max_seqlen, COMPRESSED_MASK_SEQ_LEN)
        )

        if getattr(model_config, "runner_type", None) == "pooling":
            if causal:
                return self._get_causal_mask(max_seq_len)
            else:
                return self._get_non_causal_mask(max_seq_len, model_config.dtype)

        return self._get_causal_mask(max_seq_len)

    def _get_causal_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        Internal method to get or update the cached causal attention mask.

        **FIX**: Now uses class variable cls.causal_attn_mask_cache instead of
        instance variable to avoid per-layer duplication.
        """
        if AttentionMaskBuilder310.causal_attn_mask_cache is None:
            attn_mask = self.gen_causal_additive_mask(max_seq_len, self.device)
            AttentionMaskBuilder310.causal_attn_mask_cache = torch_npu.npu_format_cast(
                nd_to_nz_2d(attn_mask), ACL_FORMAT_FRACTAL_NZ
            )
        return AttentionMaskBuilder310.causal_attn_mask_cache

    def _get_non_causal_mask(self, max_seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Internal method to get or update the cached non-causal attention mask.

        **FIX**: Now uses class variable cls.non_causal_attn_mask_cache instead of
        instance variable to avoid per-layer duplication.
        """
        if AttentionMaskBuilder310.non_causal_attn_mask_cache is not None:
            return AttentionMaskBuilder310.non_causal_attn_mask_cache

        attention_mask_npu = torch.zeros(
            size=(max_seq_len, max_seq_len),
            dtype=dtype,
            device=self.device,
        )
        attention_mask_npu = nd_to_nz_2d(attention_mask_npu)
        AttentionMaskBuilder310.non_causal_attn_mask_cache = torch_npu.npu_format_cast(
            attention_mask_npu.contiguous(), ACL_FORMAT_FRACTAL_NZ
        )

        return AttentionMaskBuilder310.non_causal_attn_mask_cache

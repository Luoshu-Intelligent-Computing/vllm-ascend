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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310


class TestAttentionMaskBuilder310(TestBase):
    def setUp(self):
        self.max_seqlen = 4096
        self.attention_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), self.max_seqlen)

    @patch("torch_npu.npu_format_cast")
    def test_get_attention_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        self.attention_mask_builder.support_compressed_mask = False
        model_config = MagicMock()
        attn_mask = self.attention_mask_builder.get_attention_mask(causal=True, model_config=model_config)
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, self.max_seqlen, 16))
        self.assertEqual(attn_mask[0][-1][0][-1], torch.tensor(float("-inf"), dtype=torch.float16))

    @patch("torch_npu.npu_format_cast")
    def test_get_splitfuse_attn_mask_uses_live_max_context(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        attn_metadata = MagicMock()
        attn_metadata.query_start_loc = torch.tensor([0, 1, 5])
        attn_metadata.seq_lens = torch.tensor([7, 4])

        attn_mask = self.attention_mask_builder.get_splitfuse_mask(attn_metadata, torch.device("cpu"))

        self.assertEqual(attn_mask.shape, (1, 1, 16, 16))

    @patch("torch_npu.npu_format_cast")
    def test_get_splitfuse_attn_mask_preserves_causality_for_non_aligned_width(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        attn_metadata = MagicMock()
        attn_metadata.query_start_loc = torch.tensor([0, 1, 3])
        attn_metadata.seq_lens = torch.tensor([33, 5])

        attn_mask = self.attention_mask_builder.get_splitfuse_mask(attn_metadata, torch.device("cpu"))
        mask_nd = attn_mask.permute(0, 2, 1, 3).reshape(16, 48)
        expected_mask = torch.zeros((3, 33), dtype=torch.float16)
        expected_mask[1, 4:] = float("-inf")
        expected_mask[2, 5:] = float("-inf")

        self.assertEqual(attn_mask.shape, (1, 3, 16, 16))
        self.assertTrue(torch.equal(mask_nd[:3, :33], expected_mask))
        self.assertTrue(torch.equal(mask_nd[:3, 33:], torch.zeros((3, 15), dtype=torch.float16)))

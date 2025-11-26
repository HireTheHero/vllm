# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test hidden states extraction during text generation for Qwen3-0.6B model."""

import pytest
import torch

from vllm import SamplingParams


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("max_tokens", [10, 20])
@torch.inference_mode
def test_qwen3_hidden_states_all_layers(vllm_runner, model: str,
                                        max_tokens: int):
    """Test that hidden states from all layers are correctly extracted."""
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Write a short poem.",
    ]

    with vllm_runner(
            model,
            max_model_len=512,
            enforce_eager=True,
            enable_chunked_prefill=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            output_hidden_states=True,
        )

        outputs = vllm_model.llm.generate(prompts, sampling_params)

        for output in outputs:
            assert output.outputs[0].hidden_states is not None, \
                "Hidden states should be returned when output_hidden_states=True"

            hidden_states = output.outputs[0].hidden_states
            assert isinstance(hidden_states, list), \
                "Hidden states should be a list of tensors"
            assert len(hidden_states) > 0, \
                "Hidden states list should not be empty"

            # Qwen3-0.6B has 28 transformer layers
            expected_num_layers = 28
            assert len(hidden_states) == expected_num_layers, \
                f"Expected {expected_num_layers} layers, got {len(hidden_states)}"

            # Check each layer
            for layer_idx, hidden_state in enumerate(hidden_states):
                assert isinstance(hidden_state, torch.Tensor), \
                    f"Layer {layer_idx} should be a torch.Tensor"
                assert len(hidden_state.shape) == 2, \
                    f"Layer {layer_idx} should have shape [seq_len, hidden_size], got {hidden_state.shape}"
                assert hidden_state.shape[0] > 0, \
                    f"Layer {layer_idx} should have non-zero sequence length"

                # Qwen3-0.6B has hidden_size = 896
                expected_hidden_size = 896
                assert hidden_state.shape[1] == expected_hidden_size, \
                    f"Layer {layer_idx} should have hidden_size={expected_hidden_size}, got {hidden_state.shape[1]}"

                # Sanity check: no NaN or Inf values
                assert not torch.isnan(hidden_state).any(), \
                    f"Layer {layer_idx} contains NaN values"
                assert not torch.isinf(hidden_state).any(), \
                    f"Layer {layer_idx} contains Inf values"

            # Verify all layers have the same sequence length
            seq_lengths = [h.shape[0] for h in hidden_states]
            assert len(set(seq_lengths)) == 1, \
                f"All layers should have the same sequence length, got {seq_lengths}"


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("layer_indices", [
    [0],  # First layer only
    [27],  # Last layer only
    [0, 13, 27],  # First, middle, and last
    [5, 10, 15, 20],  # Multiple middle layers
])
@torch.inference_mode
def test_qwen3_hidden_states_specific_layers(vllm_runner, model: str,
                                              layer_indices: list[int]):
    """Test that specific layer indices are correctly extracted."""
    prompts = ["Hello, how are you?", "What is AI?"]

    with vllm_runner(
            model,
            max_model_len=512,
            enforce_eager=True,
            enable_chunked_prefill=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=15,
            output_hidden_states=True,
            hidden_state_layers=layer_indices,
        )

        outputs = vllm_model.llm.generate(prompts, sampling_params)

        for output in outputs:
            hidden_states = output.outputs[0].hidden_states
            assert hidden_states is not None, \
                "Hidden states should be returned"

            assert len(hidden_states) == len(layer_indices), \
                f"Expected {len(layer_indices)} layers, got {len(hidden_states)}"

            # Verify each layer
            for hidden_state in hidden_states:
                assert isinstance(hidden_state, torch.Tensor)
                assert len(hidden_state.shape) == 2
                assert hidden_state.shape[1] == 896  # Qwen3-0.6B hidden_size


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@torch.inference_mode
def test_qwen3_hidden_states_disabled(vllm_runner, model: str):
    """Test that hidden states are not returned when disabled (default)."""
    prompts = ["Hello!", "How are you?"]

    with vllm_runner(
            model,
            max_model_len=512,
            enforce_eager=True,
            enable_chunked_prefill=False,
    ) as vllm_model:
        # Default: output_hidden_states=False
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
        )

        outputs = vllm_model.llm.generate(prompts, sampling_params)

        for output in outputs:
            assert output.outputs[0].hidden_states is None, \
                "Hidden states should be None when output_hidden_states=False"


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@torch.inference_mode
def test_qwen3_hidden_states_shape_consistency(vllm_runner, model: str):
    """Test that hidden states have consistent shapes across different prompts."""
    # Prompts of varying lengths
    prompts = [
        "Hi",  # Short
        "What is the meaning of life?",  # Medium
        "Please explain quantum mechanics in simple terms.",  # Longer
    ]

    with vllm_runner(
            model,
            max_model_len=512,
            enforce_eager=True,
            enable_chunked_prefill=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=15,
            output_hidden_states=True,
        )

        outputs = vllm_model.llm.generate(prompts, sampling_params)

        # Check that all outputs have hidden states
        assert all(output.outputs[0].hidden_states is not None
                   for output in outputs)

        # Check that all outputs have the same number of layers
        num_layers_list = [
            len(output.outputs[0].hidden_states) for output in outputs
        ]
        assert len(set(num_layers_list)) == 1, \
            f"All outputs should have the same number of layers, got {num_layers_list}"

        # Check that all layers have the same hidden_size
        for output in outputs:
            hidden_states = output.outputs[0].hidden_states
            hidden_sizes = [h.shape[1] for h in hidden_states]
            assert all(size == 896 for size in hidden_sizes), \
                f"All layers should have hidden_size=896, got {hidden_sizes}"

            # Check that within each output, all layers have consistent seq_len
            seq_lengths = [h.shape[0] for h in hidden_states]
            assert len(set(seq_lengths)) == 1, \
                f"Within each output, all layers should have same seq_len, got {seq_lengths}"


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@torch.inference_mode
def test_qwen3_hidden_states_negative_layer_index(vllm_runner, model: str):
    """Test that negative layer indices work correctly (Python-style indexing)."""
    prompts = ["Hello, world!"]

    with vllm_runner(
            model,
            max_model_len=512,
            enforce_eager=True,
            enable_chunked_prefill=False,
    ) as vllm_model:
        # Test negative indexing: -1 should be the last layer
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            output_hidden_states=True,
            hidden_state_layers=[-1],  # Last layer
        )

        outputs = vllm_model.llm.generate(prompts, sampling_params)

        for output in outputs:
            hidden_states = output.outputs[0].hidden_states
            assert hidden_states is not None
            assert len(hidden_states) == 1, \
                "Should return exactly one layer for [-1]"
            assert hidden_states[0].shape[1] == 896


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@torch.inference_mode
def test_qwen3_hidden_states_multiple_completions(vllm_runner, model: str):
    """Test hidden states extraction with multiple completions (n > 1)."""
    prompts = ["Write a haiku about coding."]

    with vllm_runner(
            model,
            max_model_len=512,
            enforce_eager=True,
            enable_chunked_prefill=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=20,
            n=3,  # Generate 3 completions
            output_hidden_states=True,
            hidden_state_layers=[0, 27],  # First and last layer
        )

        outputs = vllm_model.llm.generate(prompts, sampling_params)

        assert len(outputs) == 1, "Should have one output per prompt"
        assert len(outputs[0].outputs) == 3, \
            "Should have 3 completions for n=3"

        # Check each completion
        for completion_idx, completion in enumerate(outputs[0].outputs):
            hidden_states = completion.hidden_states
            assert hidden_states is not None, \
                f"Completion {completion_idx} should have hidden states"
            assert len(hidden_states) == 2, \
                f"Completion {completion_idx} should have 2 layers"

            for hidden_state in hidden_states:
                assert isinstance(hidden_state, torch.Tensor)
                assert hidden_state.shape[1] == 896

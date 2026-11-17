# Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework
This repository contains the source code and dataset link mentioned in DAC 2023 accepted paper "Faster and Stronger Lossless Compression with Optimized Autoregressive Framework".

## Files

- `v2_mlp_enc_dec.py`: entry point; compresses a file, then decompresses the result.
- `mlp_scalable_bmm.py`: the model (`mlpCompressor`).
- `arithmeticcoding_fast.py`: arithmetic coder.
- `utils.py`: converts decoded tokens to text.
- `test_cpu_smoke.py`: CPU smoke test.

## Install

```
pip install -r requirements.txt
```

`arithmeticcoding_fast.py` calls `np.asscalar`, which NumPy 1.23 removed, so
`requirements.txt` asks for `numpy<1.23`; NumPy 1.22 supports Python 3.8 to 3.10.

## Run

`v2_mlp_enc_dec.py` moves the model and data to the GPU with `.cuda()`, so it needs a CUDA GPU.

```
python v2_mlp_enc_dec.py --input_dir <input file> --prefix <name> --gpu_id <GPU id>
```

One run compresses the input file and then decompresses it again. It writes into the current directory:

- `<prefix>_<vocab_dim>_<hidden_dim>_<ffn_dim>_bs<batch_size>_<n_layers>_seq<seq_len>.compressed.combined`: the compressed file.
- `decompressed_out`: the decompressed data.
- `<prefix>_..._temp/`: the per-stream files. The run stops at start if this directory already exists, so remove it before running again with the same flags.

Flags (defaults from `v2_mlp_enc_dec.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input_dir` | `aaa` | Path of the input file. |
| `--prefix` | `text8` | Prefix of the output names. |
| `--gpu_id` | `1` | Value of `CUDA_VISIBLE_DEVICES`. |
| `--batch_size` | `512` | Number of streams the input is split into and coded in parallel. |
| `--learning_rate` | `1e-3` | Adam learning rate. |
| `--hidden_dim` | `256` | Feature dimension. |
| `--vocab_dim` | `16` | Embedding dimension. |
| `--seq_len` | `1` | Context length; multiplied by `hidden_dim // vocab_dim` at start (16 with the defaults). |
| `--vocab_size` | `256` | Vocabulary size of the data. |
| `--random_seed` | `0` | Seed for NumPy and Torch. |
| `--print_step` | `1000` | Interval to print metrics. |
| `--ffn_dim`, `--n_layers` | `4096`, `1` | Used only in the output names. |
| `--n_heads`, `--feature_type`, `--compute_type`, `--weight_decay` | `8`, `sqr`, `iter`, `0.0` | Accepted, without effect. |

## Test

```
python -m unittest -v test_cpu_smoke
```

Compresses and decompresses two small inputs with `--batch_size 8` (2021 bytes of generated text, and 1029 bytes containing every byte value) and checks that `decompressed_out` equals the input. The test replaces `.cuda()` with a no-op in the process it starts, so it runs without a GPU.

## Limitations

- The compressed stream depends on the exact torch build and hardware: the decoder has to reproduce the encoder's model outputs bit for bit. A `.combined` file therefore decodes only in the environment that wrote it.
- One run both compresses and decompresses. There is no separate decompression command: the decoder takes the input length from the input file read in the same run.

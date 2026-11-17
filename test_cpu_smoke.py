"""CPU smoke test: compress and decompress small inputs and compare with the input.

v2_mlp_enc_dec.py moves the model and every batch to the GPU with hard-coded
.cuda() calls. The test runs the unmodified script in a subprocess where
.cuda() is a no-op, so it needs no GPU and the script keeps its GPU-only path.

Run with:  python -m unittest -v test_cpu_smoke
"""
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))

RUN_ON_CPU = """
import runpy, sys, torch
torch.nn.Module.cuda = lambda self, *args, **kwargs: self
torch.Tensor.cuda = lambda self, *args, **kwargs: self
sys.path.insert(0, sys.argv[1])
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""

BATCH_SIZE = 8
# seq_len is 16 with the default --hidden_dim/--vocab_dim. An input is
# batch-aligned when (length - 16) is a multiple of BATCH_SIZE: then the
# batched streams cover everything except the final 16 bytes.
TEXT_LENGTH = 2021
ALIGNED_TEXT_LENGTH = 2032
ALL_BYTE_VALUES = bytes(range(256)) * 4 + bytes(range(5))

WORDS = (
    "the of and one in a to zero nine two is as eight for s five three was by "
    "that four six seven with on are it from or his an be this which at he "
    "also not have were has but other their its first they some had all more "
    "most can been such many who new used after time"
).split()


def make_text(length):
  """Deterministic lower-case text (a-z and space, like text8)."""
  state = 0
  words = []
  size = 0
  while size < length:
    state = (state * 1103515245 + 12345) % (1 << 31)
    word = WORDS[(state >> 16) % len(WORDS)]
    words.append(word)
    size += len(word) + 1
  return " ".join(words)[:length].encode("ascii")


class CpuSmokeTest(unittest.TestCase):

  def round_trip(self, data):
    """Runs the script on data and returns the content of decompressed_out."""
    with tempfile.TemporaryDirectory() as work_dir:
      input_file = os.path.join(work_dir, "input.bin")
      with open(input_file, "wb") as f:
        f.write(data)

      # The script writes its outputs into the current directory.
      result = subprocess.run(
          [sys.executable, "-c", RUN_ON_CPU, REPO,
           os.path.join(REPO, "v2_mlp_enc_dec.py"),
           "--input_dir", input_file, "--prefix", "smoke",
           "--batch_size", str(BATCH_SIZE)],
          cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      self.assertEqual(result.returncode, 0, result.stdout.decode(errors="replace"))

      compressed = os.path.join(
          work_dir, "smoke_16_256_4096_bs8_1_seq1.compressed.combined")
      self.assertTrue(os.path.isfile(compressed))

      with open(os.path.join(work_dir, "decompressed_out"), "rb") as f:
        return f.read()

  def test_round_trip(self):
    data = make_text(TEXT_LENGTH)
    self.assertEqual(self.round_trip(data), data)

  def test_round_trip_batch_aligned_length(self):
    self.assertEqual((ALIGNED_TEXT_LENGTH - 16) % BATCH_SIZE, 0)
    data = make_text(ALIGNED_TEXT_LENGTH)
    self.assertEqual(self.round_trip(data), data)

  def test_round_trip_all_byte_values(self):
    self.assertEqual(set(ALL_BYTE_VALUES), set(range(256)))
    self.assertEqual(self.round_trip(ALL_BYTE_VALUES), ALL_BYTE_VALUES)


if __name__ == "__main__":
  unittest.main()

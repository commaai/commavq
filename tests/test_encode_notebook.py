import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import video as video_utils
from utils import vqvae

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "encode.ipynb"


def load_code_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class EncodeNotebookSmokeTest(unittest.TestCase):
    def test_encode_notebook_runs_end_to_end_with_repo_utils(self):
        cells = load_code_cells()
        self.assertEqual(len(cells), 5, "encode notebook layout changed unexpectedly")

        state = {
            "read_video_path": None,
            "loaded_url": None,
            "load_assign": None,
            "encoder_device": None,
            "encoder_batch_shapes": [],
            "saved_path": None,
            "saved_array": None,
        }
        original_sys_path = list(sys.path)

        def fake_read_video(path):
            state["read_video_path"] = path
            return np.stack(
                [
                    np.full((720, 1280, 3), fill_value=index * 20, dtype=np.uint8)
                    for index in range(3)
                ],
                axis=0,
            )

        def fake_load_state_dict_from_url(self, url, assign=True):
            state["loaded_url"] = url
            state["load_assign"] = assign

        def fake_encoder_to(self, device=None, **_kwargs):
            state["encoder_device"] = device
            return self

        def fake_encoder_forward(self, batch):
            state["encoder_batch_shapes"].append(tuple(batch.shape))
            return torch.arange(128, dtype=torch.int64).reshape(1, 128)

        def fake_from_numpy(array):
            return torch.tensor(array.tolist())

        def fake_tensor_to(self, *args, **kwargs):
            return self

        def fake_tensor_numpy(self):
            return np.array(self.tolist())

        def fake_save(path, array):
            state["saved_path"] = path
            state["saved_array"] = np.array(array)

        namespace = {"__name__": "__main__"}

        try:
            with (
                patch.object(video_utils, "read_video", side_effect=fake_read_video),
                patch.object(vqvae.Encoder, "load_state_dict_from_url", fake_load_state_dict_from_url),
                patch.object(vqvae.Encoder, "to", fake_encoder_to),
                patch.object(vqvae.Encoder, "forward", fake_encoder_forward),
                patch.object(torch, "from_numpy", side_effect=fake_from_numpy),
                patch.object(torch.Tensor, "to", fake_tensor_to),
                patch.object(torch.Tensor, "numpy", fake_tensor_numpy),
                patch.object(np, "save", side_effect=fake_save),
            ):
                for index, cell_source in enumerate(cells):
                    exec(
                        compile(
                            cell_source,
                            f"{NOTEBOOK_PATH.name}#cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
        finally:
            sys.path[:] = original_sys_path

        self.assertEqual(state["read_video_path"], "../examples/sample_video_ecamera.hevc")
        self.assertIsInstance(namespace["config"], vqvae.CompressorConfig)
        self.assertIsInstance(namespace["encoder"], vqvae.Encoder)
        self.assertEqual(
            state["loaded_url"],
            "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin",
        )
        self.assertTrue(state["load_assign"])
        self.assertEqual(state["encoder_device"], "cuda")
        self.assertEqual(state["encoder_batch_shapes"], [(1, 3, 128, 256)] * 3)
        self.assertEqual(state["saved_path"], "../examples/tokens.npy")
        self.assertIsNotNone(state["saved_array"])
        self.assertEqual(state["saved_array"].shape, (3, 128))
        np.testing.assert_array_equal(state["saved_array"][0], np.arange(128))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.config import PROJECT_ROOT, resolve_project_path


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _update_model_init(init_path: Path, model_module: str, class_name: str) -> None:
    init_text = init_path.read_text(encoding="utf-8") if init_path.exists() else ""
    export_line = f"from src.models.{model_module} import {class_name}\n"
    if export_line not in init_text:
        init_text = export_line + init_text
    if "__all__ = [" in init_text and f'"{class_name}"' not in init_text:
        init_text = init_text.replace("__all__ = [", f'__all__ = ["{class_name}", ')
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text(init_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold a beginner-friendly custom model experiment.")
    parser.add_argument("--type", required=True, help="New experiment type, for example my_custom_ae.")
    parser.add_argument("--class-name", required=True, help="New model class name, for example MyCustomAE.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Project root. Mainly useful for tests.")
    return parser


def scaffold_model(root: Path, experiment_type: str, class_name: str) -> list[Path]:
    model_module = _snake_case(class_name)
    model_path = root / "src" / "models" / f"{model_module}.py"
    init_path = root / "src" / "models" / "__init__.py"
    test_path = root / "tests" / f"test_{model_module}.py"
    note_path = root / "src" / "experiments" / f"{experiment_type}_registration.md"

    _write_new(
        model_path,
        f"""from __future__ import annotations

from torch import nn


class {class_name}(nn.Module):
    \"\"\"TODO: replace this minimal autoencoder with your custom architecture.\"\"\"

    def __init__(self, n_features: int, enc_channels: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, enc_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(enc_channels[0], n_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
""",
    )

    _update_model_init(init_path, model_module, class_name)

    _write_new(
        test_path,
        f"""import torch

from src.models.{model_module} import {class_name}


def test_{model_module}_shape_matches_input():
    model = {class_name}(n_features=5, enc_channels=(8, 4))
    x = torch.randn(2, 5, 32)

    y = model(x)

    assert y.shape == x.shape
""",
    )

    _write_new(
        note_path,
        f"""# Register `{experiment_type}`

TODO:

1. Add `{experiment_type}` to `src/experiments/types.py` if it should be accepted by validation.
2. Add runner dispatch in `src/experiments/runner.py` to build `{class_name}`.
3. Add a config entry in `config/experiments.yaml`.
4. Run:

```powershell
python -m pytest tests/test_{model_module}.py -q
python -m src.experiments.validate
python -m src.experiments.run --name your_experiment_name --dry-run
```
""",
    )
    return [model_path, init_path, test_path, note_path]


def main() -> None:
    args = build_parser().parse_args()
    root = resolve_project_path(args.root)
    written = scaffold_model(root, args.type, args.class_name)
    for path in written:
        print(f"wrote -> {path}")


if __name__ == "__main__":
    main()

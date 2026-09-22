# Copyright 2026 Google LLC
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

"""Pre-flight validation for .litertlm model checkpoints."""

import dataclasses
import os
import struct

MAGIC = b"LITERTLM"
SIZE_OFFSET = 88
_HEADER_READ = 128


@dataclasses.dataclass(frozen=True)
class CheckpointStatus:
  """Validation status for a local .litertlm checkpoint file."""

  path: str
  actual_size: int
  declared_size: int | None
  problem: str | None

  @property
  def ok(self) -> bool:
    return self.problem is None

  @property
  def percent(self) -> float | None:
    if not self.declared_size:
      return None
    return 100.0 * self.actual_size / self.declared_size


def inspect(path: str) -> CheckpointStatus:
  """Verifies header magic bytes and file size without loading model weights."""
  if not os.path.exists(path):
    return CheckpointStatus(path, 0, None, "file does not exist")

  actual = os.path.getsize(path)

  with open(path, "rb") as f:
    header = f.read(_HEADER_READ)

  if not header.startswith(MAGIC):
    return CheckpointStatus(
        path,
        actual,
        None,
        f"not a .litertlm file (magic is {header[:8]!r}, expected {MAGIC!r})",
    )

  if len(header) < SIZE_OFFSET + 8:
    return CheckpointStatus(
        path,
        actual,
        None,
        f"file is only {actual} bytes — too short to contain a header",
    )

  declared = struct.unpack_from("<Q", header, SIZE_OFFSET)[0]

  if not 1 << 20 <= declared <= 1 << 41:
    return CheckpointStatus(
        path,
        actual,
        None,
        None if actual > 1 << 20 else "file is implausibly small",
    )

  if actual < declared:
    return CheckpointStatus(
        path,
        actual,
        declared,
        f"truncated: {actual:,} of {declared:,} bytes "
        f"({100.0 * actual / declared:.1f}%)",
    )

  if actual > declared:
    return CheckpointStatus(
        path,
        actual,
        declared,
        f"larger than declared: {actual:,} vs {declared:,} bytes",
    )

  return CheckpointStatus(path, actual, declared, None)


def require_valid(path: str) -> None:
  """Raises SystemExit if the checkpoint is missing or incomplete."""
  status = inspect(path)
  if status.ok:
    return

  lines = [
      "",
      "Checkpoint failed pre-flight validation.",
      f"  path    : {status.path}",
      f"  problem : {status.problem}",
  ]
  if status.declared_size:
    lines += [
        "",
        "Resume the download with:",
        "  python3 tools/fetch_model.py --model 26b",
    ]
  lines.append("")
  raise SystemExit("\n".join(lines))

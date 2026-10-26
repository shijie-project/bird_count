"""Settings of the labelling tools: the Label Studio server and where its files live."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr, field_validator

from birdcount.core.settings import CoreSettings, OptionalProjectPath


class LabelingSettings(CoreSettings):
    label_studio_url: str = "http://localhost:8080"
    # Account & Settings > Access Token (a personal access token, or a legacy token).
    label_studio_api_key: SecretStr = SecretStr("")
    label_studio_project_id: int | None = None
    # Label Studio's LOCAL_FILES_DOCUMENT_ROOT; blank = DATA_ROOT.
    label_studio_files_root: OptionalProjectPath = None

    @field_validator("label_studio_project_id", mode="before")
    @classmethod
    def _blank_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def files_root(self) -> Path:
        return self.label_studio_files_root or self.data_root

    @property
    def images_dir(self) -> Path:
        """Every dataset image; the split copies from here."""
        return self.data_root / "images" / "all"

    @property
    def annotations_dir(self) -> Path:
        return self.data_root / "annotations"

    def image_prefix(self, folder: Path) -> str:
        """How LS spells `folder` in a local-files URL: its path under the files root,
        with this machine's separator (LS on Windows writes backslashes)."""
        try:
            relative = folder.resolve().relative_to(self.files_root.resolve())
        except ValueError:
            raise ValueError(
                f"{folder} is not under Label Studio's files root {self.files_root} (LABEL_STUDIO_FILES_ROOT)"
            ) from None
        return os.sep.join(relative.parts) + os.sep if relative.parts else ""

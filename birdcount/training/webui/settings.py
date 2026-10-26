"""Settings of the control panel: how it starts Label Studio and its ngrok tunnel."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from birdcount.core.settings import OptionalProjectPath
from birdcount.training.labeling.settings import LabelingSettings


class WebuiSettings(LabelingSettings):
    # Label Studio's --data-dir (its database and uploads); blank = DATA_ROOT.
    label_studio_data_dir: OptionalProjectPath = None
    # The ngrok domain the public link goes through; blank = no public link.
    label_studio_public_url: str = ""

    @property
    def ls_data_dir(self) -> Path:
        return self.label_studio_data_dir or self.data_root

    @property
    def ls_port(self) -> int:
        return urlsplit(self.label_studio_url).port or 8080

    @property
    def public_domain(self) -> str:
        """The public domain, bare: what ngrok's --url and Label Studio's host want."""
        return self.label_studio_public_url.strip().removeprefix("https://").removeprefix("http://").rstrip("/")

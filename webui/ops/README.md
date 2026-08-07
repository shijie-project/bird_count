# webui/ops — the scripts the web UI drives from the Annotations tab

One script per operation, sitting next to the UI that runs them so both are in
one place. Each file is a complete, runnable CLI; the form in the browser is
generated from its `argparse` spec (see [`../schema.py`](../schema.py)).

| script                     | what it does                                                             |
| -------------------------- | ------------------------------------------------------------------------ |
| `dedupe_annotations.py`    | keep one annotation per task, delete the duplicates (dry run by default) |
| `import_ls_annotations.py` | remap an external LS export to local images and import its annotations   |
| `density_regions.py`       | split density into counted regions; optionally import them into LS       |
| `_common.py`               | shared Label Studio client + the `--project-id/--url/--api-key` parser   |

```bash
python webui/ops/dedupe_annotations.py --project-id 7            # report only
python webui/ops/dedupe_annotations.py --project-id 7 --apply    # commit
python webui/ops/density_regions.py ../data/raw/images -o ../data/raw/regions
python webui/ops/density_regions.py ../data/raw/images -o ../data/raw/regions \
  --send-to-label-studio --project-id 7 --image-prefix "raw\\images\\"
```

Run them from the project root; `density_regions.py` puts the root on `sys.path`
itself so `datasets`/`models`/`utils` resolve to this project's packages.

The Label Studio target comes from `.env` unless overridden on the command line:

```ini
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=<Account & Settings > Access Token>
LABEL_STUDIO_PROJECT_ID=7
```

The file-based pipeline (exports in, training JSON out) lives elsewhere, in
[`../../tools/annotations/`](../../tools/annotations). Density-region import is
not a separate WebUI operation: tick `--send-to-label-studio` in **Density
regions** and the generated predictions are added directly to the selected
project. `_regions_to_label_studio.py` is its internal conversion helper.
`region_mask_gui.py` is kept beside the WebUI operations but is not registered
as another picker entry.

External annotation import tries exact and LS-hash-normalized image names first,
then a conservative closest-name match. `--fuzzy-threshold` controls the minimum
similarity and `--fuzzy-margin` requires the winner to be clearly better than
the runner-up; uncertain matches are rejected rather than guessed. A matched
project task receives a new annotation through the annotation API. Only missing
images create new tasks; `--existing-only` disables even that fallback.

## Adding an operation

1. Drop a new `<verb>_<noun>.py` in this folder and expose `build_parser()` so
   the web UI can introspect it. For a live-project op, take the shared flags
   from `_common.target_parser()` as a `parents=[...]` entry and get the client
   from `_common.connect()`.
1. Register it in [`../schema.py`](../schema.py) as an `Entrypoint` with
   `pythonpath=_OPS_PATH` (use `_OPS_ROOT_PATH` instead if it imports the
   project's own packages) — the form, the command preview and the run history
   follow from the argparse spec.

Anything that deletes or overwrites work in Label Studio should default to a dry
run and require an explicit `--apply`: there is no undo on that side.

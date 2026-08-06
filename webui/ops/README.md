# webui/ops — the scripts the web UI drives from the Annotations tab

One script per operation, sitting next to the UI that runs them so both are in
one place. Each file is a complete, runnable CLI; the form in the browser is
generated from its `argparse` spec (see [`../schema.py`](../schema.py)).

| script                  | what it does                                                             |
| ----------------------- | ------------------------------------------------------------------------ |
| `dedupe_annotations.py` | keep one annotation per task, delete the duplicates (dry run by default) |
| `density_regions.py`    | split a predicted density map into regions + per-region counts           |
| `_common.py`            | shared Label Studio client + the `--project-id/--url/--api-key` parser   |

```bash
python webui/ops/dedupe_annotations.py --project-id 7            # report only
python webui/ops/dedupe_annotations.py --project-id 7 --apply    # commit
python webui/ops/density_regions.py ../data/raw/images -o ../data/raw/regions
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
[`../../tools/annotations/`](../../tools/annotations) — nothing there talks to a
live project.

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

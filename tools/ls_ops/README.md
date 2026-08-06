# tools/ls_ops — operations on a live Label Studio project

One script per operation. Each file is a complete, runnable CLI that does one
thing to a project **over the API**; nothing here reads or writes export files —
that pipeline lives next door in [`../annotations/`](../annotations).

| script                  | what it does                                                             |
| ----------------------- | ------------------------------------------------------------------------ |
| `dedupe_annotations.py` | keep one annotation per task, delete the duplicates (dry run by default) |
| `_common.py`            | shared client + the `--project-id/--url/--api-key` parent parser         |

```bash
python tools/ls_ops/dedupe_annotations.py --project-id 7            # report only
python tools/ls_ops/dedupe_annotations.py --project-id 7 --apply    # commit
```

The target comes from `.env` unless overridden on the command line:

```ini
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=<Account & Settings > Access Token>
LABEL_STUDIO_PROJECT_ID=7
```

## Adding an operation

1. Drop a new `<verb>_<noun>.py` in this folder. Take the shared flags from
   `_common.target_parser()` as a `parents=[...]` entry, expose `build_parser()`
   so the web UI can introspect it, and get the client from `_common.connect()`.
1. Register it in `webui/schema.py` as an `Entrypoint` with
   `pythonpath=_LS_OPS_PATH` — the form, the command preview and the run history
   follow from the argparse spec.

Anything that deletes or overwrites work in Label Studio should default to a dry
run and require an explicit `--apply`: there is no undo on that side.

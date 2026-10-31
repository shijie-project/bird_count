# Design notes

Things that are easy to break without noticing:

- **The calibration depends on the preprocessing.** Thresholds were fitted to
  counts at 1280×720 with hard-black masks and the density masked before the
  sum. Change `IMAGE_WIDTH`/`IMAGE_HEIGHT`, the masks or the mask handling and
  every threshold needs re-fitting (from the count log).
- **One inference path.** Deployment, evaluation, validation during training
  and the video tools all count through `core.inference.DensityPredictor` with
  a model from `core.model.load_inference_model` (eval, Conv+BN fused, fp16),
  so a number from any of them is the number the alarm would have seen.
- **Output stride 8.** The density map is 1/8 of the input in each direction;
  inputs must be multiples of 8 (`core.model.OUTPUT_STRIDE`, never a literal 8).
- **EMA weights are the model.** Validation and every saved `best_*.pth` use
  the EMA copy.
- **The OT loss is averaged over the batch**, so its weight `--wot` does not
  depend on the batch size.
- **Heatmaps use an absolute scale** (`core.visualize.HEATMAP_VMAX`): never
  normalise a density map per frame, or a sparse frame looks as hot as a
  pile-up.
- **Heavy imports stay where they are used.** The grabber, monitor, main
  process and control panel never load torch; `tests/test_import_cost.py`
  holds every entry module to an import budget.

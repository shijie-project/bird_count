"""The live monitor: every camera in one window, with the count, alarm state
and optional heatmap / region-mask overlays.

It runs in its own process (an OpenCV window needs its event loop pumped and
must not stall the consumer), reads frames and per-camera status straight
from the frame ring, and loads no torch. `MonitorHandler` is the consumer
side: it publishes each batch's status into the ring and starts / stops the
process."""

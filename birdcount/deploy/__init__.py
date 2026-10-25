"""The live multi-camera runtime. Depends on `birdcount.core` only.

Process layout: one frame-grabber process (a thread per camera), one or more
inference processes, and the main process (consumer, alarm, GUI), connected
through a shared-memory frame ring (`shm`). Only the inference processes
load torch.
"""

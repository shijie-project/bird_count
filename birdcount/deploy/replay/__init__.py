"""Replaying recorded video as the farm's days, for tuning the alarm on them.

Recordings (the AXIS recorder's export) are counted camera-day by camera-day
through the deployed pipeline (model, mask, alarm engine, count logger,
evidence) on the recordings' own clock, and land in a run folder that the farm
backend reads like the farm's own files. See `__main__` for the command,
`layout` for the folder, `day` for one camera-day.
"""

"""Threshold calibration from how crowded the pictures look (`python -m birdcount.deploy.calibrate`).

rubric.py  the five crowding levels every rating uses
rater.py   asking a vision model (Anthropic API) for a level per picture
ratings.py ratings on disk, and which pictures still need one
fit.py     ratings -> the count at which a camera's pictures turn dense
images.py  a snapshot with the area outside the camera's mask shaded
"""

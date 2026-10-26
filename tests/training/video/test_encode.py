import av
import numpy as np

from birdcount.training.video import encode
from birdcount.training.video.encode import H264Writer


def test_writes_h264_that_decodes_back_to_the_frames(tmp_path) -> None:
    path = tmp_path / "v.mp4"
    writer = H264Writer(path, 5.0, (97, 63))  # odd: 4:2:0 needs even, the last column and row go
    frames = [np.full((63, 97, 3), 40 * i, np.uint8) for i in range(6)]
    for frame in frames:
        writer.write(frame)
    writer.close()
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        assert (stream.codec_context.name, stream.width, stream.height, stream.average_rate) == ("h264", 96, 62, 5)
        decoded = [f.to_ndarray(format="bgr24") for f in video.decode(stream)]
    assert len(decoded) == len(frames)
    assert all(abs(int(d.mean()) - int(f.mean())) <= 3 for d, f in zip(decoded, frames, strict=True))


def test_without_a_hardware_encoder_libx264_writes_the_same_kind_of_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(encode, "ENCODERS", (("no_such_encoder", {}), encode.ENCODERS[-1]))
    encode.encoder.cache_clear()
    try:
        writer = H264Writer(tmp_path / "v.mp4", 5.0, (64, 48))
        assert writer.codec == "libx264"
        writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.close()
    finally:
        encode.encoder.cache_clear()
    with av.open(str(tmp_path / "v.mp4")) as video:
        assert video.streams.video[0].codec_context.name == "h264"

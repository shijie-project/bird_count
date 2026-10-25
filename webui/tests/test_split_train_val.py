import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "annotations"))

import split_train_val as splitter


def _cam(mac: str, session: str, frame: int) -> dict:
    return {"file_name": f"{frame:03d}_axis2_{mac}_{session}_ABCD_frame{frame}.jpg"}


class SplitTrainValTests(unittest.TestCase):
    def test_camera_key(self):
        self.assertEqual(
            splitter.camera_key("001_axis2_B8A44F5B4394_20260325_164628_F92A_frame926.jpg"), "B8A44F5B4394"
        )
        self.assertEqual(splitter.camera_key("axis3-B8A44F77716C_20260809_064509.jpg"), "B8A44F77716C")
        self.assertEqual(splitter.camera_key("20260311_091325_972C.mkv_20260521_105117.358.jpg"), "B8A44FD51C3C")
        self.assertEqual(splitter.camera_key("001.jpg"), splitter.UNKNOWN_CAMERA)

    def test_session_key(self):
        self.assertEqual(
            splitter.session_key("001_axis2_B8A44F5B4394_20260325_164628_F92A_frame926.jpg"), "20260325_164628"
        )
        self.assertEqual(splitter.session_key("axis3-B8A44F77716C_20260809_064509.jpg"), "20260809_064509")
        self.assertEqual(splitter.session_key("20210404_08_73.jpg"), "20210404_08")
        self.assertEqual(splitter.session_key("001.jpg"), "001.jpg")

    def test_every_camera_in_val_and_sessions_not_split(self):
        images = []
        for mac in ("AAAAAAAAAAAA", "BBBBBBBBBBBB", "CCCCCCCCCCCC"):
            for session in ("20260101_000000", "20260102_000000", "20260103_000000"):
                images += [_cam(mac, session, frame) for frame in range(3)]
        for index, image in enumerate(images):
            image["id"] = index

        train, val, per_camera = splitter.split_images(images, val_ratio=0.2, seed=0)

        self.assertEqual(set(per_camera), {"AAAAAAAAAAAA", "BBBBBBBBBBBB", "CCCCCCCCCCCC"})
        self.assertEqual({splitter.camera_key(im["file_name"]) for im in val}, set(per_camera))
        train_sessions = {
            (splitter.camera_key(im["file_name"]), splitter.session_key(im["file_name"])) for im in train
        }
        val_sessions = {(splitter.camera_key(im["file_name"]), splitter.session_key(im["file_name"])) for im in val}
        self.assertFalse(train_sessions & val_sessions)
        self.assertEqual({im["id"] for im in train} | {im["id"] for im in val}, set(range(len(images))))

    def test_oversized_sessions_still_give_camera_a_val_session(self):
        images = [dict(_cam("AAAAAAAAAAAA", "20260101_000000", frame), id=frame) for frame in range(6)]
        images += [dict(_cam("AAAAAAAAAAAA", "20260102_000000", frame), id=frame) for frame in range(6, 10)]

        train, val, _ = splitter.split_images(images, val_ratio=0.2, seed=0)

        self.assertEqual({im["id"] for im in val}, set(range(6, 10)))
        self.assertEqual(len(train), 6)

    def test_single_session_camera_stays_in_train(self):
        images = [dict(_cam("AAAAAAAAAAAA", "20260101_000000", frame), id=frame) for frame in range(10)]

        train, val, _ = splitter.split_images(images, val_ratio=0.2, seed=0)

        self.assertEqual(val, [])
        self.assertEqual(len(train), 10)

    def test_split_is_reproducible(self):
        images = [
            dict(_cam(f"{index % 4:012d}", f"2026010{index % 3}_000000", index), id=index) for index in range(30)
        ]

        first = splitter.split_images(images, val_ratio=0.2, seed=9)
        second = splitter.split_images(images, val_ratio=0.2, seed=9)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

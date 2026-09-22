from pathlib import Path
import unittest


LAUNCHER = Path(__file__).parents[1] / "teleop" / "run_g1_quest_dex3.sh"


class G1QuestDex3LauncherTest(unittest.TestCase):
    def test_pins_cyclonedds_to_the_g1_internal_bus_interface(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("CYCLONEDDS_URI", source)
        self.assertIn('NetworkInterface name="enP8p1s0"', source)


if __name__ == "__main__":
    unittest.main()

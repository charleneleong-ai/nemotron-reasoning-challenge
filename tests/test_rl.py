from src.train.rl import boxed_reward, format_reward


class TestBoxedReward:
    def test_rewards_correct_boxed(self):
        comps = [r"reason \boxed{12}", r"reason \boxed{99}"]
        assert boxed_reward(comps, answer=["12", "13"]) == [1.0, 0.0]

    def test_numeric_tolerance(self):
        assert boxed_reward([r"\boxed{3.14}"], answer=["3.145"]) == [1.0]

    def test_missing_box_is_zero(self):
        assert boxed_reward(["no box here"], answer=["12"]) == [0.0]

    def test_conversational_completion(self):
        comps = [[{"role": "assistant", "content": r"\boxed{7}"}]]
        assert boxed_reward(comps, answer=["7"]) == [1.0]


class TestFormatReward:
    def test_think_then_boxed_rewarded(self):
        good = r"<think>work</think> the answer is \boxed{5}"
        assert format_reward([good]) == [1.0]

    def test_missing_think_or_box_not_rewarded(self):
        assert format_reward([r"\boxed{5} but no think"]) == [0.0]
        assert format_reward(["<think>work</think> no box"]) == [0.0]

import unittest

from factoriollm.train_model import detect_template, subsample_per_blueprint


def record(cid, lang, prompt):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "user", "content": prompt}],
            "meta": {"canonical_id": cid, "lang": lang}}


class TestTrainModelHelpers(unittest.TestCase):
    def test_detect_template(self):
        self.assertEqual(detect_template("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"), "qwen-2.5")
        self.assertEqual(detect_template("unsloth/Qwen3-8B-unsloth-bnb-4bit"), "qwen3-instruct")
        self.assertEqual(detect_template("unsloth/llama-3-8b-Instruct-bnb-4bit"), "llama-3")
        with self.assertRaises(ValueError):
            detect_template("some/unknown-model")

    def test_subsample_keeps_one_prompt_per_language_and_blueprint(self):
        records = [record(1, "en", "a"), record(1, "en", "b"), record(1, "ru", "в"), record(1, "ru", "г"), record(2, "en", "c")]
        chosen = subsample_per_blueprint(records)
        self.assertEqual(sorted((r["meta"]["canonical_id"], r["meta"]["lang"]) for r in chosen), [(1, "en"), (1, "ru"), (2, "en")])
        self.assertEqual(chosen, subsample_per_blueprint(list(reversed(records))))  # deterministic

    def test_subsample_requires_meta(self):
        with self.assertRaises(ValueError):
            subsample_per_blueprint([{"messages": [{}, {"content": "x"}]}])


if __name__ == "__main__":
    unittest.main()

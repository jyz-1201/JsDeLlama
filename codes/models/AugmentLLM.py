import logging

import torch

from codes.evaluators.evaluators import SyntaxEvaluator, ComplexityEvaluator, SafeCodeEvaluator, CodeBLEUEvaluator, \
    CodeBertScoreEvaluator

from codes.Config import format_prompt


class RewardEvaluator:
    contrainer_name: str = "eval_js_container"
    REWARD_WEIGHT = {
        "syntax_pass": 3,
        "exe_pass": 3,
        "simplification_score": 1,
        "codebleu": 1,
        "F1": 1,
    }
    MAX_POSSIBLE_REWARD = sum(REWARD_WEIGHT.values())

    def __init__(self, device="cuda:0"):
        # create evaluators
        self.syntax_evaluator = SyntaxEvaluator()
        self.safe_code_evaluator = SafeCodeEvaluator(contrainer_name=self.contrainer_name)
        self.complexity_evaluator = ComplexityEvaluator()
        self.code_bleu_evaluator = CodeBLEUEvaluator()
        self.code_bert_score_evaluator = CodeBertScoreEvaluator(device=device)

    def compute_batch_reward(self, batch, generations):
        # print(f"compute_batch_reward: \nbatch:{batch}\ngenerations:{generations}\n", flush=True)

        batch_score = []

        for sample, output in zip(batch, generations):
            sample['deobfuscated'] = output
            reward = self.compute_reward(sample, True)
            batch_score.append(torch.tensor(reward))

        return batch_score

    def compute_reward(self, sample, with_similarity=False):
        score = 0.0

        # Evaluate syntax correctness
        res_syntax = self.syntax_evaluator.evaluate(sample)
        if res_syntax['pass']:
            score += self.REWARD_WEIGHT["syntax_pass"]
        else:
            return score

        # Evaluate semantic correctness
        res_exe = self.safe_code_evaluator.evaluate(sample)
        score += res_exe['pass'] * self.REWARD_WEIGHT["exe_pass"]

        # Evaluate code simplification
        res_cpx = self.complexity_evaluator.evaluate(sample)
        score += res_cpx["simplification_score"] * self.REWARD_WEIGHT["simplification_score"]

        if with_similarity:
            res_bleu = self.code_bleu_evaluator.evaluate(sample)
            score += res_bleu["codebleu"] * self.REWARD_WEIGHT["codebleu"]

            res_cbs = self.code_bert_score_evaluator.evaluate_dataset([sample])
            score += res_cbs["F1"] * self.REWARD_WEIGHT["F1"]

        score /= self.MAX_POSSIBLE_REWARD

        return score


class AugmentLLM:
    def __init__(self, pipe, distributed_state, tokenizer, decoding_strategy, max_length=1024*6, max_new_tokens=1024*2, num_beams=6):
        self.pipe = pipe
        self.distributed_state = distributed_state
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.decoding_strategy = decoding_strategy
        self.num_beams = num_beams

        self.reward_evaluator = RewardEvaluator()

    def generate(self, dataset):
        prompts = [format_prompt(sample, with_groundtruth=False) for sample in dataset]

        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': self.max_length}

        if self.decoding_strategy == "beamsampling":
            # Beam-search multinomial sampling
            generation_kwargs = {
                "num_beams": self.num_beams,
                "num_return_sequences": self.num_beams,
                "do_sample": True,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_new_tokens": self.max_new_tokens,
            }
        else:
            print("WARNING: No decoding strategy is provided. Default to vanilla inference setting..")
            generation_kwargs = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_new_tokens": self.max_new_tokens,
            }
        # Inference with LLM
        outputs = self.pipe(prompts, return_full_text=False, **tokenizer_kwargs, **generation_kwargs)

        # Select best sample by rewards
        for sample_in_dataset, outputs_in_sample in zip(dataset, outputs):
            generated_samples = [sample_in_dataset for _ in outputs_in_sample]
            max_score = -1
            best_sample = None
            for sample, output in zip(generated_samples, outputs_in_sample):
                sample['deobfuscated'] = output["generated_text"]
                score = self.reward_evaluator.compute_reward(sample)

                if score > max_score:
                    max_score = score
                    best_sample = sample

            sample_in_dataset = best_sample

        return dataset

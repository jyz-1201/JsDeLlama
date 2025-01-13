import os
import time

import fire
import logging

from codes.evaluators.evaluators import evaluate_deobfuscation

os.environ['TOKENIZERS_PARALLELISM'] = "true"
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%m-%d %H:%M:%S')

def main(
        results: str,
        contrainer_name: str = "eval_js",
        generated_summary: str = "",
        ):
    # contrainer_name must be used exclusively
    logging.info(f"load deobfuscation results: {results}")
    evaluate_deobfuscation(results, contrainer_name=contrainer_name, save_with_metrics=True, generated_summary_file=generated_summary)

if __name__ == "__main__":
    fire.Fire(main)
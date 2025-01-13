import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

from codes.build_dataset.data_io import read_dataset, read_jsonl_as_df, obfuscation_types


class Prompts:
    DEOBFUSCATE = lambda args: "Deobfuscate the following code:"
    OBFUSCATE = lambda obfuscation_type: f"Obfuscate the following code using {obfuscation_type}:"
    TYPE_PREDICTION = lambda args: "Predict the obfuscation type of the following code:"
    INTENTION_SUMMARIZATION = lambda args: "Summarize in 1-3 sentences the objective of the following code:"


def prepare_instructions(df, task_type, prompts_func, inputs_column, outputs_column, prompts_args):
    copied_df = df.copy()
    copied_df["task_type"] = task_type
    copied_df["instruction"] = copied_df[prompts_args].apply(prompts_func)
    copied_df["input"] = copied_df[inputs_column]
    copied_df["output"] = copied_df[outputs_column]
    return copied_df


def main():
    input_folders = [f"codenet_dataset_{obfuscation_type}" for obfuscation_type in obfuscation_types]
    input_paths = [f"./datasets/{folder}/Project_CodeNet_selected.jsonl" for folder in input_folders]

    dataframes_raw = [read_jsonl_as_df(path) for path in input_paths]

    for dataframe, obfuscation_type in zip(dataframes_raw, obfuscation_types):
        dataframe['obfuscation_type'] = obfuscation_type

    df_raw = pd.concat(dataframes_raw, ignore_index=True)

    tasks_types = ["deobfuscate", "intention_summarization", "obfuscate", "type_prediction"]
    prompts_funcs = [Prompts.DEOBFUSCATE, Prompts.INTENTION_SUMMARIZATION, Prompts.OBFUSCATE, Prompts.TYPE_PREDICTION]
    inputs_columns = ["obfuscated", "obfuscated", "code", "obfuscated"]
    outputs_columns = ["code", "gpt_summary", "obfuscated", "obfuscation_type"]


    dataframes_instruct = [prepare_instructions(df_raw, task_type, prompts_func, inputs_column, outputs_column, "obfuscation_type")
                           for task_type, prompts_func, inputs_column, outputs_column in zip(tasks_types, prompts_funcs, inputs_columns, outputs_columns)]

    # training with only one task (deobfuscate)
    dataframes_instruct = dataframes_instruct[:1]

    # training with first two task (deobfuscate and intention_summarization)
    # dataframes_instruct = dataframes_instruct[:2]


    df_instruct = pd.concat(dataframes_instruct, ignore_index=True)

    max_task_id = df_instruct["task_id"].max()
    df_above_threshold = df_instruct[df_instruct["task_id"] >= 0.15 * max_task_id]
    df_below_threshold = df_instruct[df_instruct["task_id"] < 0.15 * max_task_id]

    train_dataset = Dataset.from_pandas(df_above_threshold)
    test_dataset = Dataset.from_pandas(df_below_threshold)

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    dataset.save_to_disk("./datasets/instruct_2task")


if __name__ == "__main__":
    main()

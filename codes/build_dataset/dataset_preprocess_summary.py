import pandas as pd
from datasets import Dataset, DatasetDict

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
    input_path = f"./datasets/dataset_with_summaries.jsonl"

    df_raw_summary = read_jsonl_as_df(input_path)

    tasks_types = ["deobfuscate", "intention_summarization", "obfuscate", "type_prediction"]
    prompts_funcs = [Prompts.DEOBFUSCATE, Prompts.INTENTION_SUMMARIZATION, Prompts.OBFUSCATE, Prompts.TYPE_PREDICTION]
    inputs_columns = ["obfuscated", "obfuscated", "code", "obfuscated"]
    outputs_columns = ["code", "gpt_summary", "obfuscated", "obfuscation_type"]


    dataframes_instruct_summary = [prepare_instructions(df_raw_summary, task_type, prompts_func, inputs_column, outputs_column, "obfuscation_type")
                           for task_type, prompts_func, inputs_column, outputs_column in zip(tasks_types, prompts_funcs, inputs_columns, outputs_columns)]

    # training with task intention_summarization
    dataframes_instruct_summary = dataframes_instruct_summary[1:2]

    df_instruct_summary = pd.concat(dataframes_instruct_summary, ignore_index=True)

    # # Sort the dataframe lexicographically by the column "filename"
    # df_sorted = df_instruct.sort_values("filename")
    # # Reset the index to ensure it is sequential after sorting
    # df_sorted.reset_index(drop=True, inplace=True)
    # # Calculate the split index
    # split_index = int(len(df_sorted) * 0.15)
    # # Split the dataframe
    # first_15_percent = df_sorted.iloc[:split_index]
    # last_85_percent = df_sorted.iloc[split_index:]
    #
    # train_dataset = Dataset.from_pandas(last_85_percent)
    # test_dataset = Dataset.from_pandas(first_15_percent)

    input_folders = [f"codenet_dataset_{obfuscation_type}" for obfuscation_type in obfuscation_types]
    input_paths = [f"./datasets/{folder}/Project_CodeNet_selected.jsonl" for folder in input_folders]
    tasks_types = ["deobfuscate", "obfuscate", "type_prediction"]
    prompts_funcs = [Prompts.DEOBFUSCATE, Prompts.OBFUSCATE, Prompts.TYPE_PREDICTION]
    inputs_columns = ["obfuscated", "code", "obfuscated"]
    outputs_columns = ["code", "obfuscated", "obfuscation_type"]


    dataframes_raw = [read_jsonl_as_df(path) for path in input_paths]

    for dataframe, obfuscation_type in zip(dataframes_raw, obfuscation_types):
        dataframe['obfuscation_type'] = obfuscation_type

    df_raw = pd.concat(dataframes_raw, ignore_index=True)

    dataframes_instruct = [prepare_instructions(df_raw, task_type, prompts_func, inputs_column, outputs_column, "obfuscation_type")
                           for task_type, prompts_func, inputs_column, outputs_column in zip(tasks_types, prompts_funcs, inputs_columns, outputs_columns)]

    # training with only one task (deobfuscate)
    dataframes_instruct = dataframes_instruct[:1]

    df_instruct = pd.concat(dataframes_instruct, ignore_index=True)

    # df_subset_summary = df_instruct_summary[["filename", "obfuscation_type", "task_type", "gpt_summary"]]
    # Remove the column "task_id"
    df_subset_summary = df_instruct_summary.drop("task_id", axis=1)
    df_subset_deobfuscate = df_instruct[["filename", "obfuscation_type", "task_id"]]

    df_merged = pd.merge(df_subset_deobfuscate, df_subset_summary, on=["filename", "obfuscation_type"])

    max_task_id = df_merged["task_id"].max()
    df_above_threshold = df_merged[df_merged["task_id"] >= 0.15 * max_task_id]
    df_below_threshold = df_merged[df_merged["task_id"] < 0.15 * max_task_id]

    train_dataset = Dataset.from_pandas(df_above_threshold)
    test_dataset = Dataset.from_pandas(df_below_threshold)

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    dataset.save_to_disk("./datasets/instruct_summary")


if __name__ == "__main__":
    main()

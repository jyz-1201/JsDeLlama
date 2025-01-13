import os


def find_newest_checkpoint(path, subdir_prefix):
    """
    Finds the largest training step in the largest epoch among subdirectories.

    Args:
        path (str): Path containing subdirectories like "ppo_checkpoint_x_y".

    Returns:
        Tuple[int, int]: Largest epoch and largest training step.
    """
    if not os.path.exists(path):
        return -1, -1

    largest_epoch = -1
    largest_step = -1

    for subdir in os.listdir(path):
        if subdir.startswith(subdir_prefix):
            try:
                epoch, step = map(int, subdir.split("_")[2:])
                if epoch > largest_epoch:
                    largest_epoch = epoch
                    largest_step = step
                elif epoch == largest_epoch and step > largest_step:
                    largest_step = step
            except ValueError:
                # Ignore invalid subdirectory names
                pass

    return largest_epoch, largest_step

"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import os
import wandb
import sys
import yaml
import torch
import datetime

import utils

from networks import MNISTNet
from dataset import MyDataset
from train import train

# Experiment entrypoint. This script allows to run a single experiment, by passing hyper-parameters as command line
# arguments. It exploits Weights and Biases (W&B) groups and tags to organize experiments in a meaningful way.
# Groups will be given a mnemonic name based on hyper-parameters, in this way, repeating an experiment will automatically
# merge runs into the same group, and aggregated statistics can be computed from the W&B dashboard.
# Events triggered during training (e.g., successful completion, vanishing gradients, etc.) will be associated to a tag,
# to easily filter runs. Tags belong to two categories (informative and errors), a run with errors will be additionally
# flagged by a "Failed" status, instead of the default "Finished", for further filtering.
# This script can be used in three modes:
# - Standalone: do not pass a --wandb_project argument. It will save results locally (if --save True) or print them on screen.
# - W&B single experiment: pass a --wandb_project argument. It will upload results to W&B at the end.
# - W&B sweep experiment: invoke the script from a W&B sweep. In this mode you can fully exploit experiment replicates by
#   sweeping through multiple random seeds (which will be associated to the same group name).


def run(opts, rng):
    """
    Perform an experiment.
    :param opts: Dictionary of hyper-parameters.
    :param rng: Seeded numpy.random.Generator.
    :return: Tuple (return_flags: set of events encountered during training, model: the trained torch.nn.Module, history: list of metrics).
    """
    # Create the datasets, they may require the seeded generator if data is not loaded from disk.
    train_ds = MyDataset(opts["prefix_path"], "train", transform=None)
    val_ds = MyDataset(opts["prefix_path"], "val", transform=None)
    test_ds = MyDataset(opts["prefix_path"], "test", transform=None)

    # Build the model. Our experiments usually are performed from scratch, so we do not load pre-existing weights.
    model = MNISTNet(opts)

    # Train the model. Our experiments do not usually involve inference, so we simply terminate them after training.
    # Note: The seeded generator is not needed if random operations are not performed by numpy,
    # as global generators have been seeded as well.
    for (status, history, return_tags) in train(model, train_ds, val_ds, test_ds, rng, opts):
        # Postpone end-of-training signaling, so that the process is not killed before saving results.
        if status != "end":
            yield (status, history, return_tags) # Simply pass values to the watchdog wrapper.

    if opts["save"]:
        op = "{}/{}".format(opts["prefix_path"], opts["output_path"])
        if not os.path.exists(op):
            os.mkdir(op)

        # Save model weights.
        filename = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), experiment_name)
        torch.save(model.state_dict(), "{}/{}.pt".format(op, filename))

        # Save hyper-parameters, to reconstruct the correct model.
        with open("{}/{}.yml".format(op, filename), "w") as file:
            yaml.safe_dump(opts, file)

        # Save results.
        with open("{}/{}_results.yml".format(op, filename), "w") as file:
            yaml.safe_dump({"history": history, "tags": return_tags}, file)

    yield ("end", history, return_tags)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Warning: no arguments were provided. Will run with default values.")

    arg_parser = utils.get_arg_parser()
    opts = vars(arg_parser.parse_args())

    utils.preflight_checks(opts)
    run_experiment = utils.prune_hyperparameters(opts, arg_parser) or not opts["abort_irrelevant"]
    experiment_name = utils.generate_name(opts)

    # Override experiment device if environment variable is set.
    if os.environ.get('DEVICE') is not None:
        opts['device'] = os.environ.get('DEVICE')


    wb = None # Don't use W&B.
    if "WANDB_SWEEP_ID" in os.environ: # Use W&B in sweep mode.
        wb = wandb.init(group=experiment_name, config=opts)
    elif opts['wandb_project'] is not None: # Use W&B in standalone mode.
        wb = wandb.init(project=opts['wandb_project'], group=experiment_name, config=opts)


    if run_experiment:
        rng = utils.set_seed(opts["seed"])

        with utils.Watchdog(run, opts["heartbeat"], opts, rng) as wd:
            history, return_tags = wd.listen(opts["epoch_timeout"] * 60)

        if wb is not None:
            wb.tags = sorted(return_tags)
            for i, h in enumerate(history):
                wb.log(data=h, step=i)

            if "Success" not in return_tags:
                wb.finish(exit_code=1)
            else:
                wb.finish()

        else: # If W&B is not enabled, print results on screen.
            print({"history": history, "tags": return_tags})

    else:
        if wb is not None:
            wb.tags = ["Irrelevant hyperparameters"]
            wb.finish(exit_code=1)

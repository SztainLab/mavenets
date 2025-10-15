"""Evaluate single mlp on test set."""
from typing import Final, List, Tuple
from itertools import product
import torch
import pandas as pd  # type: ignore
from ...data import get_datasets, DATA_SPECS
from ...network import MLP, NullTuner
from ...tools import train_tunable_model
from ...report import predict

# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 2


def test_linear(
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
) -> pd.DataFrame:
    """Train model and evaluate."""

    train_dataset, valid_dataset = get_datasets(train_specs=['base'], val_specs=['base'], device=DEVICE, feat_type="onehot")

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=[],
        pre_flatten=True,
        post_squeeze=True,
    )
    model = NullTuner(underlying_model).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    best_epoch, best_val, training_record = train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=n_epochs,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        report_datasets=report_datasets,
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )

    # get test data
    _, _, eval_data = get_datasets(device=DEVICE, feat_type="onehot", include_test=True)
    pred_table = predict(model=model, dataset=eval_data, batch_size=1024)

    return best_epoch, best_val, model, training_record, pred_table


def helper(seed: int, **kwargs):
    torch.manual_seed(seed)
    best_epoch, _, _, _, _ = test_linear(
            **kwargs
    )

    new_kwargs = kwargs.copy()
    new_kwargs["n_epochs"] = best_epoch

    torch.manual_seed(seed)
    _, _, model, training_report, test_pred = test_linear(
            **new_kwargs
    )

    valid = training_report.valid.iloc[-1]
    return (model, valid, test_pred)


def run(attempt_seeds: Tuple= (1234231, 54636, 2931243)) -> None:
    wdecay = 0.0005
    n_epochs = 400
    lr = 1e-4
    record = {}
    for seed in attempt_seeds:
        model, valid, test_pred = helper(
            seed=seed,
            weight_decay=wdecay,
            n_epochs=n_epochs,
            learning_rate=lr,
        )
        record.update({valid: (model, test_pred)})
    best_model, best_table = record[min(record.keys())]
    print("val from random inits:", list(record.keys()))
    core_name = "TESTEVAL_linear_wdecay{}_learningrate{}_baseonly_nulltuner_multiattempt".format(wdecay, lr)
    best_table.to_csv(core_name+".csv")
    torch.save(best_model, core_name+".pt")


if __name__ == "__main__":
    scan()

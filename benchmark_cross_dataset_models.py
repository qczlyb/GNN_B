import argparse
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
import pyamg
import scipy.sparse as sp
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pyamg.aggregation.adaptive import adaptive_sa_solver
from pyamg.relaxation.smoothing import change_smoothers
from scipy.sparse.linalg import gmres

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_cg.loss import _make_sparse_tensor
from neural_cg.nn.gnns import NodeEdgeProcessing
from neural_cg.utils.datamodule import FolderDataModule, MultiFolderDataModule


warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: str


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def torch_sparse_to_scipy_csr(torch_tensor: torch.Tensor, shape: tuple[int, int]) -> sp.csr_matrix:
    if not torch_tensor.is_sparse:
        raise ValueError("Input must be a sparse tensor.")
    tensor_cpu = torch_tensor.detach().cpu().coalesce()
    indices = tensor_cpu.indices().numpy()
    values = tensor_cpu.values().numpy()
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr().astype(np.float64)


def configure_smoothers(ml) -> None:
    smoothers = ("jacobi", {"iterations": 3, "omega": 0.66})
    change_smoothers(ml, presmoother=smoothers, postsmoother=smoothers)


def build_sa_preconditioner(A: sp.csr_matrix):
    start = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(A)
    configure_smoothers(ml)
    setup_time = time.perf_counter() - start
    return ml.aspreconditioner(cycle="V"), setup_time


def build_adaptive_preconditioner(A: sp.csr_matrix, num_candidates: int):
    start = time.perf_counter()
    ml, _ = adaptive_sa_solver(A, num_candidates=num_candidates)
    configure_smoothers(ml)
    setup_time = time.perf_counter() - start
    return ml.aspreconditioner(cycle="V"), setup_time


def build_gnn_preconditioner(A: sp.csr_matrix, B_candidates: np.ndarray):
    start = time.perf_counter()
    B_candidates = np.ascontiguousarray(B_candidates.astype(np.float64, copy=False))
    ml = pyamg.smoothed_aggregation_solver(A, B=B_candidates)
    configure_smoothers(ml)
    setup_time = time.perf_counter() - start
    return ml.aspreconditioner(cycle="V"), setup_time


def solve_with_gmres(
    A: sp.csr_matrix,
    b: np.ndarray,
    M,
    tol: float,
    restart: int,
    maxiter: int,
) -> dict:
    residuals: list[float] = []

    def callback(rk):
        if np.ndim(rk) == 0:
            residuals.append(float(rk))
        else:
            residuals.append(float(np.linalg.norm(rk)))

    start = time.perf_counter()
    _, info = gmres(
        A,
        b,
        M=M,
        rtol=tol,
        restart=restart,
        maxiter=maxiter,
        callback=callback,
        callback_type="pr_norm",
    )
    solve_time = time.perf_counter() - start
    return {
        "iterations": len(residuals),
        "solve_time_s": solve_time,
        "info": int(info),
        "converged": info == 0,
    }


class CrossDatasetModelBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(__file__).resolve().parent
        self.device = torch.device(args.device)
        self.datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
        self.output_dir = resolve_repo_path(self.repo_root, args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def model_specs_for_dataset(self, dataset_name: str) -> list[ModelSpec]:
        specs = [
            ModelSpec(
                label=f"GNN-{dataset_name}",
                checkpoint=f"saved_models/{dataset_name}_6qr.ckpt",
            ),
            ModelSpec(
                label="GNN-mixed",
                checkpoint="saved_models/mixed_6qr.ckpt",
            ),
        ]
        if dataset_name == "anisotropy":
            specs.append(
                ModelSpec(
                    label="GNN-anisotropy-Psa",
                    checkpoint="saved_models/anisotropy_Psa_6qr.ckpt",
                )
            )
        return specs

    def load_data(self, dataset_name: str):
        data_prefix = f"generated/{dataset_name}"
        overrides = [
            f"exp_name={dataset_name}",
            f"data.prefix={data_prefix}",
            "data.is_fixed_topology=false",
            "data.has_shared_features=false",
            "data.use_node_features=false",
            "data.use_edge_features_as_node_feature=mean",
            "data.load_into_memory=false",
        ]
        overrides.extend(self.args.override or [])

        config_dir = resolve_repo_path(self.repo_root, self.args.config_dir)
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
        cfg = hydra.compose(config_name="basic", overrides=overrides)

        use_multidata = "all_prefix" in cfg.data
        data_module_class = MultiFolderDataModule if use_multidata else FolderDataModule
        data_module = data_module_class(data_config=cfg.data, split_config=cfg.split, batch_size=1)
        loader = data_module.test_dataloader()

        test_indices = list(getattr(data_module, "val_idx", range(len(loader))))
        print(
            f"[data:{dataset_name}] prefix={data_prefix}, "
            f"total_samples={len(data_module.dataset)}, test_samples={len(loader)}"
        )
        print(
            f"[data:{dataset_name}] node_features={data_module.dataset.num_node_features}, "
            f"edge_features={data_module.dataset.num_edge_features}, block_size={data_module.dataset.block_size}"
        )
        return data_module, loader, test_indices

    @staticmethod
    def resolve_gnn_config(hparams: dict) -> dict:
        cfg = OmegaConf.create(
            {
                "gnn_features": hparams.get("gnn_features", 16),
                "gnn_mlp_layers": hparams.get("gnn_mlp_layers", 2),
                "gnn": hparams["gnn"],
            }
        )
        return OmegaConf.to_container(cfg.gnn, resolve=True)

    @staticmethod
    def infer_node_out_features(state_dict: dict) -> int:
        key = "gnn.node_dec.proj.0.weight"
        if key not in state_dict:
            raise KeyError(f"Cannot infer node output dimension; missing checkpoint key: {key}")
        return int(state_dict[key].shape[0])

    def load_models_for_dataset(self, dataset_name: str, data_module) -> dict[str, dict]:
        models = {}
        for spec in self.model_specs_for_dataset(dataset_name):
            ckpt_path = resolve_repo_path(self.repo_root, spec.checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

            print(f"[load:{dataset_name}] {spec.label}: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            hparams = checkpoint.get("hyper_parameters", {})
            gnn_cfg = self.resolve_gnn_config(hparams)
            node_out_features = self.infer_node_out_features(checkpoint["state_dict"])

            model = NodeEdgeProcessing(
                node_in_features=data_module.dataset.num_node_features,
                node_out_features=node_out_features,
                edge_in_features=data_module.dataset.num_edge_features,
                edge_out_features=data_module.dataset.block_size * data_module.dataset.block_size,
                **gnn_cfg,
            )
            gnn_state = {
                key.removeprefix("gnn."): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("gnn.")
            }
            model.load_state_dict(gnn_state, strict=True)
            model.eval().to(self.device)
            models[spec.label] = {
                "model": model,
                "node_out_features": node_out_features,
                "checkpoint": str(ckpt_path),
            }
        return models

    def make_linear_system(self, batch):
        num_nodes = int(batch.num_nodes)
        A_sparse = _make_sparse_tensor(
            batch.edge_index,
            batch.matrix_values.flatten(),
            num_nodes,
            batch.mask,
            dtype=torch.float32,
        ).coalesce()
        A_scipy = torch_sparse_to_scipy_csr(A_sparse, (num_nodes, num_nodes))

        if hasattr(batch, "residual"):
            b = batch.residual.detach().cpu().numpy().astype(np.float64).flatten()
        else:
            b = np.ones(num_nodes, dtype=np.float64)
        return A_scipy, b, num_nodes

    def predict_candidates(self, model: torch.nn.Module, batch, expected_dim: int):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            B_tensor, _ = model(batch.x, batch.edge_index, batch.edge_attr)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gnn_time = time.perf_counter() - start

        B_candidates = B_tensor.detach().cpu().numpy().astype(np.float64)
        if B_candidates.ndim != 2 or B_candidates.shape[1] != expected_dim:
            raise ValueError(f"Expected B shape [n, {expected_dim}], got {B_candidates.shape}.")
        return B_candidates, gnn_time

    def run_classical_method(self, method_label: str, A_scipy: sp.csr_matrix, b: np.ndarray):
        if method_label == "SA-AMG":
            M, setup_time = build_sa_preconditioner(A_scipy)
        elif method_label == "Adaptive-SA":
            M, setup_time = build_adaptive_preconditioner(A_scipy, self.args.adaptive_candidates)
        else:
            raise ValueError(f"Unknown classical method: {method_label}")

        solve_result = solve_with_gmres(
            A_scipy,
            b,
            M,
            tol=self.args.tol,
            restart=self.args.restart,
            maxiter=self.args.maxiter,
        )
        return {
            "iterations": solve_result["iterations"],
            "setup_time_s": setup_time,
            "solve_time_s": solve_result["solve_time_s"],
            "total_time_s": setup_time + solve_result["solve_time_s"],
            "gnn_time_s": 0.0,
            "amg_setup_time_s": setup_time,
            "info": solve_result["info"],
            "converged": solve_result["converged"],
        }

    def run_gnn_method(self, model_info: dict, A_scipy: sp.csr_matrix, b: np.ndarray, batch):
        B_candidates, gnn_time = self.predict_candidates(
            model_info["model"],
            batch,
            model_info["node_out_features"],
        )
        M, amg_setup_time = build_gnn_preconditioner(A_scipy, B_candidates)
        solve_result = solve_with_gmres(
            A_scipy,
            b,
            M,
            tol=self.args.tol,
            restart=self.args.restart,
            maxiter=self.args.maxiter,
        )
        setup_time = gnn_time + amg_setup_time
        return {
            "iterations": solve_result["iterations"],
            "setup_time_s": setup_time,
            "solve_time_s": solve_result["solve_time_s"],
            "total_time_s": setup_time + solve_result["solve_time_s"],
            "gnn_time_s": gnn_time,
            "amg_setup_time_s": amg_setup_time,
            "info": solve_result["info"],
            "converged": solve_result["converged"],
        }

    def run_dataset(self, dataset_name: str) -> pd.DataFrame:
        data_module, loader, test_indices = self.load_data(dataset_name)
        models = self.load_models_for_dataset(dataset_name, data_module)
        set_global_seed(self.args.seed)

        records = []
        max_samples = self.args.max_samples if self.args.max_samples > 0 else None
        classical_methods = ["SA-AMG", "Adaptive-SA"]

        for sample_id, batch in enumerate(loader):
            if max_samples is not None and sample_id >= max_samples:
                break

            batch = batch.to(self.device)
            A_scipy, b, num_nodes = self.make_linear_system(batch)
            dataset_idx = test_indices[sample_id] if sample_id < len(test_indices) else sample_id
            print(f"[{dataset_name} sample {sample_id}] n={num_nodes}, dataset_idx={dataset_idx}")

            method_items = [(label, "classical", None) for label in classical_methods]
            method_items += [(label, "gnn", info) for label, info in models.items()]

            for method_label, method_type, model_info in method_items:
                record = {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "dataset_idx": dataset_idx,
                    "method": method_label,
                    "method_type": method_type,
                    "num_nodes": num_nodes,
                    "node_out_features": np.nan,
                    "iterations": np.nan,
                    "setup_time_s": np.nan,
                    "solve_time_s": np.nan,
                    "total_time_s": np.nan,
                    "gnn_time_s": np.nan,
                    "amg_setup_time_s": np.nan,
                    "info": np.nan,
                    "converged": False,
                    "error": "",
                }

                try:
                    if method_type == "classical":
                        result = self.run_classical_method(method_label, A_scipy, b)
                    else:
                        assert model_info is not None
                        result = self.run_gnn_method(model_info, A_scipy, b, batch)
                        record["node_out_features"] = model_info["node_out_features"]

                    record.update(result)
                    print(
                        f"  {method_label:<18} iter={record['iterations']:<5} "
                        f"setup={record['setup_time_s']:.4f}s "
                        f"solve={record['solve_time_s']:.4f}s "
                        f"total={record['total_time_s']:.4f}s"
                    )
                except Exception as exc:
                    record["error"] = repr(exc)
                    print(f"  {method_label:<18} failed: {exc}")

                records.append(record)

        return pd.DataFrame(records)

    @staticmethod
    def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
        if detail_df.empty:
            return pd.DataFrame()

        return (
            detail_df.groupby(["dataset", "method", "method_type"], as_index=False)
            .agg(
                samples=("sample_id", "count"),
                converged=("converged", "sum"),
                iterations=("iterations", "mean"),
                setup_time_s=("setup_time_s", "mean"),
                solve_time_s=("solve_time_s", "mean"),
                total_time_s=("total_time_s", "mean"),
            )
            .sort_values(["dataset", "method_type", "method"])
        )

    def plot_dataset_boxplots(self, detail_df: pd.DataFrame, dataset_name: str) -> None:
        df = detail_df[detail_df["dataset"] == dataset_name].copy()
        if df.empty:
            return

        method_colors = {
            "Baseline": "lightblue",
            "Adaptive": "lightgoldenrodyellow",
            "GNN(Org)": "lightsalmon",
            "GNN(Mix)": "lightgreen",
            "GNN(Psa)": "plum",
            "GNN(Sparse)": "plum",
            "GNN(Smooth)": "lightcyan",
        }
        preferred_methods = [
            "SA-AMG",
            "Adaptive-SA",
            f"GNN-{dataset_name}",
            "GNN-mixed",
            "GNN-anisotropy-Psa",
        ]
        available_methods = list(dict.fromkeys(df["method"].tolist()))
        methods = [method for method in preferred_methods if method in available_methods]
        methods.extend(method for method in available_methods if method not in methods)

        def method_label(method: str) -> str:
            method_lower = method.lower()
            if method == "SA-AMG":
                return "Baseline"
            if method == "Adaptive-SA":
                return "Adaptive"
            if method == "GNN-mixed":
                return "GNN(Mix)"
            if method == f"GNN-{dataset_name}":
                return "GNN(Org)"
            if "psa" in method_lower:
                return "GNN(Psa)"
            if "sparse" in method_lower or method_lower.endswith("-spa"):
                return "GNN(Sparse)"
            if "smooth" in method_lower or method_lower.endswith("-smo"):
                return "GNN(Smooth)"
            if method.startswith("GNN-"):
                return f"GNN({method[4:]})"
            return method

        def finite_array(values: pd.Series) -> np.ndarray:
            numeric = pd.to_numeric(values, errors="coerce")
            return numeric.dropna().to_numpy(dtype=float)

        def format_mean(metric: str, value: float) -> str:
            if metric == "iterations":
                return f"{value:.1f}"
            return f"{value:.3f}"

        labels = [method_label(method) for method in methods]
        colors = [method_colors.get(label, "lightgray") for label in labels]
        metrics = [
            ("iterations", "GMRES Iterations", "Iterations"),
            ("setup_time_s", "Setup Time (GNN Infer + AMG Build)", "Time (s)"),
            ("solve_time_s", "Solve Time (GMRES Iteration)", "Time (s)"),
            ("total_time_s", "Total Time (Setup + Solve)", "Time (s)"),
        ]

        plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 16})
        fig, axes = plt.subplots(1, 4, figsize=(36, 8))
        line_width = 2.0

        for ax, (metric, title, ylabel) in zip(axes, metrics):
            values = [finite_array(df.loc[df["method"] == method, metric]) for method in methods]
            bplot = ax.boxplot(
                values,
                tick_labels=labels,
                showfliers=False,
                patch_artist=True,
                medianprops=dict(linewidth=line_width, color="red"),
                boxprops=dict(linewidth=line_width),
                whiskerprops=dict(linewidth=line_width),
                capprops=dict(linewidth=line_width),
            )
            for patch, color in zip(bplot["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.tick_params(axis="x", rotation=15)

            finite_values = [array for array in values if len(array) > 0]
            if not finite_values:
                continue

            all_values = np.concatenate(finite_values)
            y_min = float(np.min(all_values))
            y_max = float(np.max(all_values))
            span = y_max - y_min
            if span <= 0:
                span = max(abs(y_max), 1.0)
            ax.set_ylim(max(0.0, y_min - 0.08 * span), y_max + 0.18 * span)

            for i, value_array in enumerate(values):
                if len(value_array) == 0:
                    continue
                mean_value = float(np.mean(value_array))
                ax.text(
                    i + 1,
                    mean_value,
                    format_mean(metric, mean_value),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color="darkblue",
                )

        fig.suptitle(f"{dataset_name.capitalize()} Benchmark", fontsize=18, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plot_path = self.output_dir / f"{dataset_name}_boxplots.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"[save] plot: {plot_path}")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_details = []
        for dataset_name in self.datasets:
            dataset_df = self.run_dataset(dataset_name)
            all_details.append(dataset_df)

        detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
        summary_df = self.summarize(detail_df)
        return detail_df, summary_df

    def save_outputs(self, detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        detail_path = self.output_dir / "cross_dataset_model_details.csv"
        summary_path = self.output_dir / "cross_dataset_model_summary.csv"
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        for dataset_name in self.datasets:
            dataset_detail_path = self.output_dir / f"{dataset_name}_details.csv"
            dataset_summary_path = self.output_dir / f"{dataset_name}_summary.csv"
            dataset_detail = detail_df[detail_df["dataset"] == dataset_name]
            dataset_summary = summary_df[summary_df["dataset"] == dataset_name]
            dataset_detail.to_csv(dataset_detail_path, index=False)
            dataset_summary.to_csv(dataset_summary_path, index=False)
            self.plot_dataset_boxplots(detail_df, dataset_name)

        printable = summary_df.copy()
        for col in ["iterations", "setup_time_s", "solve_time_s", "total_time_s"]:
            if col in printable:
                printable[col] = printable[col].round(6)

        print("\n=== Summary ===")
        try:
            print(printable.to_markdown(index=False))
        except Exception:
            print(printable.to_string(index=False))

        print(f"\n[save] detail:  {detail_path}")
        print(f"[save] summary: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare classical AMG methods, in-domain GNN models, and mixed-data GNN models across datasets."
    )
    parser.add_argument(
        "--datasets",
        default="anisotropy,elasticity,maxwell,synthetic",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--config-dir", default="config", help="Hydra config directory.")
    parser.add_argument(
        "--output-dir",
        default="results/cross_dataset_models",
        help="Output folder for CSV tables and boxplots.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate each full test set.")
    parser.add_argument("--adaptive-candidates", type=int, default=6)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be repeated, e.g. --override data.normalize_matrix=mean.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        set_global_seed(args.seed)
        benchmark = CrossDatasetModelBenchmark(args)
        detail_df, summary_df = benchmark.run()
        benchmark.save_outputs(detail_df, summary_df)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

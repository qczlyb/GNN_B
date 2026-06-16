import argparse
import random
import time
import traceback
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyamg
import scipy.sparse as sp
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pyamg.relaxation.smoothing import change_smoothers
from scipy.sparse.linalg import gmres

from neural_cg.loss import _make_sparse_tensor
from neural_cg.nn.gnns import NodeEdgeProcessing
from neural_cg.utils.datamodule import FolderDataModule, MultiFolderDataModule


warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")


def parse_feature_dims(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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


def build_amg_preconditioner(A: sp.csr_matrix, B_candidates: np.ndarray):
    start = time.perf_counter()
    B_candidates = np.ascontiguousarray(B_candidates.astype(np.float64, copy=False))
    ml = pyamg.smoothed_aggregation_solver(A, B=B_candidates)
    smoothers = ("jacobi", {"iterations": 3, "omega": 0.66})
    change_smoothers(ml, presmoother=smoothers, postsmoother=smoothers)
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


class NodeOutFeatureBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(__file__).resolve().parent
        self.device = torch.device(args.device)
        self.feature_dims = parse_feature_dims(args.feature_dims)
        self.output_dir = resolve_repo_path(self.repo_root, args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(args.seed)
        self.loader = self.load_data()
        self.models = self.load_models()
        set_global_seed(args.seed)

    def load_models(self) -> dict[int, torch.nn.Module]:
        models = {}
        for dim in self.feature_dims:
            ckpt_raw = self.args.checkpoint_template.format(dim=dim)
            ckpt_path = resolve_repo_path(self.repo_root, ckpt_raw)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint for dim={dim} does not exist: {ckpt_path}")

            print(f"[load] dim={dim}: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            hparams = checkpoint.get("hyper_parameters", {})
            gnn_cfg = self.resolve_gnn_config(hparams)

            model = NodeEdgeProcessing(
                node_in_features=self.num_node_features,
                node_out_features=dim,
                edge_in_features=self.num_edge_features,
                edge_out_features=self.block_size * self.block_size,
                **gnn_cfg,
            )
            gnn_state = {
                key.removeprefix("gnn."): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("gnn.")
            }
            model.load_state_dict(gnn_state, strict=True)
            model.eval().to(self.device)
            models[dim] = model
        return models

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

    def load_data(self):
        data_prefix = self.args.data_prefix or f"generated/{self.args.exp_name}"
        overrides = [
            f"exp_name={self.args.exp_name}",
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
        self.test_indices = list(getattr(data_module, "val_idx", range(len(loader))))
        self.num_node_features = data_module.dataset.num_node_features
        self.num_edge_features = data_module.dataset.num_edge_features
        self.block_size = data_module.dataset.block_size
        print(f"[data] prefix={data_prefix}, total_samples={len(data_module.dataset)}, test_samples={len(loader)}")
        print(
            f"[data] node_features={self.num_node_features}, "
            f"edge_features={self.num_edge_features}, block_size={self.block_size}"
        )
        return loader

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

    def predict_candidates(self, model: torch.nn.Module, batch, dim: int):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            B_tensor, _ = model(batch.x, batch.edge_index, batch.edge_attr)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gnn_time = time.perf_counter() - start

        B_candidates = B_tensor.detach().cpu().numpy().astype(np.float64)
        if B_candidates.ndim != 2 or B_candidates.shape[1] != dim:
            raise ValueError(f"Expected B shape [n, {dim}], got {B_candidates.shape}.")
        return B_candidates, gnn_time

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        records = []
        max_samples = self.args.max_samples if self.args.max_samples > 0 else None

        for sample_id, batch in enumerate(self.loader):
            if max_samples is not None and sample_id >= max_samples:
                break

            batch = batch.to(self.device)
            A_scipy, b, num_nodes = self.make_linear_system(batch)
            dataset_idx = self.test_indices[sample_id] if sample_id < len(self.test_indices) else sample_id

            print(f"[sample {sample_id}] n={num_nodes}, dataset_idx={dataset_idx}")
            for dim, model in self.models.items():
                record = {
                    "sample_id": sample_id,
                    "dataset_idx": dataset_idx,
                    "node_out_features": dim,
                    "num_nodes": num_nodes,
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
                    B_candidates, gnn_time = self.predict_candidates(model, batch, dim)
                    M, amg_setup_time = build_amg_preconditioner(A_scipy, B_candidates)
                    solve_result = solve_with_gmres(
                        A_scipy,
                        b,
                        M,
                        tol=self.args.tol,
                        restart=self.args.restart,
                        maxiter=self.args.maxiter,
                    )

                    setup_time = gnn_time + amg_setup_time
                    record.update(
                        {
                            "iterations": solve_result["iterations"],
                            "setup_time_s": setup_time,
                            "solve_time_s": solve_result["solve_time_s"],
                            "total_time_s": setup_time + solve_result["solve_time_s"],
                            "gnn_time_s": gnn_time,
                            "amg_setup_time_s": amg_setup_time,
                            "info": solve_result["info"],
                            "converged": solve_result["converged"],
                        }
                    )
                    print(
                        f"  dim={dim:<2} iter={record['iterations']:<5} "
                        f"setup={record['setup_time_s']:.4f}s "
                        f"solve={record['solve_time_s']:.4f}s "
                        f"total={record['total_time_s']:.4f}s"
                    )
                except Exception as exc:
                    record["error"] = repr(exc)
                    print(f"  dim={dim:<2} failed: {exc}")

                records.append(record)

        detail_df = pd.DataFrame(records)
        summary_df = self.summarize(detail_df)
        return detail_df, summary_df

    @staticmethod
    def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
        if detail_df.empty:
            return pd.DataFrame()

        summary_df = (
            detail_df.groupby("node_out_features", as_index=False)
            .agg(
                samples=("sample_id", "count"),
                converged=("converged", "sum"),
                iterations=("iterations", "mean"),
                setup_time_s=("setup_time_s", "mean"),
                solve_time_s=("solve_time_s", "mean"),
                total_time_s=("total_time_s", "mean"),
            )
            .sort_values("node_out_features")
        )
        return summary_df

    def save_and_print(self, detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        detail_path = self.output_dir / "node_out_feature_dim_details.csv"
        summary_path = self.output_dir / "node_out_feature_dim_summary.csv"
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

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
        description="Benchmark PyAMG+GNN candidates with different node output feature dimensions."
    )
    parser.add_argument("--feature-dims", default="3,6,9", help="Comma-separated dimensions, e.g. 3,6,9.")
    parser.add_argument(
        "--checkpoint-template",
        default="saved_models/anisotropy_{dim}qr.ckpt",
        help="Checkpoint path template. Use {dim} as the feature dimension placeholder.",
    )
    parser.add_argument("--exp-name", default="anisotropy", help="Hydra exp_name and default data folder name.")
    parser.add_argument("--data-prefix", default="", help="Dataset folder. Defaults to generated/{exp_name}.")
    parser.add_argument("--config-dir", default="config", help="Hydra config directory.")
    parser.add_argument(
        "--output-dir",
        default="results/node_out_feature_dims",
        help="Output folder for CSV result tables.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all samples.")
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
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        benchmark = NodeOutFeatureBenchmark(args)
        detail_df, summary_df = benchmark.run()
        benchmark.save_and_print(detail_df, summary_df)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

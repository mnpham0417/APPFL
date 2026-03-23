"""
MPI run script for Decentralized CLIP + LoRA fine-tuning with dLoRA AB_SVD.

Implements decentralized federated learning where:
  - Each client fine-tunes a CLIP model with LoRA adapters locally.
  - The server performs topology-aware AB_SVD aggregation of LoRA parameters.
  - Two topologies are supported (selected via the server config):
      "ring" — each agent aggregates with its 2 ring neighbors (weight 1/3).
      "fc"   — all agents aggregate globally (weight 1/N or sample-size).

This mirrors how DiMAT was integrated into APPFL for decentralized learning,
but uses CLIPLoRATrainer + DecentralizedDLoRABSVDAggregator instead.

Two-phase execution:
    Phase 1 (optional): Each client pre-trains its CLIP+LoRA model independently
        for ``--pretrain_steps`` local gradient steps with no communication.
    Phase 2: Consensus-training loop — local train → send LoRA params →
        receive AB_SVD-aggregated LoRA params → repeat.

Usage (ring topology, 5 clients, Flowers-102):
    mpirun -n 6 python run_mpi_decentralized_clip_lora.py \\
        --server_config ./resources/configs/flower102/server_dlora_ab_svd_ring.yaml \\
        --client_config ./resources/configs/flower102/client_dlora_ab_svd_decentralized.yaml \\
        --clip_lora_root /path/to/clip_lora \\
        --pretrain_steps 0

Usage (FC topology, 5 clients, Flowers-102):
    mpirun -n 6 python run_mpi_decentralized_clip_lora.py \\
        --server_config ./resources/configs/flower102/server_dlora_ab_svd_fc.yaml \\
        --client_config ./resources/configs/flower102/client_dlora_ab_svd_decentralized.yaml \\
        --clip_lora_root /path/to/clip_lora \\
        --pretrain_steps 0

Command-line arguments:
    --server_config    Path to server YAML config (selects topology + aggregator).
    --client_config    Path to shared client YAML config template.
    --clip_lora_root   Absolute path to the clip_lora project directory.
                       Overrides null values in both config files.
    --data_path        Root path to the Flower102 dataset directory.
    --shots            Few-shot samples per class (overrides client config).
    --data_dist        Data distribution: "iid" or "non-iid".
    --pretrain_steps   Number of local gradient steps before consensus begins.
                       Default 0 (skip pre-training). Set >0 to enable.
"""

import argparse
import os
import time

from mpi4py import MPI
from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Decentralized federated CLIP + LoRA fine-tuning with dLoRA AB_SVD"
)
parser.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/flower102/server_dlora_ab_svd_ring.yaml",
    help="Path to server/shared configuration YAML (selects topology).",
)
parser.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/flower102/client_dlora_ab_svd_decentralized.yaml",
    help="Path to client configuration YAML (shared template).",
)
parser.add_argument(
    "--clip_lora_root",
    type=str,
    default="/work/mech-ai-scratch/nsaadati/projects/dlora/others/CLIP-LoRA",
    help="Absolute path to the clip_lora project directory.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/work/mech-ai-scratch/nsaadati/projects/dlora/others/CLIP-LoRA/data",
    help="Root data directory (parent of Flower102/). The dataset loader appends 'Flower102' internally.",
)
parser.add_argument(
    "--shots",
    type=int,
    default=40,
    help="Few-shot samples per class. Overrides client config.",
)
parser.add_argument(
    "--data_dist",
    type=str,
    default="iid",
    choices=["iid", "non-iid"],
    help="Data distribution. Overrides client config.",
)
parser.add_argument(
    "--pretrain_steps",
    type=int,
    default=0,
    help=(
        "Number of local gradient steps for pre-training before consensus. "
        "Default 0 (no pre-training). "
        "Requires train_configs.mode='step' in the client config."
    ),
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build results output path from args + config names
# Structure: ./results/<config_stem>/<data_dist>_shots<shots>/
#   config_stem — server config filename without extension (encodes topology)
#   data_dist   — iid / non-iid
#   shots       — few-shot samples per class
# ---------------------------------------------------------------------------

_config_stem = os.path.splitext(os.path.basename(args.server_config))[0]
# Strip leading "server_" prefix if present for brevity
if _config_stem.startswith("server_"):
    _config_stem = _config_stem[len("server_"):]

OUTPUT_DIR = os.path.join(
    "results",
    _config_stem,
    f"{args.data_dist}_shots{args.shots}",
)

# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1  # rank 0 is the server

# ======================================================================
# Server process (rank 0)
# ======================================================================

if rank == 0:
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients
    server_agent_config.server_configs.logging_output_dirname = OUTPUT_DIR
    server_agent_config.client_configs.train_configs.logging_output_dirname = OUTPUT_DIR

    # Propagate lora_rank from model_kwargs.r to aggregator_kwargs if not
    # already set explicitly, ensuring SVD rank matches the client LoRA rank.
    model_kwargs = server_agent_config.client_configs.model_configs.get(
        "model_kwargs", {}
    )
    if model_kwargs and "r" in model_kwargs:
        server_agent_config.server_configs.aggregator_kwargs.lora_rank = (
            model_kwargs.r
        )

    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    server_communicator.serve()

# ======================================================================
# Client processes (ranks 1 … N)
# ======================================================================

else:
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.client_id = f"Client{rank}"
    client_agent_config.train_configs.logging_output_dirname = OUTPUT_DIR

    # Per-client partitioning indices
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1

    # Command-line overrides for clip_lora path and dataset parameters
    if args.clip_lora_root is not None:
        client_agent_config.data_configs.dataset_kwargs.clip_lora_root = (
            args.clip_lora_root
        )
        # Also propagate to train_configs so CLIPLoRATrainer can find loralib
        client_agent_config.train_configs["clip_lora_root"] = args.clip_lora_root
    if args.data_path is not None:
        client_agent_config.data_configs.dataset_kwargs.root_path = args.data_path
    if args.shots is not None:
        client_agent_config.data_configs.dataset_kwargs.shots = args.shots
    if args.data_dist is not None:
        client_agent_config.data_configs.dataset_kwargs.data_dist = args.data_dist

    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(
        comm, server_rank=0, client_id=client_agent_config.client_id
    )

    # ------------------------------------------------------------------
    # Bootstrap: load config, initial model, and register sample size
    # ------------------------------------------------------------------
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=sample_size
    )

    print(
        f"[Client{rank}] Dataset partition ready — "
        f"{sample_size} training samples."
    )

    # ------------------------------------------------------------------
    # Phase 1: Local pre-training (no communication)
    # Pre-trains the LoRA adapters independently before any consensus.
    # ------------------------------------------------------------------
    pretrain_steps = args.pretrain_steps
    if pretrain_steps > 0:
        original_steps = client_agent.trainer.train_configs.num_local_steps

        client_agent.trainer.train_configs.num_local_steps = pretrain_steps
        print(
            f"[Client{rank}] Pre-training for {pretrain_steps} local steps ..."
        )
        t0 = time.time()
        client_agent.train()
        print(
            f"[Client{rank}] Pre-training done in {time.time() - t0:.1f}s"
        )

        client_agent.trainer.train_configs.num_local_steps = original_steps

    # ------------------------------------------------------------------
    # Phase 2: Decentralized consensus-training loop
    # Each round: local LoRA update → send params → receive AB_SVD-
    # aggregated params from topology neighborhood → repeat.
    # ------------------------------------------------------------------
    while True:
        t0 = time.time()
        client_agent.train()

        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}

        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )
        dt = time.time() - t0

        if metadata["status"] == "DONE":
            print(f"[Client{rank}] Training complete ({dt:.1f}s last round).")
            break

        if "local_steps" in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata[
                "local_steps"
            ]

        client_agent.load_parameters(new_global_model)

    client_communicator.invoke_custom_action(action="close_connection")

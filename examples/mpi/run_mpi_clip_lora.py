"""
MPI run script for dLoRA AB_SVD federated CLIP + LoRA fine-tuning.

Mirrors run_mpi_dimat_pretrain.py in structure:
  Rank 0  — server (runs the DLoRABSVDAggregator).
  Rank 1+ — clients (each runs CLIPLoRATrainer on its local data partition).

Usage (5 clients on a single node):
    mpirun -n 6 python run_mpi_clip_lora.py \\
        --server_config ./resources/configs/clip_lora/server_dlora_ab_svd.yaml \\
        --client_config ./resources/configs/clip_lora/client_dlora_ab_svd.yaml

Command-line arguments:
    --server_config   Path to the server YAML config.
    --client_config   Path to the client YAML config (shared template).
    --clip_lora_root  Absolute path to the clip_lora project directory.
                      Overrides null values in both config files.
    --dataset_name    Dataset name (overrides client config value).
    --root_path       Raw dataset root (overrides client config value).
    --shots           Few-shot samples per class (overrides client config).
"""

import argparse
import time
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Federated CLIP + LoRA fine-tuning with dLoRA AB_SVD"
)
parser.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/clip_lora/server_dlora_ab_svd.yaml",
)
parser.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/clip_lora/client_dlora_ab_svd.yaml",
)
parser.add_argument(
    "--clip_lora_root",
    type=str,
    default=None,
    help="Absolute path to the clip_lora project directory.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="Dataset name (e.g. 'dtd', 'caltech101'). Overrides config.",
)
parser.add_argument(
    "--root_path",
    type=str,
    default=None,
    help="Root path to raw dataset files. Overrides config.",
)
parser.add_argument(
    "--shots",
    type=int,
    default=None,
    help="Few-shot count per class. Overrides config.",
)
parser.add_argument(
    "--data_dist",
    type=str,
    default=None,
    choices=["iid", "non-iid"],
    help="Data distribution. Overrides config.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

# ---------------------------------------------------------------------------
# Server (rank 0)
# ---------------------------------------------------------------------------

if rank == 0:
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients

    # Propagate aggregator lora_rank from model_kwargs if not set explicitly
    # (ensures lora_rank matches the LoRA r used by clients)
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

# ---------------------------------------------------------------------------
# Clients (ranks 1 … N)
# ---------------------------------------------------------------------------

else:
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.client_id = f"Client{rank}"

    # Set per-client partitioning indices
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1

    # Command-line overrides for dataset location and parameters
    if args.clip_lora_root is not None:
        client_agent_config.data_configs.dataset_kwargs.clip_lora_root = (
            args.clip_lora_root
        )
        # Also propagate to model_kwargs so CLIPLoRAModel can find loralib
        client_agent_config.train_configs["clip_lora_root"] = args.clip_lora_root
    if args.dataset_name is not None:
        client_agent_config.data_configs.dataset_kwargs.dataset_name = (
            args.dataset_name
        )
    if args.root_path is not None:
        client_agent_config.data_configs.dataset_kwargs.root_path = args.root_path
    if args.shots is not None:
        client_agent_config.data_configs.dataset_kwargs.shots = args.shots
    if args.data_dist is not None:
        client_agent_config.data_configs.dataset_kwargs.data_dist = args.data_dist

    # Create client agent and communicator
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(
        comm, server_rank=0, client_id=client_agent_config.client_id
    )

    # Bootstrap: load config and initial global model from server
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)
    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    # Report local sample size for sample-weighted aggregation
    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=sample_size
    )

    print(
        f"[Client{rank}] Dataset partition ready — "
        f"{sample_size} training samples."
    )

    # -------------------------------------------------------------------
    # Federated training loop
    # -------------------------------------------------------------------
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

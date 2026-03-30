"""
MPI run script for DeCaF: federated CLIP + LoRA fine-tuning with dLoRA AB_SVD.

Usage:
    mpirun --oversubscribe --bind-to none -n 6 python mpi/run_mpi_decaf.py \\
        --server_config resources/configs/decaf/server_decaf.yaml \\
        --client_config resources/configs/decaf/client_decaf.yaml

Command-line arguments:
    --server_config    Path to server YAML config.
    --client_config    Path to shared client YAML config.
    --pretrain_steps   Number of local gradient steps before FL begins.
                       Default 0 (skip pre-training).
    --seed             Global random seed for reproducibility. Default 42.
"""

import argparse
import random
import time

import numpy as np
import torch
from mpi4py import MPI
from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

parser = argparse.ArgumentParser(
    description="Federated DeCaF: CLIP + LoRA fine-tuning with dLoRA AB_SVD"
)
parser.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/decaf/server_decaf.yaml",
)
parser.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/decaf/client_decaf.yaml",
)
parser.add_argument(
    "--shots",
    type=int,
    default=None,
    help="Few-shot samples per class. Overrides client config dataset_kwargs.shots.",
)
parser.add_argument(
    "--pretrain_steps",
    type=int,
    default=0,
    help=(
        "Number of local gradient steps for pre-training before FL begins. "
        "Requires train_configs.mode='step' in the client config."
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Global random seed. Each MPI rank gets seed+rank for reproducibility.",
)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

# Each rank gets a unique but reproducible seed.
_rank_seed = args.seed + rank
random.seed(_rank_seed)
np.random.seed(_rank_seed)
torch.manual_seed(_rank_seed)
torch.cuda.manual_seed_all(_rank_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================================
# Server process (rank 0)
# ======================================================================

if rank == 0:
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients

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
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    if args.shots is not None:
        client_agent_config.data_configs.dataset_kwargs.shots = args.shots

    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(
        comm, server_rank=0, client_id=client_agent_config.client_id
    )

    # Bootstrap: load config, initial model, and register sample size
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=sample_size
    )

    # Phase 1 (optional): local pre-training before FL begins
    if args.pretrain_steps > 0:
        original_steps = client_agent.trainer.train_configs.num_local_steps
        client_agent.trainer.train_configs.num_local_steps = args.pretrain_steps
        print(f"[Client{rank}] Pre-training for {args.pretrain_steps} steps ...")
        t0 = time.time()
        client_agent.train()
        print(f"[Client{rank}] Pre-training done in {time.time() - t0:.1f}s")
        client_agent.trainer.train_configs.num_local_steps = original_steps

    # Phase 2: FL training loop
    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}

        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )

        if metadata["status"] == "DONE":
            break
        if "local_steps" in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
        client_agent.load_parameters(new_global_model)

    client_communicator.invoke_custom_action(action="close_connection")

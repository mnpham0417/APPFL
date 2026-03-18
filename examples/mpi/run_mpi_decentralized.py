"""
MPI run script for Decentralized Learning with pre-training phase.

Supports both topology variants:
  - Fully connected (FC): all agents average parameters with equal weight.
  - Ring: each agent averages with its two ring neighbors (weight 1/3 each).

Usage (ring, CIFAR-100, 5 clients + 1 server = 6 MPI processes):
    mpirun -n 6 python run_mpi_decentralized.py \\
        --server_config ./resources/configs/cifar100/server_decentralized_ring.yaml \\
        --client_config ./resources/configs/cifar100/client_decentralized_ring.yaml \\
        --pretrain_epochs 100

Usage (FC, CIFAR-10, 5 clients + 1 server = 6 MPI processes):
    mpirun -n 6 python run_mpi_decentralized.py \\
        --server_config ./resources/configs/cifar10/server_decentralized_fc.yaml \\
        --client_config ./resources/configs/cifar10/client_decentralized_fc.yaml \\
        --pretrain_epochs 100

Execution phases:
    1. Pre-train each agent independently for ``--pretrain_epochs`` local
       epochs without any communication.
    2. Run the consensus-training loop: local train → send params to server
       → receive topology-averaged params → repeat.
"""

import argparse
import time

from mpi4py import MPI
from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/cifar100/server_decentralized_ring.yaml",
    help="Path to server/shared configuration YAML.",
)
parser.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/cifar100/client_decentralized_ring.yaml",
    help="Path to client configuration YAML.",
)
parser.add_argument(
    "--pretrain_epochs",
    type=int,
    default=100,
    help="Number of local pre-training epochs before consensus-training begins.",
)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1  # rank 0 is the server

# ======================================================================
# Server process
# ======================================================================
if rank == 0:
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients

    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    server_communicator.serve()

# ======================================================================
# Client processes
# ======================================================================
else:
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.client_id = f"Client{rank}"
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    # Enable visualization for the first client only
    if "visualization" in client_agent_config.data_configs.dataset_kwargs:
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            rank == 1
        )

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

    # ------------------------------------------------------------------
    # Phase 1: Local pre-training (no communication)
    # ------------------------------------------------------------------
    pretrain_epochs = args.pretrain_epochs
    if pretrain_epochs > 0:
        original_epochs = client_agent.trainer.train_configs.num_local_epochs

        client_agent.trainer.train_configs.num_local_epochs = pretrain_epochs
        print(
            f"[{client_agent_config.client_id}] "
            f"Pre-training for {pretrain_epochs} epochs ..."
        )
        t0 = time.time()
        client_agent.train()
        print(
            f"[{client_agent_config.client_id}] "
            f"Pre-training done in {time.time() - t0:.1f}s"
        )

        client_agent.trainer.train_configs.num_local_epochs = original_epochs

    # ------------------------------------------------------------------
    # Phase 2: Consensus-training loop
    # ------------------------------------------------------------------
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
            client_agent.trainer.train_configs.num_local_steps = metadata[
                "local_steps"
            ]

        client_agent.load_parameters(new_global_model)

    client_communicator.invoke_custom_action(action="close_connection")

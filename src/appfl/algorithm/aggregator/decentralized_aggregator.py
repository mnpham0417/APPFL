"""
DecentralizedAggregator: Server-side aggregator implementing decentralized
learning via consensus-based weighted parameter averaging.

Adapts decentralized SGD (CDSGD/SGP-style) algorithms to APPFL's
server-client architecture. The server performs the consensus step
centrally: for each client it computes the topology-weighted average of
its neighborhood's parameters, then returns a per-client model.

Supports two topologies:
  - "fc" (fully connected): Every agent communicates with every other
    agent. Each client receives the uniform average of all client models.
    Mixing weight = 1/N for all N clients.
  - "ring": Each agent communicates only with its two nearest ring
    neighbors (self, left, right). Each client receives the average of
    exactly 3 models. Mixing weight = 1/3 for each neighbor.

Both topologies return a dict mapping client_id -> per-client state dict,
so the server-agent dispatch logic works identically for both.

References:
    CDSGD: Consensus Decentralized SGD
    SGP:   Stochastic Gradient Push (Assran et al., ICML 2019)
"""

import copy
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from appfl.algorithm.aggregator import BaseAggregator


class DecentralizedAggregator(BaseAggregator):
    """
    Decentralized learning aggregator based on consensus-weighted parameter
    averaging over a network topology.

    aggregator_kwargs (YAML config):
        topology (str): "fc" (default) or "ring".
        client_weights_mode (str): "equal" (default) or "sample_size".
            "equal"       – uniform mixing weight 1/N for FC, 1/3 for ring.
            "sample_size" – weight proportional to each client's sample count
                            (only meaningful for FC topology; ring always uses
                            1/3 equal weights to match the decentralized
                            algorithm semantics).
        device (str): Device for computation. Default: "cpu".
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs

        self.topology = aggregator_configs.get("topology", "fc")
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )
        self.device = aggregator_configs.get("device", "cpu")

        # Per-client sample sizes (populated by set_sample_size calls)
        self.sample_sizes: Dict[str, int] = {}

        # Global state: for FC this is the single averaged model;
        # for ring this is the first client's model (for get_parameters()).
        self.global_state: Optional[Dict] = None

    # ------------------------------------------------------------------
    # BaseAggregator interface
    # ------------------------------------------------------------------

    def get_parameters(self, **kwargs) -> Dict:
        """Return a representative global state dict."""
        if self.global_state is not None:
            return self.global_state
        if self.model is not None:
            return self.model.state_dict()
        raise ValueError(
            "DecentralizedAggregator has no model or global state to return."
        )

    def aggregate(
        self, local_models: Dict, **kwargs
    ) -> Union[Dict, Dict[str, Dict]]:
        """
        Aggregate local models via topology-weighted parameter averaging.

        Args:
            local_models: dict mapping client_id -> state_dict

        Returns:
            dict mapping client_id -> per-client averaged state dict.
            (For FC topology all values are identical; for ring they differ.)
        """
        if self.topology == "ring":
            return self._aggregate_ring(local_models, **kwargs)
        else:
            return self._aggregate_fc(local_models, **kwargs)

    # ------------------------------------------------------------------
    # Topology implementations
    # ------------------------------------------------------------------

    def _aggregate_fc(
        self, local_models: Dict, **kwargs
    ) -> Dict[str, Dict]:
        """
        Fully-connected consensus: each client gets the weighted average
        of ALL client models.
        """
        client_ids = list(local_models.keys())
        num_clients = len(client_ids)

        # Compute per-client mixing weights
        weights = self._fc_weights(client_ids)

        # Compute the consensus average
        averaged_state = self._weighted_average(local_models, weights)

        if self.logger:
            self.logger.info(
                f"[Decentralized-FC] Averaged {num_clients} client models "
                f"(weights_mode={self.client_weights_mode})"
            )

        # Store as global state
        self.global_state = averaged_state

        # All clients receive the same global model
        return {cid: copy.deepcopy(averaged_state) for cid in client_ids}

    def _aggregate_ring(
        self, local_models: Dict, **kwargs
    ) -> Dict[str, Dict]:
        """
        Ring topology consensus: each client gets the average of its own
        model plus the models of its two ring neighbors (weight = 1/3 each).
        """
        client_ids = list(local_models.keys())
        num_agents = len(client_ids)

        # Build ring adjacency: pi[i] = {j: weight} for j in neighborhood
        pi = self._build_ring_pi(num_agents)  # list of dicts

        per_client_states: Dict[str, Dict] = {}
        for agent_idx, cid in enumerate(client_ids):
            neighbor_weights = pi[agent_idx]  # {j_idx: weight}

            # Collect neighborhood state dicts with their weights
            neighborhood: Dict[str, Dict] = {}
            nbr_weights_by_id: Dict[str, float] = {}
            for j_idx, w in neighbor_weights.items():
                nbr_cid = client_ids[j_idx]
                neighborhood[nbr_cid] = local_models[nbr_cid]
                nbr_weights_by_id[nbr_cid] = w

            averaged = self._weighted_average(neighborhood, nbr_weights_by_id)
            per_client_states[cid] = averaged

            if self.logger:
                neighbor_names = [client_ids[j] for j in neighbor_weights]
                self.logger.info(
                    f"[Decentralized-Ring] Agent {cid} consensus with "
                    f"neighbors {neighbor_names} "
                    f"(weights {list(neighbor_weights.values())})"
                )

        # Store first client's result as representative global state
        self.global_state = per_client_states[client_ids[0]]

        return per_client_states

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fc_weights(self, client_ids) -> Dict[str, float]:
        """Compute per-client mixing weights for FC topology."""
        if (
            self.client_weights_mode == "sample_size"
            and self.sample_sizes
        ):
            total = sum(
                self.sample_sizes.get(cid, 1) for cid in client_ids
            )
            return {
                cid: self.sample_sizes.get(cid, 1) / total
                for cid in client_ids
            }
        # Default: equal weights
        w = 1.0 / len(client_ids)
        return {cid: w for cid in client_ids}

    @staticmethod
    def _build_ring_pi(num_agents: int):
        """
        Build ring mixing weights.
        Returns a list of dicts: pi[i] = {j: 1/3} for j in {i-1, i, i+1}.
        """
        pi = []
        for i in range(num_agents):
            neighbors = [
                (i - 1) % num_agents,
                i,
                (i + 1) % num_agents,
            ]
            pi.append({j: 1.0 / len(neighbors) for j in neighbors})
        return pi

    @staticmethod
    def _weighted_average(
        state_dicts: Dict[str, Dict],
        weights: Dict[str, float],
    ) -> Dict:
        """
        Compute a weighted average of state dicts.

        Args:
            state_dicts: {client_id: state_dict}
            weights:     {client_id: float}  (should sum to ~1.0)

        Returns:
            Averaged state dict (all tensors on CPU).
        """
        averaged: Dict = {}
        for cid, sd in state_dicts.items():
            w = weights[cid]
            for k, v in sd.items():
                tensor = v.float()
                if k not in averaged:
                    averaged[k] = w * tensor
                else:
                    averaged[k] = averaged[k] + w * tensor

        # Cast back to original dtypes
        first_sd = next(iter(state_dicts.values()))
        for k in averaged:
            averaged[k] = averaged[k].to(first_sd[k].dtype).cpu()

        return averaged

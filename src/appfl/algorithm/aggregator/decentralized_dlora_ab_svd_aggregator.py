"""
DecentralizedDLoRABSVDAggregator: Server-side aggregator implementing
federated CLIP + LoRA fine-tuning via AB_SVD aggregation.

Performs standard federated (fully-connected) aggregation with AB_SVD:

  For each LoRA pair (w_lora_A, w_lora_B):
    1. Compute the (optionally sample-size-weighted) average of B @ A.
    2. Truncated SVD:  (B@A)_avg ≈ U_r * diag(S_r) * Vh_r
    3. Symmetric split:
         new_B = U_r  * sqrt(S_r)    shape: (out_dim, r)
         new_A = sqrt(S_r) * Vh_r    shape: (r, in_dim)
  All other parameters: standard weighted average (FedAvg).

References:
    dLoRA AB_SVD: clip_lora/dlora_utils/fed_global.py (fedavg_AB_SVD)
"""

import copy
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from appfl.algorithm.aggregator import BaseAggregator


class DecentralizedDLoRABSVDAggregator(BaseAggregator):
    """
    Federated LoRA aggregator using AB_SVD aggregation.

    All clients contribute to a single global aggregation (fully-connected).
    The aggregated model is sent back to every client.

    aggregator_kwargs (YAML config):
        lora_rank (int):
            SVD truncation rank. Must match the LoRA rank r used by all
            clients. Default: 2.
        client_weights_mode (str):
            "equal"       – uniform mixing weight (default).
            "sample_size" – weight proportional to dataset size.
        device (str):
            Device for SVD computation. Default: "cuda" if available else "cpu".
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

        self.lora_rank = aggregator_configs.get("lora_rank", 2)
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )
        _default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = aggregator_configs.get("device", _default_device)

        # Per-client sample sizes (populated by set_sample_size calls)
        self.sample_sizes: Dict[str, int] = {}

        # Representative global state (returned by get_parameters)
        self.global_state: Optional[Dict] = None

    # ------------------------------------------------------------------
    # BaseAggregator interface
    # ------------------------------------------------------------------

    def get_parameters(self, **kwargs) -> Dict:
        """Return the current global state dict."""
        if self.global_state is not None:
            return self.global_state
        if self.model is not None:
            return self.model.state_dict()
        raise ValueError(
            "DecentralizedDLoRABSVDAggregator has no model or global state."
        )

    def aggregate(self, local_models: Dict, **kwargs) -> Dict[str, Dict]:
        """
        Aggregate local LoRA models using AB_SVD federated aggregation.

        Args:
            local_models: dict mapping client_id -> full model state_dict.

        Returns:
            Dict mapping every client_id to the same aggregated state dict.
        """
        client_ids = list(local_models.keys())
        weights = self._compute_weights(client_ids)
        averaged_state = self._ab_svd_aggregate(local_models, weights)

        if self.logger:
            self.logger.info(
                f"[DeCaF-DLoRA] AB_SVD over {len(client_ids)} clients "
                f"(lora_rank={self.lora_rank}, weights_mode={self.client_weights_mode})"
            )

        self.global_state = averaged_state
        return {cid: copy.deepcopy(averaged_state) for cid in client_ids}

    # ------------------------------------------------------------------
    # Core AB_SVD aggregation
    # ------------------------------------------------------------------

    def _ab_svd_aggregate(
        self,
        local_models: Dict[str, Dict],
        weights: Dict[str, float],
    ) -> Dict:
        """
        AB_SVD aggregation over a set of local models with given weights.

        For w_lora_A / w_lora_B pairs:
            weighted_product = sum_i( weights[i] * B_i @ A_i )
            U_r, S_r, Vh_r  = truncated_SVD(weighted_product.T, r=lora_rank)
            new_B = U_r * sqrt(S_r)   (in_dim, r) -> transpose matches B shape
            new_A = sqrt(S_r) * Vh_r  (r, out_dim)

        For all other parameters: weighted average (FedAvg).
        """
        client_ids = list(local_models.keys())
        first_state = local_models[client_ids[0]]

        global_dict: Dict = {}
        written_lora_b_keys = set()

        for key in first_state.keys():
            # w_lora_B keys are handled together with their paired w_lora_A key
            if "w_lora_B" in key:
                continue

            if "w_lora_A" in key:
                layer_prefix = key.replace("w_lora_A", "")
                key_A = f"{layer_prefix}w_lora_A"
                key_B = f"{layer_prefix}w_lora_B"

                # Weighted sum of B @ A products across clients
                summed_product = None
                for cid in client_ids:
                    A = local_models[cid][key_A]
                    B = local_models[cid][key_B]
                    product = B @ A  # (out_dim, in_dim)
                    scaled = product * weights[cid]
                    if summed_product is None:
                        summed_product = scaled.clone()
                    else:
                        summed_product = summed_product + scaled

                summed_product = summed_product.to(self.device)

                # Truncated SVD on the transposed product
                # summed_product.t() shape: (in_dim, out_dim)
                U, S, Vh = torch.linalg.svd(
                    summed_product.t(), full_matrices=False
                )
                r = self.lora_rank
                U_r = U[:, :r]      # (in_dim, r)
                S_r = S[:r]         # (r,)
                Vh_r = Vh[:r, :]    # (r, out_dim)
                sqrt_S_r = torch.sqrt(S_r)

                # Symmetric split — shapes match original A and B
                new_B = U_r * sqrt_S_r.unsqueeze(0)         # (in_dim, r)
                new_A = sqrt_S_r.unsqueeze(1) * Vh_r        # (r, out_dim)

                global_dict[key_B] = new_B.cpu()
                global_dict[key_A] = new_A.cpu()
                written_lora_b_keys.add(key_B)

            else:
                # Standard weighted average for non-LoRA parameters
                param = first_state[key]
                if param.dtype in (torch.int32, torch.int64, torch.bool):
                    global_dict[key] = param.clone()
                else:
                    aggregated = sum(
                        local_models[cid][key].float() * weights[cid]
                        for cid in client_ids
                    )
                    global_dict[key] = aggregated.to(param.dtype).cpu()

        # Safety pass: handle any w_lora_B keys not yet written
        for key in first_state.keys():
            if (
                "w_lora_B" in key
                and key not in written_lora_b_keys
                and key not in global_dict
            ):
                param = first_state[key]
                aggregated = sum(
                    local_models[cid][key].float() * weights[cid]
                    for cid in client_ids
                )
                global_dict[key] = aggregated.to(param.dtype).cpu()

        return global_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_weights(self, client_ids: List[str]) -> Dict[str, float]:
        """Compute per-client mixing weights."""
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
        w = 1.0 / len(client_ids)
        return {cid: w for cid in client_ids}

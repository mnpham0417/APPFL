"""
DLoRABSVDAggregator: Server-side aggregator implementing the dLoRA AB_SVD
federated aggregation method for CLIP + LoRA fine-tuning.

For each LoRA layer pair (w_lora_A, w_lora_B):
  1. Compute the sample-weighted average of the full-rank product B @ A
     across all participating clients.
  2. Factorize that average via truncated SVD:
         (B@A)_avg ≈ U_r * diag(S_r) * Vh_r
  3. Redistribute the singular values symmetrically:
         new_B = U_r  * sqrt(S_r)    shape: (out_dim, r)
         new_A = sqrt(S_r) * Vh_r    shape: (r, in_dim)

For all other parameters: standard sample-weighted average (FedAvg).

This mirrors the `fedavg_AB_SVD` branch in
clip_lora/dlora_utils/fed_global.py, adapted to APPFL's
server-aggregator interface.

Reference:
    dLoRA: Federated LoRA fine-tuning with AB_SVD aggregation
    (clip_lora/dlora_utils/fed_global.py, lines 21-54)
"""

import torch
from omegaconf import DictConfig
from typing import Dict, Union, Optional, Any

from appfl.algorithm.aggregator import BaseAggregator


class DLoRABSVDAggregator(BaseAggregator):
    """
    DLoRA AB_SVD aggregator for CLIP + LoRA federated fine-tuning.

    Required aggregator_kwargs in YAML config:
        lora_rank (int): SVD truncation rank. Must match the LoRA rank r
            used by all clients. Default: 2.

    Optional aggregator_kwargs:
        device (str): Device for SVD computation. Default: "cuda" if
            available else "cpu".
        client_weights_mode (str): Weighting strategy for aggregation.
            "sample_size" uses per-client dataset sizes (recommended);
            "equal" weights all clients equally. Default: "sample_size".
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
            "client_weights_mode", "sample_size"
        )
        _default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = aggregator_configs.get("device", _default_device)

        self.global_state = None
        # Populated by the server via the "set_sample_size" custom action
        # (same mechanism as FedAvgAggregator).
        self.client_sample_size = {}

    def get_parameters(self, **kwargs) -> Dict:
        """Return the current global model state dict."""
        if self.global_state is not None:
            return self.global_state
        elif self.model is not None:
            return self.model.state_dict()
        else:
            raise ValueError(
                "DLoRABSVDAggregator has no model or global state to return."
            )

    def aggregate(
        self,
        local_models: Dict[Union[str, int], Dict],
        **kwargs,
    ) -> Dict:
        """
        Aggregate local LoRA models using the AB_SVD method.

        For each w_lora_A / w_lora_B key pair:
            - Weighted sum of B @ A across clients.
            - Truncated SVD of the averaged product (transposed).
            - Split singular values symmetrically into new A and B.

        For all other keys:
            - Standard sample-weighted average.

        Args:
            local_models: dict mapping client_id -> full model state_dict.

        Returns:
            Aggregated global state dict (also stored in self.global_state).
        """
        client_ids = list(local_models.keys())
        num_clients = len(client_ids)

        # --- Compute per-client mixture weights ---
        if (
            self.client_weights_mode == "sample_size"
            and self.client_sample_size
        ):
            total_samples = sum(
                self.client_sample_size.get(cid, 1) for cid in client_ids
            )
            weights = {
                cid: self.client_sample_size.get(cid, 1) / total_samples
                for cid in client_ids
            }
        else:
            weights = {cid: 1.0 / num_clients for cid in client_ids}

        if self.logger:
            self.logger.info(
                f"[DLoRA-AB-SVD] Aggregating {num_clients} client models "
                f"with lora_rank={self.lora_rank}, device={self.device}, "
                f"weights_mode={self.client_weights_mode}"
            )

        first_state = list(local_models.values())[0]
        global_dict = {}
        # Track B-keys already written during the A-key pass.
        written_lora_b_keys = set()

        for key in first_state.keys():
            # w_lora_B keys are handled together with their paired w_lora_A key.
            if "w_lora_B" in key:
                continue

            if "w_lora_A" in key:
                # --- AB_SVD aggregation for this LoRA layer ---
                layer_prefix = key.replace("w_lora_A", "")
                key_A = f"{layer_prefix}w_lora_A"
                key_B = f"{layer_prefix}w_lora_B"

                # Weighted sum of B @ A products
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

                # Truncated SVD: summed_product.t() ≈ U * diag(S) * Vh
                # (transposing matches the original fed_global.py convention)
                U, S, Vh = torch.linalg.svd(
                    summed_product.t(), full_matrices=False
                )
                U_r = U[:, : self.lora_rank]       # (in_dim, r)
                S_r = S[: self.lora_rank]           # (r,)
                Vh_r = Vh[: self.lora_rank, :]      # (r, out_dim)
                sqrt_S_r = torch.sqrt(S_r)

                # Symmetric split: new_B (out_dim, r), new_A (r, in_dim)
                new_B = U_r * sqrt_S_r.unsqueeze(0)        # broadcast (in_dim, r)
                new_A = sqrt_S_r.unsqueeze(1) * Vh_r       # (r, out_dim)

                global_dict[key_B] = new_B.cpu()
                global_dict[key_A] = new_A.cpu()
                written_lora_b_keys.add(key_B)

            else:
                # --- Standard weighted average for non-LoRA parameters ---
                param = first_state[key]
                if param.dtype in (torch.int32, torch.int64, torch.bool):
                    # Integer/bool buffers: take from first client unchanged
                    global_dict[key] = param.clone()
                else:
                    aggregated = sum(
                        local_models[cid][key].float() * weights[cid]
                        for cid in client_ids
                    )
                    global_dict[key] = aggregated.to(param.dtype).cpu()

        # Safety pass: handle any w_lora_B keys not yet written
        # (can occur if a key_A had no matching key_B in the state dict)
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

        self.global_state = global_dict

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)

        return self.global_state

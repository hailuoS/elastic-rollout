"""Only the device and native weight-transfer engine differ across transports."""


class CollectiveBackend:
    def __init__(self, name):
        if name not in ("hccl", "nccl"):
            raise ValueError("collective backend must be hccl or nccl")
        self.name = name
        self.device = "npu" if name == "hccl" else "cuda"

    def engine(self):
        if self.name == "hccl":
            from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

            return HCCLWeightTransferEngine
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        return NCCLWeightTransferEngine

    def synchronize(self):
        import torch

        getattr(torch, self.device).current_stream().synchronize()

    def destroy(self, group):
        """Confirm native teardown on the actor's device thread before slot reuse."""
        library = getattr(group, self.name, None)
        destroy = getattr(library, f"{self.name}CommDestroy", None)
        if not callable(destroy) or getattr(group, "comm", None) is None:
            raise RuntimeError(
                f"native {self.name.upper()} destroy unavailable; receiver remains quarantined"
            )
        # vLLM's NCCL destroy() uses a background abort with a timeout. That
        # cannot prove retirement completed. Our transfer groups use no graphs.
        self.synchronize()
        result = destroy(group.comm)
        if result not in (None, 0):
            raise RuntimeError(f"native {self.name.upper()} destroy failed: {result}")
        group.comm, group.available, group.disabled = None, False, True

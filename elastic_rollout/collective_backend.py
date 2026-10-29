"""Only the device and native weight-transfer engine differ across transports."""

from .transfer_config import validate_backend


class CollectiveBackend:
    def __init__(self, name):
        validate_backend(name)
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

        # Native packed engines create their own streams. A current-stream
        # fence cannot protect bucket lifetime or the next FSDP allocation.
        getattr(torch, self.device).synchronize()

    def initialize(self, session):
        if self.name == "nccl":
            import torch

            # NCCL allocates outside PyTorch's caching allocator. Training can
            # retain almost all free VRAM in that cache before a dynamic join.
            # Reclaim unused blocks once per new group, never on group reuse.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return self.engine().trainer_init(session)

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
        import torch

        # Packed transfers can use auxiliary streams. Retirement must fence
        # all device work, not just the current stream's enqueue boundary.
        getattr(torch, self.device).synchronize()
        result = destroy(group.comm)
        if result not in (None, 0):
            raise RuntimeError(f"native {self.name.upper()} destroy failed: {result}")
        group.comm, group.available, group.disabled = None, False, True

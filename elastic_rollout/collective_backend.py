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

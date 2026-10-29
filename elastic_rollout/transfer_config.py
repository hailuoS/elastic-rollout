"""Shared Direct transport options and boundary validation."""

from dataclasses import dataclass

METADATA_LIMIT = 32768


def validate_backend(backend):
    if backend not in ("hccl", "nccl"):
        raise ValueError("backend must be hccl or nccl")


def validate_rendezvous(address, port, *, required=False):
    if (required or address is not None) and (not isinstance(address, str) or not address.strip()):
        raise ValueError("master_address must name the trainer rank0 interface")
    if (required or port is not None) and (type(port) is not int or not 1 <= port <= 65535):
        raise ValueError("master_port must be between 1 and 65535")


def validate_packing(bucket_size, packed, num_buffers):
    if type(bucket_size) is not int or bucket_size <= 0:
        raise ValueError("bucket_size must be a positive integer")
    if type(packed) is not bool:
        raise ValueError("packed must be boolean")
    if type(num_buffers) is not int or num_buffers <= 0:
        raise ValueError("packed_num_buffers must be positive")


@dataclass(frozen=True)
class TransferOptions:
    bucket_size: int = 1024 * 1024 * 1024
    packed: bool = True
    packed_num_buffers: int = 2
    master_address: str | None = None
    master_port: int | None = None
    backend: str = "hccl"
    metadata_group_limit_bytes: int = METADATA_LIMIT

    def __post_init__(self):
        validate_backend(self.backend)
        validate_packing(self.bucket_size, self.packed, self.packed_num_buffers)
        validate_rendezvous(self.master_address, self.master_port)
        if not 256 <= self.metadata_group_limit_bytes <= METADATA_LIMIT:
            raise ValueError("metadata_group_limit_bytes must be between 256 and 32768")

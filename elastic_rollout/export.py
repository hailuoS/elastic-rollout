"""Framework-neutral coordination of lazy, all-rank parameter exports."""


def synchronize_manifest(dist, group, error, manifest):
    values = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(values, {"error": error, "manifest": manifest}, group=group)
    errors = [value["error"] for value in values if value["error"]]
    if errors:
        raise RuntimeError(f"distributed HCCL export failed: {errors}")
    if any(value["manifest"] != values[0]["manifest"] for value in values):
        raise RuntimeError("training ranks produced different HCCL weight manifests")


def coordinated_export(weights, synchronize, *, transform=None, include_dtype=True):
    """Compare each item/end/error before the next native collective; always close."""
    weights = iter(weights)
    try:
        while True:
            item = manifest = error = None
            try:
                name, tensor = next(weights)
                if transform is not None:
                    tensor = transform(tensor)
                item = (name, tensor)
                manifest = {"name": name, "shape": list(tensor.shape)}
                if include_dtype:
                    manifest["dtype"] = str(tensor.dtype)
            except StopIteration:
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            synchronize(error, manifest)
            if item is None:
                return
            yield item
    finally:
        close = getattr(weights, "close", None)
        if close is not None:
            close()

"""vLLM worker extension for synchronous, acknowledged transfer retirement."""

from elastic_rollout.collective_backend import CollectiveBackend


class WeightTransferWorker:
    def elastic_weight_transfer_capability(self):
        if not hasattr(getattr(self, "weight_transfer_engine", None), "model_update_group"):
            raise RuntimeError("native weight-transfer engine does not support retirement")
        return {"retirement_protocol": 1}

    def elastic_destroy_weight_transfer(self, backend, retirement_id):
        # This RPC executes on every native worker/device thread. Only the
        # dedicated weight-transfer group is retired; TP groups remain intact.
        previous = getattr(self, "_elastic_retirement_id", None)
        if previous is not None:
            if previous != retirement_id:
                raise RuntimeError("weight-transfer group already retired with another identity")
            return {"retirement_id": previous}
        engine = self.weight_transfer_engine
        CollectiveBackend(backend).destroy(engine.model_update_group)
        engine.model_update_group = None
        self._elastic_retirement_id = retirement_id
        return {"retirement_id": retirement_id}

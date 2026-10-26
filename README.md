# ElasticRollout

ElasticRollout provides a framework-independent control plane for external vLLM replicas.

- Static logical capacity slots; Ray owns physical NPU allocation.
- Desired-state reconciliation, health checks, warmup and failed-replica replacement.
- Sticky least-inflight routing with expiring and renewable request leases.
- Graceful scale-in waits for active leases and backend running/waiting requests.
- Process supervision and resource quarantine when cleanup cannot be confirmed.
- HTTP control API, isolated lease-renewal executor and a command-line interface.
- Direct token generation with log probabilities and per-token policy versions.
- Optional AReaL workflow and verl AgentLoop adapters share the same runtime.

This branch implements fixed-weight inference with single-node replicas (PP=1).
It excludes training weight updates, HCCL transport and pipeline autoscaling.
Core uses the Python standard library; Ray and RL frameworks are optional integrations.
Framework adapters require compatible external-server extension hooks in the host framework.
They do not install or patch either framework.

# ADR-0003: Use Redis Streams for Real-time Signal Delivery

## Status

**Accepted** (2026-01-14)

## Context

The USDCOP RL Trading system needs to deliver trading signals from the inference engine to multiple consumers (dashboard, paper trader, alerting system) in real-time with low latency.

Requirements:
- Sub-100ms delivery latency
- Multiple consumers for the same signals
- Message persistence for replay/debugging
- Ordered message delivery
- Consumer group support for horizontal scaling

Options considered:
1. PostgreSQL NOTIFY/LISTEN
2. Redis Pub/Sub
3. Redis Streams
4. Apache Kafka
5. RabbitMQ

## Decision

We chose **Redis Streams** for real-time signal delivery.

## Rationale

### Why Redis Streams over alternatives:

| Criterion | Redis Streams | Kafka | RabbitMQ | Redis Pub/Sub | PG NOTIFY |
|-----------|--------------|-------|----------|---------------|-----------|
| Latency | <1ms | ~5ms | ~2ms | <1ms | ~10ms |
| Persistence | Yes | Yes | Optional | No | No |
| Consumer Groups | Yes | Yes | Yes | No | No |
| Complexity | Low | High | Medium | Low | Low |
| Existing Infra | Yes (Redis) | No | No | Yes | Yes |

### Key advantages:

1. **Already have Redis** - Used for caching, no new infrastructure
2. **Built-in persistence** - Messages stored until acknowledged
3. **Consumer groups** - Multiple dashboard instances can share load
4. **XREAD BLOCK** - Efficient long polling for consumers
5. **Automatic ID** - Time-based IDs for ordering and deduplication

### Why not Kafka?

While Kafka offers superior durability and throughput, it introduces:
- Significant operational complexity
- Additional infrastructure (ZooKeeper/KRaft)
- Higher latency for low-volume use case
- Overkill for ~60 messages/day

## Consequences

### Positive

- Simple deployment (Redis already deployed)
- Low latency signal delivery
- Built-in message replay for debugging
- Consumer group support for scaling

### Negative

- Redis is single-threaded (potential bottleneck at scale)
- No cross-datacenter replication without Redis Cluster
- Limited message retention (we set 1000 messages max)

### Mitigations

- Monitor Redis CPU utilization
- Configure appropriate MAXLEN for streams
- Plan migration path to Kafka if volume exceeds 10K messages/day

## Implementation

```python
# Producer (inference engine)
redis.xadd(
    "signals:ppo_primary:stream",
    {"signal": "LONG", "confidence": "0.85"},
    maxlen=1000
)

# Consumer (dashboard)
messages = redis.xread(
    {"signals:ppo_primary:stream": last_id},
    block=5000
)
```

## References

- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)
- [Redis Streams vs Kafka](https://redis.com/blog/redis-streams-vs-kafka/)

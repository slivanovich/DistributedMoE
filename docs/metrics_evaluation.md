# DistMoE Metrics Evaluation System

## Overview

The DistMoE Metrics Evaluation System provides comprehensive performance monitoring and analysis capabilities for distributed Mixture of Experts (MoE) operations. This system collects detailed timing, throughput, and operational metrics during MoE forward passes, enabling performance analysis, optimization, and debugging.

## Architecture

### Core Components

1. **MetricsCollector**: Central component for collecting and aggregating metrics
2. **Metrics Data Structures**: Hierarchical data structures representing different levels of metrics
3. **Integration Points**: Hooks in DistMoEBlock and PipelineTask classes for automatic metrics collection
4. **Export System**: Multiple export formats (JSON, CSV, summary reports)

### Metrics Hierarchy

```
Metrics
├── ForwardMetrics (per forward pass)
│   ├── DispatchMetrics (per dispatch operation)
│   ├── CombineMetrics (per combine operation)
│   └── ExpertsBlockMetrics (per expert block)
│       └── ExpertsRunMetrics (per expert run)
```

## Data Structures

### Core Metrics Classes

#### `DispatchMetrics`
```python
@dataclass
class DispatchMetrics:
    e2e_latency: float  # End-to-end latency in milliseconds
```

#### `CombineMetrics`
```python
@dataclass
class CombineMetrics:
    e2e_latency: float  # End-to-end latency in milliseconds
```

#### `ExpertsRunMetrics`
```python
@dataclass
class ExpertsRunMetrics:
    e2e_latency: float  # End-to-end latency in milliseconds
    read_transfer_throughput: float  # Read throughput in MB/ms
    write_transfer_throughput: float  # Write throughput in MB/ms
```

#### `ExpertsBlockMetrics`
```python
@dataclass
class ExpertsBlockMetrics:
    number_of_runs: int
    number_of_transfers: int  # Total read + write transfers
    avg_e2e_latency: float  # Average end-to-end latency in milliseconds
    avg_transfers_throughput: float  # Average transfer throughput in MB/ms
    experts_runs: List[ExpertsRunMetrics]  # Individual expert run metrics
```

#### `ForwardMetrics`
```python
@dataclass
class ForwardMetrics:
    e2e_latency: float  # Total forward pass latency in milliseconds
    setup_latency: float  # Setup phase latency in milliseconds
    avg_dispatches_e2e_latency: float  # Average dispatch latency
    avg_combines_e2e_latency: float  # Average combine latency
    dispatches: List[DispatchMetrics]
    combines: List[CombineMetrics]
    experts_blocks: List[ExpertsBlockMetrics]
```

#### `Metrics`
```python
@dataclass
class Metrics:
    number_of_forwards: int
    forwards: List[ForwardMetrics]
```

## Configuration

### MetricsCollectorConfig

```python
@dataclass
class MetricsCollectorConfig:
    enable_collection: bool = True  # Enable/disable all metrics collection
    enable_detailed_timing: bool = True  # Collect individual dispatch/combine timing
    enable_transfer_metrics: bool = True  # Collect transfer throughput metrics
    enable_expert_block_metrics: bool = True  # Collect expert block metrics
    max_stored_forwards: int = 1000  # Maximum forward passes to store in memory
    auto_export_interval: Optional[int] = None  # Auto-export every N forwards
    export_directory: str = "metrics_output"  # Directory for exported files
```

#### Configuration Options Explained

- **`enable_collection`**: Master switch for all metrics collection. When `False`, no metrics are collected regardless of other settings.

- **`enable_detailed_timing`**: Controls collection of individual dispatch and combine operation timings. When `True`, each dispatch and combine operation is timed individually and stored in the `dispatches` and `combines` lists of `ForwardMetrics`. When `False`, only overall forward pass timing is collected, reducing memory usage and overhead.

- **`enable_transfer_metrics`**: Controls collection of transfer throughput metrics during expert block operations. When `True`, read and write transfer performance is measured and stored in `ExpertsRunMetrics`.

- **`enable_expert_block_metrics`**: Controls collection of expert block execution metrics. When `True`, detailed metrics about expert block runs are collected including latency and transfer counts.

- **`max_stored_forwards`**: Maximum number of forward pass metrics to keep in memory. Older metrics are automatically discarded when this limit is exceeded.

- **`auto_export_interval`**: If set, metrics are automatically exported every N forward passes. Useful for long-running experiments to prevent data loss.

- **`export_directory`**: Directory where exported metrics files are saved.

## Usage

### Basic Setup

```python
from MoE.MetricsCollector import MetricsCollector, MetricsCollectorConfig

# Create configuration
config = MetricsCollectorConfig(
    enable_collection=True,
    export_directory="./metrics_output"
)

# Create metrics collector
metrics_collector = MetricsCollector(config)

# Create DistMoEBlock with metrics collection
moe_block = DistMoEBlock(moe_config, tte_config, metrics_collector)
```

### Global Metrics Collector

```python
from MoE.MetricsCollector import initialize_metrics_collector, get_metrics_collector

# Initialize global collector
initialize_metrics_collector(config)

# Access global collector anywhere
collector = get_metrics_collector()
```

### Manual Metrics Collection

```python
# Start forward pass metrics
metrics_collector.start_forward_pass()

# Mark end of setup phase
metrics_collector.end_setup_phase()

# Collect dispatch/combine metrics
task_id = "unique_task_id"
metrics_collector.start_dispatch(task_id)
# ... dispatch operation ...
metrics_collector.end_dispatch(task_id)

metrics_collector.start_combine(task_id)
# ... combine operation ...
metrics_collector.end_combine(task_id)

# Collect expert block metrics
metrics_collector.start_expert_run(task_id, experts_block_index)
# ... expert block execution ...
metrics_collector.end_expert_run(task_id)

# End forward pass
metrics_collector.end_forward_pass()
```

## Data Access and Analysis

### Getting Raw Metrics

```python
# Get complete metrics object
metrics = metrics_collector.get_metrics()
print(f"Total forwards: {metrics.number_of_forwards}")

# Access individual forward metrics
for forward in metrics.forwards:
    print(f"Forward E2E: {forward.e2e_latency:.2f}ms")
    print(f"Setup: {forward.setup_latency:.2f}ms")
    print(f"Dispatches: {len(forward.dispatches)}")
    print(f"Combines: {len(forward.combines)}")
```

### Aggregated Statistics

```python
# Get aggregated statistics
stats = metrics_collector.get_aggregated_statistics()

print(f"Total forwards: {stats['total_forwards']}")
print(f"Average E2E latency: {stats['e2e_latency']['mean']:.2f}ms")
print(f"Average dispatch latency: {stats['dispatch_latency']['mean']:.2f}ms")
print(f"Average combine latency: {stats['combine_latency']['mean']:.2f}ms")
print(f"Average read throughput: {stats['read_throughput']['mean']:.2f} MB/ms")
```

### Summary Report

```python
# Generate human-readable summary
report = metrics_collector.get_summary_report()
print(report)
```

Example output:
```
=== DistMoE Metrics Summary ===
Total Forward Passes: 100

E2E Latency (ms):
  Mean: 45.23, Min: 32.10, Max: 67.89
  Median: 44.56, Std: 8.45

Setup Latency (ms):
  Mean: 2.34, Min: 1.89, Max: 3.45

Dispatch Latency (ms):
  Mean: 12.45, Min: 8.90, Max: 18.23

Combine Latency (ms):
  Mean: 8.67, Min: 6.12, Max: 12.34

Transfer Throughput (MB/ms):
  Read - Mean: 125.67, Min: 98.45, Max: 156.78
  Write - Mean: 118.23, Min: 89.67, Max: 145.89

Operation Counts:
  Avg Dispatches per Forward: 4.2
  Avg Combines per Forward: 4.2
  Avg Expert Blocks per Forward: 2.1
```

## Export Functionality

### Export Formats

The system supports multiple export formats:

1. **JSON**: Complete metrics data with full hierarchy
2. **CSV**: Flattened summary data for spreadsheet analysis
3. **Statistics JSON**: Aggregated statistics only

### Exporting Metrics

```python
# Export all metrics
exported_files = metrics_collector.export_metrics("experiment_1")

# Returns dictionary with file paths
# {
#   "json": "/path/to/experiment_1_timestamp.json",
#   "csv": "/path/to/experiment_1_summary_timestamp.csv", 
#   "stats": "/path/to/experiment_1_stats_timestamp.json"
# }
```

### Auto-Export

```python
# Configure auto-export every 50 forward passes
config = MetricsCollectorConfig(
    auto_export_interval=50,
    export_directory="./auto_exports"
)
```

## Integration with Existing Code

### DistMoEBlock Integration

The metrics collection is automatically integrated into [`DistMoEBlock`](../src/python/MoE/DistMoEBlock.py) when a metrics collector is provided:

```python
# Metrics collection happens automatically during forward passes
moe_block = DistMoEBlock(config, tte_config, metrics_collector)
output = moe_block(input_batch)  # Metrics collected automatically
```

### PipelineTask Integration

Both [`HostPipelineTask`](../src/python/MoE/PipelineTask.py) and [`ExpertsBlockPipelineTask`](../src/python/MoE/PipelineTask.py) automatically collect metrics when a metrics collector is available.

## Performance Considerations

### Memory Usage

- The system stores metrics for up to `max_stored_forwards` forward passes
- Older metrics are automatically discarded when the limit is reached
- Use auto-export to persist metrics before they're discarded

### Overhead

- Metrics collection adds minimal overhead (~1-2% in most cases)
- Can be completely disabled by setting `enable_collection=False`
- Individual metric types can be disabled selectively

### Thread Safety

- All metrics collection operations are thread-safe
- Uses internal locking to ensure data consistency
- Safe to use with concurrent dispatch/combine operations

## Best Practices

### Configuration

1. **Enable selective collection**: Disable unused metric types for better performance
2. **Set appropriate limits**: Configure `max_stored_forwards` based on available memory
3. **Use auto-export**: For long-running experiments, enable auto-export to prevent data loss

### Analysis

1. **Use aggregated statistics**: For quick performance overview
2. **Export raw data**: For detailed analysis and visualization
3. **Monitor trends**: Track metrics over time to identify performance regressions

### Debugging

1. **Check summary reports**: Quick way to identify performance issues
2. **Analyze individual forwards**: For debugging specific performance problems
3. **Compare configurations**: Use metrics to evaluate different system configurations

## Example Analysis Workflows

### Performance Optimization

```python
# Collect baseline metrics
baseline_collector = MetricsCollector(config)
moe_block = DistMoEBlock(config, tte_config, baseline_collector)

# Run workload
for batch in dataset:
    output = moe_block(batch)

# Get baseline statistics
baseline_stats = baseline_collector.get_aggregated_statistics()

# Test optimization
optimized_collector = MetricsCollector(config)
optimized_moe_block = DistMoEBlock(optimized_config, tte_config, optimized_collector)

# Compare results
optimized_stats = optimized_collector.get_aggregated_statistics()
improvement = (baseline_stats['e2e_latency']['mean'] - optimized_stats['e2e_latency']['mean']) / baseline_stats['e2e_latency']['mean'] * 100
print(f"Performance improvement: {improvement:.1f}%")
```

### Bottleneck Analysis

```python
# Analyze where time is spent
stats = metrics_collector.get_aggregated_statistics()

setup_ratio = stats['setup_latency']['mean'] / stats['e2e_latency']['mean']
dispatch_ratio = stats['dispatch_latency']['mean'] / stats['e2e_latency']['mean'] 
combine_ratio = stats['combine_latency']['mean'] / stats['e2e_latency']['mean']

print(f"Time breakdown:")
print(f"  Setup: {setup_ratio*100:.1f}%")
print(f"  Dispatch: {dispatch_ratio*100:.1f}%") 
print(f"  Combine: {combine_ratio*100:.1f}%")
```

### Transfer Performance Analysis

```python
# Analyze transfer performance
stats = metrics_collector.get_aggregated_statistics()

read_throughput = stats['read_throughput']['mean']
write_throughput = stats['write_throughput']['mean']
theoretical_max = 100  # MB/ms (example)

read_efficiency = read_throughput / theoretical_max * 100
write_efficiency = write_throughput / theoretical_max * 100

print(f"Transfer efficiency:")
print(f"  Read: {read_efficiency:.1f}% ({read_throughput:.1f} MB/ms)")
print(f"  Write: {write_efficiency:.1f}% ({write_throughput:.1f} MB/ms)")
```

## Troubleshooting

### Common Issues

1. **No metrics collected**: Check that `enable_collection=True` and metrics collector is passed to DistMoEBlock
2. **Missing transfer metrics**: Ensure `enable_transfer_metrics=True` and transfer operations are properly instrumented
3. **Memory usage**: Reduce `max_stored_forwards` or enable auto-export
4. **Export failures**: Check that export directory exists and is writable

### Debugging

```python
# Check collector state
print(f"Collection enabled: {metrics_collector.config.enable_collection}")
print(f"Forwards collected: {len(metrics_collector._completed_forwards)}")
print(f"Current forward active: {metrics_collector._current_forward is not None}")

# Clear metrics if needed
metrics_collector.clear_metrics()
```

## Future Enhancements

Potential future improvements to the metrics system:

1. **Real-time monitoring**: Live dashboard for metrics visualization
2. **Alerting**: Automatic alerts for performance regressions
3. **Distributed collection**: Aggregate metrics across multiple nodes
4. **Advanced analytics**: Statistical analysis and anomaly detection
5. **Integration**: Export to monitoring systems (Prometheus, Grafana)

## Related Documentation

- [MoE P2P Introduction](moe_p2p_intro.md)
- [Code Review Guidelines](code_review.md)
- [Known Issues](known_issues.md)
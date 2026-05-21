import pytest
import tempfile
import os
import json
import csv
from unittest.mock import Mock, patch
import torch

from MoE.MetricsCollector import MetricsCollector, MetricsCollectorConfig, initialize_metrics_collector, get_metrics_collector
from MoE.DistMoEBlock import DistMoEBlock, DistMoEBlockConfig
from MoE.Metrics import DispatchMetrics, CombineMetrics, ExpertsBlockMetrics, ForwardMetrics, Metrics, ExpertsRunMetrics
from TensorTransferEngine.TensorTransferEngine import TensorTransferEngineBackend, TensorTransferEngineConfig
from TensorTransferEngine.utils import HandshakeType, TransferProtocol


class TestMetricsCollector:
    """Test suite for the MetricsCollector class"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def metrics_config(self, temp_dir):
        """Create a test metrics collector configuration"""
        return MetricsCollectorConfig(
            enable_collection=True,
            enable_detailed_timing=True,
            enable_transfer_metrics=True,
            enable_expert_block_metrics=True,
            max_stored_forwards=100,
            auto_export_interval=None,
            export_directory=temp_dir,
        )

    @pytest.fixture
    def metrics_collector(self, metrics_config):
        """Create a MetricsCollector instance for testing"""
        return MetricsCollector(metrics_config)

    def test_metrics_collector_initialization(self, metrics_collector, temp_dir):
        """Test that MetricsCollector initializes correctly"""
        assert metrics_collector.config.enable_collection is True
        assert metrics_collector.config.export_directory == temp_dir
        assert os.path.exists(temp_dir)
        assert len(metrics_collector._completed_forwards) == 0
        assert metrics_collector._forward_counter == 0

    def test_forward_pass_metrics_collection(self, metrics_collector):
        """Test basic forward pass metrics collection"""
        # Start a forward pass
        metrics_collector.start_forward_pass()
        assert metrics_collector._current_forward is not None
        assert metrics_collector._current_forward_start_time is not None

        # End setup phase
        metrics_collector.end_setup_phase()
        assert metrics_collector._current_forward.setup_latency > 0

        # End forward pass
        metrics_collector.end_forward_pass()
        assert len(metrics_collector._completed_forwards) == 1
        assert metrics_collector._forward_counter == 1
        assert metrics_collector._current_forward is None

    def test_dispatch_combine_metrics(self, metrics_collector):
        """Test dispatch and combine metrics collection"""
        metrics_collector.start_forward_pass()

        # Test dispatch metrics
        task_id = "test_task_123"
        metrics_collector.start_dispatch(task_id)
        metrics_collector.end_dispatch(task_id)

        # Test combine metrics
        metrics_collector.start_combine(task_id)
        metrics_collector.end_combine(task_id)

        metrics_collector.end_forward_pass()

        # Verify metrics were collected
        forward = metrics_collector._completed_forwards[0]
        assert len(forward.dispatches) == 1
        assert len(forward.combines) == 1
        assert forward.dispatches[0].e2e_latency > 0
        assert forward.combines[0].e2e_latency > 0

    def test_expert_run_metrics(self, metrics_collector):
        """Test expert block run metrics collection"""
        metrics_collector.start_forward_pass()

        task_id = "expert_task_456"
        experts_block_index = 0

        # Start expert run
        metrics_collector.start_expert_run(task_id, experts_block_index)

        # Simulate some work
        import time

        time.sleep(0.001)  # 1ms

        # End expert run
        metrics_collector.end_expert_run(task_id)

        metrics_collector.end_forward_pass()

        # Verify expert block metrics were collected
        forward = metrics_collector._completed_forwards[0]
        assert len(forward.experts_blocks) == 1
        expert_block = forward.experts_blocks[0]
        assert expert_block.number_of_runs == 1
        assert expert_block.avg_e2e_latency > 0
        assert len(expert_block.experts_runs) == 1

    def test_aggregated_statistics(self, metrics_collector):
        """Test aggregated statistics calculation"""
        # Collect multiple forward passes
        for i in range(3):
            metrics_collector.start_forward_pass()

            # Add some dispatch/combine operations
            task_id = f"task_{i}"
            metrics_collector.start_dispatch(task_id)
            metrics_collector.end_dispatch(task_id)
            metrics_collector.start_combine(task_id)
            metrics_collector.end_combine(task_id)

            # Add expert run
            metrics_collector.start_expert_run(f"expert_{i}", 0)
            metrics_collector.end_expert_run(f"expert_{i}")

            metrics_collector.end_forward_pass()

        # Get aggregated statistics
        stats = metrics_collector.get_aggregated_statistics()

        assert stats["total_forwards"] == 3
        assert "e2e_latency" in stats
        assert "dispatch_latency" in stats
        assert "combine_latency" in stats
        assert "expert_block_latency" in stats
        assert stats["avg_dispatches_per_forward"] == 1.0
        assert stats["avg_combines_per_forward"] == 1.0
        assert stats["avg_expert_blocks_per_forward"] == 1.0

    def test_metrics_export(self, metrics_collector, temp_dir):
        """Test metrics export functionality"""
        # Collect some metrics
        metrics_collector.start_forward_pass()
        metrics_collector.start_dispatch("test_task")
        metrics_collector.end_dispatch("test_task")
        metrics_collector.end_forward_pass()

        # Export metrics
        exported_files = metrics_collector.export_metrics("test_export")

        # Verify files were created
        assert "json" in exported_files
        assert "stats" in exported_files
        assert "csv" in exported_files

        # Verify JSON file content
        json_file = exported_files["json"]
        assert os.path.exists(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)
        assert data["number_of_forwards"] == 1
        assert len(data["forwards"]) == 1

        # Verify CSV file content
        csv_file = exported_files["csv"]
        assert os.path.exists(csv_file)
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "e2e_latency" in rows[0]

    def test_summary_report(self, metrics_collector):
        """Test summary report generation"""
        # Collect some metrics
        metrics_collector.start_forward_pass()
        metrics_collector.end_forward_pass()

        # Generate summary report
        report = metrics_collector.get_summary_report()

        assert "DistMoE Metrics Summary" in report
        assert "Total Forward Passes: 1" in report
        assert "E2E Latency (ms):" in report

    def test_metrics_clearing(self, metrics_collector):
        """Test metrics clearing functionality"""
        # Collect some metrics
        metrics_collector.start_forward_pass()
        metrics_collector.end_forward_pass()

        assert len(metrics_collector._completed_forwards) == 1
        assert metrics_collector._forward_counter == 1

        # Clear metrics
        metrics_collector.clear_metrics()

        assert len(metrics_collector._completed_forwards) == 0
        assert metrics_collector._forward_counter == 0
        assert metrics_collector._current_forward is None

    def test_disabled_collection(self):
        """Test that metrics collection can be disabled"""
        config = MetricsCollectorConfig(enable_collection=False)
        collector = MetricsCollector(config)

        # Try to collect metrics
        collector.start_forward_pass()
        collector.start_dispatch("test")
        collector.end_dispatch("test")
        collector.end_forward_pass()

        # Verify no metrics were collected
        assert len(collector._completed_forwards) == 0
        assert collector._forward_counter == 0

    def test_max_stored_forwards_limit(self, metrics_config):
        """Test that the max stored forwards limit is respected"""
        metrics_config.max_stored_forwards = 2
        collector = MetricsCollector(metrics_config)

        # Collect more forwards than the limit
        for i in range(5):
            collector.start_forward_pass()
            collector.end_forward_pass()

        # Verify only the limit number of forwards are stored
        assert len(collector._completed_forwards) == 2
        assert collector._forward_counter == 5  # Counter should still be accurate

    def test_concurrent_operations(self, metrics_collector):
        """Test handling of concurrent dispatch/combine operations"""
        metrics_collector.start_forward_pass()

        # Start multiple operations
        task_ids = ["task1", "task2", "task3"]
        for task_id in task_ids:
            metrics_collector.start_dispatch(task_id)
            metrics_collector.start_combine(task_id)

        # End operations in different order
        for task_id in reversed(task_ids):
            metrics_collector.end_dispatch(task_id)
            metrics_collector.end_combine(task_id)

        metrics_collector.end_forward_pass()

        # Verify all operations were tracked
        forward = metrics_collector._completed_forwards[0]
        assert len(forward.dispatches) == 3
        assert len(forward.combines) == 3


class TestMetricsIntegration:
    """Integration tests for metrics with DistMoEBlock"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_tte_config(self):
        """Create a mock TensorTransferEngine config"""
        return TensorTransferEngineConfig(
            host_address="localhost",
            host_port=10000,
            host_p2p_ports=[11000],
            remote_addresses=["localhost"],
            remote_ports=[20000],
            remote_p2p_ports=[21000],
            handshake_type=HandshakeType.SERVER,
            transfer_protocol=TransferProtocol.RDMA,
            ib_device_names="",
            p2p_timeout_duration=3000,
            metadata_schema="http",
            metadata_host="localhost",
            metadata_port=8080,
            metadata_dir="metadata",
        )

    @pytest.fixture
    def mock_moe_config(self):
        """Create a mock DistMoEBlock config"""
        return DistMoEBlockConfig(
            backend=TensorTransferEngineBackend.NIXL,
            number_of_experts=4,
            expert_blocks=[[0, 1], [2, 3]],
            number_of_io_buffers=2,
            io_buffer_size=1000,
            top_k=2,
            hidden_features=512,
            dtype=torch.float32,
            moe_device=torch.device("cpu"),
            eb_devices=[torch.device("cpu"), torch.device("cpu")],
            forward_timeout=5000,
            host_health_check_ports=[12000, 13000],
            remote_health_check_ports=[22000, 23000],
        )

    def test_global_metrics_collector(self, temp_dir):
        """Test global metrics collector initialization and access"""
        config = MetricsCollectorConfig(export_directory=temp_dir)

        # Initialize global collector
        collector = initialize_metrics_collector(config)
        assert collector is not None

        # Access global collector
        global_collector = get_metrics_collector()
        assert global_collector is not None
        assert global_collector is collector

        # Test metrics collection through global collector
        global_collector.start_forward_pass()
        global_collector.end_forward_pass()

        assert len(global_collector._completed_forwards) == 1


class TestMetricsDataStructures:
    """Test the metrics data structures"""

    def test_dispatch_metrics(self):
        """Test DispatchMetrics data structure"""
        metrics = DispatchMetrics(e2e_latency=10.5)
        assert metrics.e2e_latency == 10.5

    def test_combine_metrics(self):
        """Test CombineMetrics data structure"""
        metrics = CombineMetrics(e2e_latency=8.2)
        assert metrics.e2e_latency == 8.2

    def test_experts_run_metrics(self):
        """Test ExpertsRunMetrics data structure"""
        metrics = ExpertsRunMetrics(experts_block_index=0, e2e_latency=15.0, read_transfer_throughput=100.5, write_transfer_throughput=95.3)
        assert metrics.experts_block_index == 0
        assert metrics.e2e_latency == 15.0
        assert metrics.read_transfer_throughput == 100.5
        assert metrics.write_transfer_throughput == 95.3

    def test_experts_block_metrics(self):
        """Test ExpertsBlockMetrics data structure"""
        expert_run = ExpertsRunMetrics(
            experts_block_index=0, e2e_latency=15.0, read_transfer_throughput=100.5, write_transfer_throughput=95.3
        )

        metrics = ExpertsBlockMetrics(
            number_of_runs=1, number_of_transfers=2, avg_e2e_latency=15.0, avg_transfers_throughput=97.9, experts_runs=[expert_run]
        )

        assert metrics.number_of_runs == 1
        assert metrics.number_of_transfers == 2
        assert metrics.avg_e2e_latency == 15.0
        assert metrics.avg_transfers_throughput == 97.9
        assert len(metrics.experts_runs) == 1
        assert metrics.experts_runs[0] == expert_run

    def test_forward_metrics(self):
        """Test ForwardMetrics data structure"""
        dispatch = DispatchMetrics(e2e_latency=10.0)
        combine = CombineMetrics(e2e_latency=8.0)
        expert_run = ExpertsRunMetrics(
            experts_block_index=0, e2e_latency=15.0, read_transfer_throughput=100.0, write_transfer_throughput=95.0
        )
        expert_block = ExpertsBlockMetrics(
            number_of_runs=1, number_of_transfers=2, avg_e2e_latency=15.0, avg_transfers_throughput=97.5, experts_runs=[expert_run]
        )

        metrics = ForwardMetrics(
            e2e_latency=50.0,
            setup_latency=5.0,
            avg_dispatches_e2e_latency=10.0,
            avg_combines_e2e_latency=8.0,
            dispatches=[dispatch],
            combines=[combine],
            experts_blocks=[expert_block],
        )

        assert metrics.e2e_latency == 50.0
        assert metrics.setup_latency == 5.0
        assert metrics.avg_dispatches_e2e_latency == 10.0
        assert metrics.avg_combines_e2e_latency == 8.0
        assert len(metrics.dispatches) == 1
        assert len(metrics.combines) == 1
        assert len(metrics.experts_blocks) == 1

    def test_metrics_container(self):
        """Test main Metrics container"""
        forward = ForwardMetrics(
            e2e_latency=50.0,
            setup_latency=5.0,
            avg_dispatches_e2e_latency=10.0,
            avg_combines_e2e_latency=8.0,
            dispatches=[],
            combines=[],
            experts_blocks=[],
        )

        metrics = Metrics(number_of_forwards=1, forwards=[forward])

        assert metrics.number_of_forwards == 1
        assert len(metrics.forwards) == 1
        assert metrics.forwards[0] == forward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for Stage 6a: CircuitGraph schema and CircuitGraphBuilder.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from dft_hotspot.dft.scan_metadata import ScanCell, ScanChain, ScanEndpoint, ScanMetadata
from dft_hotspot.features.def_parser import DefComponent, DefData, DefDieArea, DefNet
from dft_hotspot.graphs.builder import CircuitGraphBuilder
from dft_hotspot.graphs.schema import (
    CircuitGraph,
    EDGE_FEATURE_NAMES,
    EDGE_TYPE_NETLIST,
    EDGE_TYPE_SCAN_FORWARD,
    EDGE_TYPE_SCAN_REVERSE,
    NODE_FEATURE_NAMES,
)


class TestCircuitGraphSchema(unittest.TestCase):
    """Tests for CircuitGraph dataclass, serialization, and PyG conversion."""

    def setUp(self):
        self.num_nodes = 5
        self.node_dim = len(NODE_FEATURE_NAMES)
        self.edge_dim = len(EDGE_FEATURE_NAMES)
        self.node_features = np.random.rand(self.num_nodes, self.node_dim).astype(np.float32)
        self.edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
        self.edge_features = np.random.rand(4, self.edge_dim).astype(np.float32)
        self.node_names = [f"cell_{i}" for i in range(self.num_nodes)]
        self.node_positions = np.random.rand(self.num_nodes, 2).astype(np.float32)
        self.node_positions_dbu = self.node_positions * 1000.0
        self.node_grid_coords = np.random.randint(0, 64, size=(self.num_nodes, 2), dtype=np.int64)
        self.metadata = {"design": "test_synth", "sample_id": "test_s001"}

        self.graph = CircuitGraph(
            node_features=self.node_features,
            edge_index=self.edge_index,
            edge_features=self.edge_features,
            node_names=self.node_names,
            node_positions=self.node_positions,
            node_positions_dbu=self.node_positions_dbu,
            node_grid_coords=self.node_grid_coords,
            metadata=self.metadata,
        )

    def test_properties(self):
        self.assertEqual(self.graph.num_nodes, 5)
        self.assertEqual(self.graph.num_edges, 4)
        self.assertEqual(self.graph.node_dim, 10)
        self.assertEqual(self.graph.edge_dim, 4)

    def test_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_graph.npz"
            self.graph.save_npz(save_path)
            self.assertTrue(save_path.is_file())

            loaded = CircuitGraph.load_npz(save_path)
            self.assertEqual(loaded.num_nodes, self.graph.num_nodes)
            self.assertEqual(loaded.num_edges, self.graph.num_edges)
            np.testing.assert_allclose(loaded.node_features, self.graph.node_features, rtol=1e-5)
            np.testing.assert_array_equal(loaded.edge_index, self.graph.edge_index)
            np.testing.assert_allclose(loaded.edge_features, self.graph.edge_features, rtol=1e-5)
            self.assertEqual(loaded.node_names, self.graph.node_names)
            self.assertEqual(loaded.metadata["design"], "test_synth")

    def test_pyg_conversion_and_save_pt(self):
        try:
            import torch
            import torch_geometric
        except ImportError:
            self.skipTest("PyTorch / PyG not installed in current environment")

        pyg_data = self.graph.to_pyg(y=np.zeros((64, 64), dtype=np.float32))
        self.assertEqual(pyg_data.num_nodes, 5)
        self.assertEqual(pyg_data.edge_index.shape, (2, 4))
        self.assertEqual(pyg_data.x.shape, (5, 10))
        self.assertEqual(pyg_data.edge_attr.shape, (4, 4))
        self.assertEqual(pyg_data.pos.shape, (5, 2))
        self.assertEqual(pyg_data.grid_coords.shape, (5, 2))
        self.assertEqual(pyg_data.y.shape, (64, 64))

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = Path(tmpdir) / "test_graph.pt"
            self.graph.save_pt(pt_path, y=np.zeros((64, 64), dtype=np.float32))
            self.assertTrue(pt_path.is_file())

            loaded_pt = CircuitGraph.load_pt(pt_path)
            self.assertEqual(loaded_pt.num_nodes, 5)


class TestCircuitGraphBuilder(unittest.TestCase):
    """Tests for CircuitGraphBuilder logic, edge creation, and scan integration."""

    def setUp(self):
        self.builder = CircuitGraphBuilder(grid_size=64, max_net_degree=10, include_scan_edges=True)

    def test_synthetic_graph_building(self):
        # 4 components: 2 DFFs and 2 NANDs
        components = [
            DefComponent(name="_s0_", master="DFFR_X1", x=2000.0, y=2000.0),
            DefComponent(name="_s1_", master="DFFR_X1", x=6000.0, y=6000.0),
            DefComponent(name="_c0_", master="NAND2_X1", x=3000.0, y=3000.0),
            DefComponent(name="_c1_", master="NAND2_X1", x=4000.0, y=4000.0),
        ]
        die = DefDieArea(x_min=0.0, y_min=0.0, x_max=10000.0, y_max=10000.0, dbu_per_micron=1000)
        # Net 1 connects _s0_ and _c0_
        # Net 2 connects _c0_ and _c1_
        nets = [
            DefNet(name="net1", connections=[("_s0_", "Q"), ("_c0_", "A1")]),
            DefNet(name="net2", connections=[("_c0_", "ZN"), ("_c1_", "A1")]),
        ]
        def_data = DefData(die_area=die, components=components, nets=nets, dbu_per_micron=1000)

        # 1 Scan Chain connecting _s0_ -> _s1_
        chain = ScanChain(
            chain_id="chain_0",
            start=ScanEndpoint(type="PIN", name="scan_in"),
            stop=ScanEndpoint(type="PIN", name="scan_out"),
            cell_count=2,
            cells=[
                ScanCell(index=0, instance="_s0_", master="DFFR_X1", x=2000.0, y=2000.0, placed=True),
                ScanCell(index=1, instance="_s1_", master="DFFR_X1", x=6000.0, y=6000.0, placed=True),
            ],
        )
        scan_meta = ScanMetadata(source="synthetic_proxy", chains=[chain])

        graph = self.builder.build(def_data=def_data, scan_metadata=scan_meta, design="toy")

        self.assertEqual(graph.num_nodes, 4)
        # Node features check
        # Node 0 (_s0_): is_seq=1, is_comb=0, is_scan=1, scan_chain_id=1.0, scan_order_index=0.0
        self.assertEqual(graph.node_features[0, 2], 1.0)  # is_seq
        self.assertEqual(graph.node_features[0, 3], 0.0)  # is_comb
        self.assertEqual(graph.node_features[0, 4], 1.0)  # is_scan
        self.assertEqual(graph.node_features[0, 5], 1.0)  # scan_chain_id
        self.assertEqual(graph.node_features[0, 6], 0.0)  # scan_order_index

        # Node 1 (_s1_): is_seq=1, is_scan=1, scan_order_index=1.0
        self.assertEqual(graph.node_features[1, 2], 1.0)
        self.assertEqual(graph.node_features[1, 4], 1.0)
        self.assertEqual(graph.node_features[1, 6], 1.0)

        # Node 2 (_c0_): is_seq=0, is_comb=1, is_scan=0
        self.assertEqual(graph.node_features[2, 2], 0.0)
        self.assertEqual(graph.node_features[2, 3], 1.0)
        self.assertEqual(graph.node_features[2, 4], 0.0)

        # Edges check:
        # Net 1: 2 bidirectional edges (_s0_ <-> _c0_)
        # Net 2: 2 bidirectional edges (_c0_ <-> _c1_)
        # Scan chain: 2 bidirectional edges (_s0_ <-> _s1_)
        # Total edges = 2 + 2 + 2 = 6
        self.assertEqual(graph.num_edges, 6)
        self.assertEqual(graph.metadata["num_netlist_edges"], 4)
        self.assertEqual(graph.metadata["num_scan_edges"], 2)

    def test_max_net_degree_filtering(self):
        # 5 components connected to a single high-fanout clock net
        components = [DefComponent(name=f"c_{i}", master="DFFR_X1", x=i * 100.0, y=i * 100.0) for i in range(5)]
        die = DefDieArea(x_min=0, y_min=0, x_max=1000, y_max=1000)
        high_fanout_net = DefNet(name="clk", connections=[(f"c_{i}", "CLK") for i in range(5)])
        def_data = DefData(die_area=die, components=components, nets=[high_fanout_net])

        # Builder with max_net_degree = 3 (less than 5)
        builder = CircuitGraphBuilder(max_net_degree=3)
        graph = builder.build(def_data=def_data)
        # Net is skipped from clique expansion
        self.assertEqual(graph.num_edges, 0)

        # Builder with max_net_degree = 10 (greater than 5)
        builder_permissive = CircuitGraphBuilder(max_net_degree=10)
        graph_perm = builder_permissive.build(def_data=def_data)
        # 5 nodes -> 5 * 4 = 20 directed edges
        self.assertEqual(graph_perm.num_edges, 20)

    def test_empty_def_handling(self):
        def_data = DefData()
        graph = self.builder.build(def_data=def_data)
        self.assertEqual(graph.num_nodes, 0)
        self.assertEqual(graph.num_edges, 0)

    def test_real_sample_graph_extraction(self):
        """Validates graph extraction on an actual processed sample from expanded_v002."""
        sample_dir = Path("data/processed/datasets/expanded_v002/samples/dft_counter_nangate45_u45_clk3p0_ar1p0_chains1")
        if not sample_dir.is_dir():
            self.skipTest("Sample directory not found; skipping integration check")

        meta_file = sample_dir / "metadata.json"
        scan_file = sample_dir / "scan_metadata.json"
        if not meta_file.is_file() or not scan_file.is_file():
            self.skipTest("Sample metadata missing")

        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        def_path = Path(meta["def_path"])
        if not def_path.is_file():
            # Check relative Windows path fallback
            def_path = Path(str(meta["def_path"]).replace("/mnt/e/", "E:/"))
            if not def_path.is_file():
                self.skipTest(f"DEF file not found: {def_path}")

        graph = self.builder.build_from_files(
            def_path=def_path,
            scan_metadata_path=scan_file,
            design="dft_counter",
            sample_id="dft_counter_nangate45_u45_clk3p0_ar1p0_chains1",
            pdk="nangate45",
        )

        self.assertEqual(graph.num_nodes, 84)  # 84 components in dft_counter
        self.assertEqual(graph.metadata["scan_cells_count"], 16)
        self.assertGreater(graph.num_edges, 100)
        self.assertEqual(graph.metadata["num_scan_edges"], 30)  # 16 cells -> 15 links * 2 = 30 edges

        # Verify PyG conversion
        try:
            pyg_data = graph.to_pyg()
            self.assertEqual(pyg_data.num_nodes, 84)
            self.assertEqual(pyg_data.x.shape, (84, 10))
            self.assertEqual(pyg_data.edge_index.shape[0], 2)
            self.assertEqual(pyg_data.edge_attr.shape[1], 4)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()

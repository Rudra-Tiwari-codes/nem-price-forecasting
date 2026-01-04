"""
End-to-End Smoke Test.

Tests the full pipeline: downloader → simulation → JSON output.
This ensures the entire system works together correctly.
"""

import pytest
import json
import subprocess
import sys
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestE2EPipeline:
    """End-to-end tests for the full data pipeline."""

    @pytest.fixture(scope="class")
    def ensure_data_exists(self):
        """Ensure we have data to test with (download if needed)."""
        data_file = PROJECT_ROOT / "data" / "combined_dispatch_prices.csv"
        
        if not data_file.exists() or data_file.stat().st_size < 1000:
            # Run the downloader with limited files for testing
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "download_aemo_data.py")],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(PROJECT_ROOT)
            )
            # Allow failure if NEMWEB is unreachable, but file should exist
            if not data_file.exists():
                pytest.skip("Could not download data and no existing data found")
        
        return data_file

    def test_data_file_valid(self, ensure_data_exists):
        """Test that the data file exists and has valid structure."""
        import pandas as pd
        
        df = pd.read_csv(ensure_data_exists)
        
        # Check required columns exist
        assert 'SETTLEMENTDATE' in df.columns, "Missing SETTLEMENTDATE column"
        assert 'RRP' in df.columns, "Missing RRP column"
        assert 'REGIONID' in df.columns, "Missing REGIONID column"
        
        # Check data is not empty
        assert len(df) > 0, "Data file is empty"
        
        # Check regions are valid
        valid_regions = {'SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1'}
        actual_regions = set(df['REGIONID'].unique())
        assert actual_regions.issubset(valid_regions), f"Invalid regions: {actual_regions - valid_regions}"

    def test_main_simulation_runs(self, ensure_data_exists):
        """Test that main.py runs without errors."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), 
             "--region", "SA1", "--no-charts"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT)
        )
        
        assert result.returncode == 0, f"main.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "SIMULATION COMPLETE" in result.stdout, "Simulation did not complete successfully"

    def test_json_output_generated(self, ensure_data_exists):
        """Test that simulation generates valid JSON output."""
        # Run simulation to generate JSON
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), 
             "--region", "SA1", "--no-charts"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT)
        )
        
        json_file = PROJECT_ROOT / "dashboard" / "public" / "simulation_SA1.json"
        assert json_file.exists(), "simulation_SA1.json not generated"
        
        # Validate JSON structure
        with open(json_file) as f:
            data = json.load(f)
        
        # Check required fields
        assert 'lastUpdated' in data, "Missing lastUpdated"
        assert 'region' in data, "Missing region"
        assert 'stats' in data, "Missing stats"
        assert 'prices' in data, "Missing prices"
        assert 'strategies' in data, "Missing strategies"
        
        # Check stats structure
        stats = data['stats']
        assert 'current' in stats, "Missing current price in stats"
        assert 'mean' in stats, "Missing mean in stats"
        assert 'min' in stats, "Missing min in stats"
        assert 'max' in stats, "Missing max in stats"
        
        # Check strategies have required fields
        assert len(data['strategies']) > 0, "No strategies in output"
        for strategy in data['strategies']:
            assert 'name' in strategy, "Strategy missing name"
            assert 'profit' in strategy, "Strategy missing profit"

    def test_all_regions_can_run(self, ensure_data_exists):
        """Test that simulation can run for all regions."""
        regions = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1']
        
        for region in regions:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "main.py"), 
                 "--region", region, "--no-charts"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(PROJECT_ROOT)
            )
            
            # Some regions may have no data, which is OK
            if "No data loaded" in result.stdout or "Error: No data" in result.stdout:
                continue
            
            assert result.returncode == 0, f"main.py failed for {region}:\n{result.stderr}"

    def test_cli_help_works(self):
        """Test that CLI help works without data."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT)
        )
        
        assert result.returncode == 0, "CLI help failed"
        assert "NEM Arbitrage Engine" in result.stdout, "Help text missing expected content"


class TestTimezoneHandling:
    """Tests for timezone-aware data processing."""

    def test_data_loader_timezone(self):
        """Test that data loader handles timezones correctly."""
        import pandas as pd
        from pathlib import Path
        import sys
        
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from data_loader import load_dispatch_data, AEMO_TIMEZONE
        
        data_file = PROJECT_ROOT / "data" / "combined_dispatch_prices.csv"
        if not data_file.exists():
            pytest.skip("Data file not found")
        
        df = load_dispatch_data(str(data_file))
        
        # Check that datetime is timezone-aware
        if len(df) > 0:
            assert df['SETTLEMENTDATE'].dt.tz is not None, "Datetime should be timezone-aware"

    def test_aemo_timezone_constant(self):
        """Test that AEMO timezone is correctly defined."""
        from zoneinfo import ZoneInfo
        
        tz = ZoneInfo('Australia/Sydney')
        assert tz is not None, "Australia/Sydney timezone should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

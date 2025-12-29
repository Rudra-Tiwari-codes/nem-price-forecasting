"""
Update README with live simulation results.

This script runs after main.py and updates the README.md with
actual profit values from the most recent simulation.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone


def update_readme():
    """Update README.md with latest simulation results."""
    project_root = Path(__file__).parent
    readme_path = project_root / "README.md"
    
    # Load simulation results from dashboard JSON
    results = {}
    regions = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1']
    
    for region in regions:
        json_path = project_root / "dashboard" / "public" / f"simulation_{region}.json"
        if json_path.exists():
            with open(json_path) as f:
                results[region] = json.load(f)
    
    if not results:
        print("No simulation results found, skipping README update")
        return
    
    # Use SA1 as the primary region for README
    sa1_data = results.get('SA1', {})
    if not sa1_data:
        print("No SA1 data found, skipping README update")
        return
    
    strategies = sa1_data.get('strategies', [])
    if not strategies:
        print("No strategy data found, skipping README update")
        return
    
    # Build the results table
    table_rows = []
    for s in strategies:
        name = s.get('name', 'Unknown')
        profit = s.get('profit', 0)
        charges = s.get('charges', 'N/A')
        discharges = s.get('discharges', 'N/A')
        table_rows.append(f"| {name} | ${profit:,.0f} | {charges} | {discharges} |")
    
    new_table = """| Strategy | Profit | Charge Cycles | Discharge Cycles |
|----------|--------|---------------|------------------|
""" + "\n".join(table_rows)
    
    # Get best strategy info
    best_strategy = sa1_data.get('bestStrategy', 'Perfect Foresight')
    best_profit = sa1_data.get('bestProfit', 0)
    
    # Always use current UTC time for README timestamp
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    # Update the results section
    # Find and replace the results table
    table_pattern = r'\| Strategy \| Profit \| Charge Cycles \| Discharge Cycles \|.*?\n\n'
    
    new_content = re.sub(
        table_pattern,
        new_table + "\n\n",
        readme_content,
        flags=re.DOTALL
    )
    
    # Update the key insight line with actual percentage
    if strategies:
        pf_profit = next((s['profit'] for s in strategies if 'Perfect' in s.get('name', '')), best_profit)
        greedy_profit = next((s['profit'] for s in strategies if 'Greedy' in s.get('name', '')), 0)
        if pf_profit > 0 and greedy_profit > 0:
            pct = (greedy_profit / pf_profit) * 100
            insight = f"**Key Insight:** Perfect Foresight provides the theoretical upper bound at ${pf_profit:,.0f}. The greedy strategy achieves ~{pct:.0f}% of optimal. Last updated: {date_str}."
            new_content = re.sub(
                r'\*\*Key Insight:\*\*[^\n]*',
                insight,
                new_content
            )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"README updated with latest results from {date_str}")
    print(f"Best strategy: {best_strategy} with ${best_profit:,.0f} profit")


if __name__ == "__main__":
    update_readme()

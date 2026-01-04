"""
Update README with live simulation results.

This script runs after main.py and updates the README.md with
actual profit values from the most recent simulation.
Preserves the polished README format while updating only the results.
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
    
    # Find Perfect Foresight profit for % of Optimal calculation
    pf_profit = next(
        (s['profit'] for s in strategies if 'Perfect' in s.get('name', '')), 
        None
    )
    
    if not pf_profit or pf_profit <= 0:
        print("No Perfect Foresight data found, skipping README update")
        return
    
    # Build the results table with % of Optimal column
    # Order: Perfect Foresight, Dynamic Programming, Forecast EMA, Sliding Window, Greedy
    strategy_order = ['Perfect Foresight', 'Dynamic Programming', 'Forecast', 'Sliding Window', 'Greedy']
    
    table_rows = []
    for order_name in strategy_order:
        for s in strategies:
            name = s.get('name', 'Unknown')
            if order_name in name:
                profit = s.get('profit', 0)
                charges = s.get('charges', 'N/A')
                discharges = s.get('discharges', 'N/A')
                pct_optimal = (profit / pf_profit) * 100
                
                # Format numbers with commas
                profit_str = f"${profit:,.0f}"
                charges_str = f"{charges:,}" if isinstance(charges, int) else str(charges)
                discharges_str = f"{discharges:,}" if isinstance(discharges, int) else str(discharges)
                
                table_rows.append(
                    f"| {name} | {profit_str} | {pct_optimal:.1f}% | {charges_str} | {discharges_str} |"
                )
                break
    
    new_table = """| Strategy | Profit | % of Optimal | Charge Cycles | Discharge Cycles |
|----------|--------|--------------|---------------|------------------|
""" + "\n".join(table_rows)
    
    # Get best strategy info for the insight
    best_strategy = sa1_data.get('bestStrategy', 'Perfect Foresight')
    best_profit = sa1_data.get('bestProfit', pf_profit)
    
    # Find greedy and EMA profits for insight text
    greedy_profit = next(
        (s['profit'] for s in strategies if 'Greedy' in s.get('name', '')), 
        0
    )
    ema_profit = next(
        (s['profit'] for s in strategies if 'Forecast' in s.get('name', '') or 'EMA' in s.get('name', '')), 
        0
    )
    
    greedy_pct = (greedy_profit / pf_profit) * 100 if pf_profit > 0 else 0
    ema_pct = (ema_profit / pf_profit) * 100 if pf_profit > 0 else 0
    
    # Build new insight text
    insight = (
        f"**Key Insight:** The Perfect Foresight algorithm establishes a theoretical upper bound of "
        f"${pf_profit:,.0f}. Real-world strategies without future knowledge achieve "
        f"{int(greedy_pct)}-{int(ema_pct)}% of optimal, with the EMA-based forecast strategy "
        f"performing best among implementable approaches."
    )
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    # Update the results table (match the new format with 5 columns)
    table_pattern = r'\| Strategy \| Profit \| % of Optimal \| Charge Cycles \| Discharge Cycles \|.*?\n\n'
    
    new_content = re.sub(
        table_pattern,
        new_table + "\n\n",
        readme_content,
        flags=re.DOTALL
    )
    
    # Update the key insight line
    new_content = re.sub(
        r'\*\*Key Insight:\*\*[^\n]*',
        insight,
        new_content
    )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"README updated with latest results")
    print(f"  Perfect Foresight: ${pf_profit:,.0f}")
    print(f"  Greedy: ${greedy_profit:,.0f} ({greedy_pct:.1f}% of optimal)")
    print(f"  EMA Forecast: ${ema_profit:,.0f} ({ema_pct:.1f}% of optimal)")


if __name__ == "__main__":
    update_readme()

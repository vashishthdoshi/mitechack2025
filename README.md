# Natural Gas Power Plant Portfolio Analysis

This repository contains a comprehensive analysis of natural gas power plant operations across the United States, focusing on profitability, efficiency, and regulatory compliance. The project includes both exploratory data analysis (notebook) and an interactive executive dashboard (Streamlit app).

## Project Overview

The analysis examines the economic viability of natural gas power plants by integrating multiple data sources including electricity prices, fuel costs, generation data, and state-level emission regulations. The project identifies underperforming facilities and provides scenario modeling capabilities for portfolio optimization.

## Files

### `final_notebook.ipynb`
Jupyter notebook containing the complete data analysis pipeline:
- Data loading and cleaning from multiple CSV sources
- Natural gas plant filtering and aggregation
- Economic metrics calculation (operational costs, revenues, profitability)
- Efficiency analysis (build vs. buy comparisons)
- State-level aggregation and ranking
- Identification of "zombie plants" (facilities with zero generation but positive costs)

**Key Analysis Steps:**
1. Load and process electricity pricing data (2023, industrial sector)
2. Extract natural gas fuel cost data by state
3. Filter EIA-923 facility data for natural gas plants
4. Calculate per-plant operational costs and revenue
5. Compute efficiency gaps and profitability metrics
6. Aggregate results at state and portfolio levels

### `executive_briefing.py`
Streamlit-based interactive dashboard for executive-level portfolio analysis:

**Features:**
- Real-time "what-if" scenario modeling with adjustable parameters:
  - Gas price adjustments (-50% to +100%)
  - Grid price adjustments (-50% to +50%)
  - Generation efficiency improvements (0% to +50%)
- Portfolio-wide KPI tracking
- State-level profitability rankings
- Top 10 worst-performing plants analysis
- Deep-dive plant diagnostics with regulatory context
- Build vs. buy efficiency comparisons
- Local efficiency improvement scenarios

**Key Components:**
- Sidebar controls for global assumptions
- Headline KPIs (total profit, generation costs, revenue)
- State-level analysis with sortable metrics
- Plant-specific regulatory compliance information
- Interactive efficiency modeling tools

## Data Requirements

The following CSV files are required in the working directory:

1. **`average_retail_price_of_electricity_annual.csv`**
   - Source: EIA electricity pricing data
   - Contains industrial electricity prices by state
   - Header row on line 5

2. **`Average_cost_of_fossil_fuels_for_electricity_generation_natural_gas_all_sectors_annual.csv`**
   - Source: EIA fuel cost data
   - Natural gas prices by state ($/MMBtu)

3. **`EIA923_Schedules_2_3_4_5_M_12_2023_Final_Revision.csv`**
   - Source: EIA Form 923
   - Plant-level generation and fuel consumption data
   - Contains facility identifiers, generation output, and fuel usage

4. **`Carbon_Pricing_Table.csv`**
   - State-level emission regulations and penalties
   - Emission thresholds and compliance costs
   - Coverage and key standards by jurisdiction

## Installation

Install required dependencies:

```bash
pip install streamlit pandas numpy
```

For the Jupyter notebook, also install:
```bash
pip install jupyter matplotlib seaborn
```

## Usage

### Running the Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run executive_briefing.py
```

The dashboard will open in your browser with:
- Sidebar controls for scenario modeling
- Portfolio overview and KPIs
- Interactive state and plant-level analysis

### Running the Notebook

Open and execute the analysis notebook:
```bash
jupyter notebook final_notebook.ipynb
```

Run cells sequentially to reproduce the data pipeline and analysis.

## Key Metrics

**Plant-Level Metrics:**
- Generation Cost (Million USD): Total fuel consumption cost
- Net Revenue (Million USD): Electricity sales revenue
- Net Operating Profit: Revenue minus generation costs
- Build Cost per MWh: Cost to generate electricity
- Grid Price per MWh: Market price for electricity
- Efficiency Gap: Build cost minus grid price (negative = unprofitable)

**Portfolio Metrics:**
- Total Net Operating Profit across all facilities
- Aggregate generation costs and revenue
- State-level profitability rankings
- Worst-performing plant identification

## Analysis Insights

The analysis identifies several categories of plants:

1. **Profitable Plants**: Build cost below grid price (positive efficiency gap)
2. **Marginal Plants**: Build cost slightly above grid price
3. **Zombie Plants**: Zero generation output with positive costs
4. **High-Loss Plants**: Significant operational losses requiring intervention

State-level regulations are integrated to assess compliance exposure and potential penalties for emission threshold violations.

## Scenario Modeling

The dashboard supports three types of what-if analyses:

1. **Gas Price Shocks**: Model impacts of fuel cost changes on portfolio profitability
2. **Grid Price Adjustments**: Assess revenue impacts from market price fluctuations
3. **Efficiency Improvements**: Evaluate ROI of operational optimization investments

Adjustments are applied globally across the portfolio with real-time KPI updates.

## Regulatory Compliance

The dashboard integrates state emission regulations including:
- Main regulations and rules
- Emission thresholds (tCO2)
- Penalties per ton of excess emissions
- Coverage criteria and key standards
- Compliance cost estimates

This information helps contextualize plant performance within the regulatory framework.

## Future Enhancements

Potential extensions to this analysis:
- Renewable energy integration scenarios
- Carbon capture cost-benefit analysis
- Long-term fuel price forecasting
- Retirement vs. retrofit decision modeling
- Portfolio optimization recommendations

## Data Sources

- U.S. Energy Information Administration (EIA)
- EIA Form 923 (2023 data)
- State regulatory databases
- Industrial electricity pricing reports

## License

This project is for analytical and educational purposes.
